//go:build windows || darwin

package updater

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func configureManifestTest(t *testing.T, manifestURL string) {
	t.Helper()
	oldManifestURL := UpdateManifestURL
	oldSource := UpdateSource
	oldChannel := UpdateChannel
	oldRepository := UpdateRepository
	oldMarketing := CurrentMarketingVersion
	oldBuild := CurrentBuildVersion
	oldBundleID := ExpectedBundleID
	oldTeamID := ExpectedAppleTeamID
	oldInsecure := allowInsecureUpdateURLs
	t.Cleanup(func() {
		UpdateManifestURL = oldManifestURL
		UpdateSource = oldSource
		UpdateChannel = oldChannel
		UpdateRepository = oldRepository
		CurrentMarketingVersion = oldMarketing
		CurrentBuildVersion = oldBuild
		ExpectedBundleID = oldBundleID
		ExpectedAppleTeamID = oldTeamID
		allowInsecureUpdateURLs = oldInsecure
	})
	UpdateManifestURL = manifestURL
	UpdateSource = MLXPreviewSource
	UpdateChannel = "preview"
	UpdateRepository = "pd95/ollama"
	CurrentMarketingVersion = "0.34.1"
	CurrentBuildVersion = "34.1.1"
	ExpectedBundleID = "com.electron.ollama"
	ExpectedAppleTeamID = "P8CC95REUG"
	allowInsecureUpdateURLs = true
}

func manifestJSON(marketing, build, assetURL string, mutate ...func(*updateManifest)) string {
	releaseTag := "preview/mlx-" + marketing + "-test"
	manifest := updateManifest{
		SchemaVersion:    1,
		Source:           MLXPreviewSource,
		Channel:          "preview",
		Repository:       "pd95/ollama",
		MarketingVersion: marketing,
		BuildVersion:     build,
		ReleaseTag:       releaseTag,
		ReleasePageURL:   "https://github.com/pd95/ollama/releases/tag/" + strings.ReplaceAll(releaseTag, "/", "%2F"),
		Platform:         runtime.GOOS,
		Architecture:     runtime.GOARCH,
		Asset: updateManifestAsset{
			Name:      Installer,
			URL:       assetURL,
			SizeBytes: 7,
			SHA256:    strings.Repeat("a", 64),
		},
	}
	for _, fn := range mutate {
		fn(&manifest)
	}
	return fmt.Sprintf(`{"schema_version":%d,"source":%q,"channel":%q,"repository":%q,"marketing_version":%q,"build_version":%q,"release_tag":%q,"release_page_url":%q,"platform":%q,"architecture":%q,"asset":{"name":%q,"url":%q,"size_bytes":%d,"sha256":%q}}`,
		manifest.SchemaVersion, manifest.Source, manifest.Channel, manifest.Repository,
		manifest.MarketingVersion, manifest.BuildVersion, manifest.ReleaseTag,
		manifest.ReleasePageURL, manifest.Platform, manifest.Architecture,
		manifest.Asset.Name, manifest.Asset.URL, manifest.Asset.SizeBytes,
		manifest.Asset.SHA256)
}

func TestCompareUpdateVersions(t *testing.T) {
	for _, test := range []struct {
		name                           string
		marketing, build               string
		currentMarketing, currentBuild string
		want                           int
	}{
		{"new marketing", "0.35.0", "35.0.1", "0.34.9", "34.9.9", 1},
		{"new build", "0.34.1", "34.1.2", "0.34.1", "34.1.1", 1},
		{"same", "0.34.1", "34.1.1", "0.34.1", "34.1.1", 0},
		{"older marketing", "0.34.0", "34.0.9", "0.34.1", "34.1.1", -1},
		{"older build", "0.34.1", "34.1.0", "0.34.1", "34.1.1", -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := compareUpdateVersions(test.marketing, test.build, test.currentMarketing, test.currentBuild)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("comparison = %d, want %d", got, test.want)
			}
		})
	}
}

func TestCustomManifestSelection(t *testing.T) {
	var body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()
	configureManifestTest(t, server.URL)

	for _, test := range []struct {
		name      string
		marketing string
		build     string
		available bool
	}{
		{"new marketing", "0.35.0", "35.0.1", true},
		{"new build", "0.34.1", "34.1.2", true},
		{"same", "0.34.1", "34.1.1", false},
		{"older", "0.34.0", "34.0.9", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			body = manifestJSON(test.marketing, test.build, server.URL+"/"+Installer)
			available, candidate, err := (&Updater{}).checkForUpdate(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if available != test.available {
				t.Fatalf("available = %v, want %v", available, test.available)
			}
			if available && (candidate.Source != MLXPreviewSource || candidate.BuildVersion != test.build) {
				t.Fatalf("candidate = %+v", candidate)
			}
		})
	}
}

func TestCustomManifestRejectsMalformedOrWrongIdentity(t *testing.T) {
	var body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()
	configureManifestTest(t, server.URL)

	for _, test := range []struct {
		name string
		body func() string
		want string
	}{
		{"malformed", func() string { return `{` }, "decode"},
		{"unknown field", func() string { return `{"schema_version":1,"unexpected":true}` }, "unknown field"},
		{"wrong asset", func() string {
			return manifestJSON("0.35.0", "35.0.1", server.URL+"/wrong.zip", func(m *updateManifest) { m.Asset.Name = "wrong.zip" })
		}, "unexpected update asset"},
		{"wrong architecture", func() string {
			return manifestJSON("0.35.0", "35.0.1", server.URL+"/"+Installer, func(m *updateManifest) { m.Architecture = "wrong" })
		}, "targets"},
		{"wrong source", func() string {
			return manifestJSON("0.35.0", "35.0.1", server.URL+"/"+Installer, func(m *updateManifest) { m.Source = OfficialUpdateSource })
		}, "does not match"},
	} {
		t.Run(test.name, func(t *testing.T) {
			body = test.body()
			_, _, err := (&Updater{}).checkForUpdate(t.Context())
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestCustomManifestEndpointFailures(t *testing.T) {
	t.Run("rate limited", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
		}))
		defer server.Close()
		configureManifestTest(t, server.URL)
		_, _, err := (&Updater{}).checkForUpdate(t.Context())
		if err == nil || !strings.Contains(err.Error(), "status 429") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("unavailable", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
		}))
		configureManifestTest(t, server.URL)
		server.Close()
		_, _, err := (&Updater{}).checkForUpdate(t.Context())
		if err == nil || !strings.Contains(err.Error(), "check MLX preview update") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("unexpected redirect host", func(t *testing.T) {
		target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		defer target.Close()
		server := httptest.NewServer(http.RedirectHandler(target.URL, http.StatusFound))
		defer server.Close()
		configureManifestTest(t, server.URL)
		_, _, err := (&Updater{}).checkForUpdate(t.Context())
		if err == nil || !strings.Contains(err.Error(), "redirect host") {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestOfficialAssetURLPolicy(t *testing.T) {
	oldInsecure := allowInsecureUpdateURLs
	allowInsecureUpdateURLs = false
	t.Cleanup(func() { allowInsecureUpdateURLs = oldInsecure })

	for _, test := range []struct {
		name string
		url  string
		ok   bool
	}{
		{"release asset", "https://github.com/ollama/ollama/releases/download/v0.34.2/Ollama-darwin.zip", true},
		{"http", "http://github.com/ollama/ollama/releases/download/v0.34.2/Ollama-darwin.zip", false},
		{"wrong repository", "https://github.com/example/ollama/releases/download/v0.34.2/Ollama-darwin.zip", false},
		{"wrong asset", "https://github.com/ollama/ollama/releases/download/v0.34.2/Ollama-linux.tgz", false},
		{"credentials", "https://token@github.com/ollama/ollama/releases/download/v0.34.2/Ollama-darwin.zip", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			parsed, err := url.Parse(test.url)
			if err != nil {
				t.Fatal(err)
			}
			err = validateOfficialAssetURL(parsed)
			if (err == nil) != test.ok {
				t.Fatalf("validateOfficialAssetURL() error = %v, want valid %v", err, test.ok)
			}
		})
	}
}

func TestOfficialReplacementSelection(t *testing.T) {
	if !OfficialReplacementAvailable() {
		t.Skip("official replacement is available only on pinned darwin-arm64 builds")
	}
	oldURL := UpdateCheckURLBase
	oldManifestURL := UpdateManifestURL
	oldSource := UpdateSource
	oldMarketing := CurrentMarketingVersion
	oldInsecure := allowInsecureUpdateURLs
	t.Cleanup(func() {
		UpdateCheckURLBase = oldURL
		UpdateManifestURL = oldManifestURL
		UpdateSource = oldSource
		CurrentMarketingVersion = oldMarketing
		allowInsecureUpdateURLs = oldInsecure
	})
	UpdateManifestURL = "https://updates.example.invalid/manifest.json"
	UpdateSource = MLXPreviewSource
	CurrentMarketingVersion = "0.34.1"
	allowInsecureUpdateURLs = true

	var release = "v0.34.2"
	status := http.StatusOK
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if status != http.StatusOK {
			w.WriteHeader(status)
			return
		}
		_, _ = fmt.Fprintf(w, `{"url":%q}`, server.URL+"/ollama/ollama/releases/download/"+release+"/Ollama-darwin.zip")
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL

	available, candidate, err := (&Updater{}).checkForUpdateSource(t.Context(), OfficialUpdateSource)
	if err != nil {
		t.Fatal(err)
	}
	if !available || candidate.Source != OfficialUpdateSource || candidate.MarketingVersion != "0.34.2" || candidate.Architecture != "arm64" {
		t.Fatalf("official candidate = %+v, available %v", candidate, available)
	}

	for _, sameOrOlder := range []string{"v0.34.1", "v0.34.0"} {
		release = sameOrOlder
		available, _, err = (&Updater{}).checkForUpdateSource(t.Context(), OfficialUpdateSource)
		if err != nil {
			t.Fatal(err)
		}
		if available {
			t.Fatalf("official release %s must not replace 0.34.1", sameOrOlder)
		}
	}

	status = http.StatusTooManyRequests
	_, _, err = (&Updater{}).checkForUpdateSource(t.Context(), OfficialUpdateSource)
	if err == nil || !strings.Contains(err.Error(), "status 429") {
		t.Fatalf("rate-limited official service error = %v", err)
	}
}

func TestSourceQualifiedStagingKeepsBothCandidates(t *testing.T) {
	oldStageDir := UpdateStageDir
	oldManifestURL := UpdateManifestURL
	oldSource := UpdateSource
	oldVerifyCandidate := VerifyCandidate
	UpdateStageDir = t.TempDir()
	UpdateManifestURL = "https://updates.example.invalid/manifest.json"
	UpdateSource = MLXPreviewSource
	VerifyCandidate = func(string, UpdateResponse) error { return nil }
	t.Cleanup(func() {
		UpdateStageDir = oldStageDir
		UpdateManifestURL = oldManifestURL
		UpdateSource = oldSource
		VerifyCandidate = oldVerifyCandidate
	})

	for _, candidate := range []UpdateResponse{
		{Source: MLXPreviewSource, UpdateVersion: "0.34.1", MarketingVersion: "0.34.1", BuildVersion: "34.1.2", AssetName: Installer},
		{Source: OfficialUpdateSource, UpdateVersion: "v0.34.2", MarketingVersion: "0.34.2", BuildVersion: "0.34.2"},
	} {
		dir := filepath.Join(updateSourceStageDir(candidate.Source), "verified")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		bundle := filepath.Join(dir, Installer)
		payload := []byte(candidate.Source)
		if candidate.Source == MLXPreviewSource {
			sum := sha256.Sum256(payload)
			candidate.SHA256 = hex.EncodeToString(sum[:])
		}
		if err := os.WriteFile(bundle, payload, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := writeUpdateCandidateMetadata(bundle, candidate); err != nil {
			t.Fatal(err)
		}
	}

	for _, source := range []string{MLXPreviewSource, OfficialUpdateSource} {
		staged, ok := StagedUpdateForSource(source)
		if !ok || staged.Source != source {
			t.Fatalf("staged %s update = %+v, %v", source, staged, ok)
		}
	}
}

func TestCustomBackgroundCheckerDownloadsButDoesNotAuthorizeStartupInstall(t *testing.T) {
	payload := []byte("archive")
	sum := sha256.Sum256(payload)
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/manifest" {
			_, _ = w.Write([]byte(manifestJSON("0.35.0", "35.0.1", server.URL+"/"+Installer, func(manifest *updateManifest) {
				manifest.Asset.SizeBytes = int64(len(payload))
				manifest.Asset.SHA256 = hex.EncodeToString(sum[:])
			})))
			return
		}
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	configureManifestTest(t, server.URL+"/manifest")

	oldStageDir := UpdateStageDir
	oldVerifyCandidate := VerifyCandidate
	oldDisableUpdates := DisableUpdates
	oldInitialDelay := UpdateCheckInitialDelay
	oldInterval := UpdateCheckInterval
	oldDownloaded := UpdateDownloaded
	t.Cleanup(func() {
		UpdateStageDir = oldStageDir
		VerifyCandidate = oldVerifyCandidate
		DisableUpdates = oldDisableUpdates
		UpdateCheckInitialDelay = oldInitialDelay
		UpdateCheckInterval = oldInterval
		UpdateDownloaded = oldDownloaded
	})
	UpdateStageDir = t.TempDir()
	VerifyCandidate = func(string, UpdateResponse) error { return nil }
	DisableUpdates = "false"
	UpdateCheckInitialDelay = time.Millisecond
	UpdateCheckInterval = time.Hour
	UpdateDownloaded = false

	st := &store.Store{DBPath: filepath.Join(t.TempDir(), "settings.db")}
	defer st.Close()
	settings, err := st.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := st.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	callback := make(chan string, 1)
	done := (&Updater{Store: st}).startBackgroundUpdaterChecker(ctx, func(version string) error {
		callback <- version
		return nil
	})
	select {
	case version := <-callback:
		if version != "0.35.0" {
			t.Fatalf("callback version = %q", version)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("background checker did not download the custom update")
	}
	if AutomaticInstallAllowed() {
		t.Fatal("custom source must require an explicit restart action")
	}
	if staged, ok := StagedUpdate(); !ok || staged.Source != MLXPreviewSource {
		t.Fatalf("staged update = %+v, %v", staged, ok)
	}
	cancel()
	<-done
}

func TestCustomDownloadStagesVerifiedCandidate(t *testing.T) {
	payload := []byte("archive")
	sum := sha256.Sum256(payload)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("ETag", `"candidate"`)
		w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, Installer))
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	configureManifestTest(t, server.URL)
	oldStageDir := UpdateStageDir
	oldVerifyCandidate := VerifyCandidate
	oldDownloaded := UpdateDownloaded
	t.Cleanup(func() {
		UpdateStageDir = oldStageDir
		VerifyCandidate = oldVerifyCandidate
		UpdateDownloaded = oldDownloaded
	})
	UpdateStageDir = t.TempDir()
	verified := false
	VerifyCandidate = func(path string, candidate UpdateResponse) error {
		verified = true
		if candidate.Source != MLXPreviewSource {
			return fmt.Errorf("wrong candidate source %q", candidate.Source)
		}
		return nil
	}
	candidate := UpdateResponse{
		Source: MLXPreviewSource, Channel: "preview", Repository: "pd95/ollama",
		MarketingVersion: "0.35.0", BuildVersion: "35.0.1",
		UpdateVersion: "0.35.0", AssetName: Installer,
		UpdateURL: server.URL, ExpectedSize: int64(len(payload)),
		SHA256: hex.EncodeToString(sum[:]), Architecture: runtime.GOARCH,
	}
	if err := (&Updater{}).DownloadNewRelease(t.Context(), candidate); err != nil {
		t.Fatal(err)
	}
	if !verified {
		t.Fatal("candidate verifier was not called")
	}
	staged, ok := StagedUpdate()
	if !ok || staged.Source != MLXPreviewSource || staged.BuildVersion != "35.0.1" {
		t.Fatalf("staged update = %+v, %v", staged, ok)
	}
}

func TestCustomDownloadRejectsChecksumSizeAndRedirect(t *testing.T) {
	payload := []byte("archive")
	oldStageDir := UpdateStageDir
	oldVerifyCandidate := VerifyCandidate
	oldAttempts := UpdateDownloadAttempts
	oldRetryWait := UpdateDownloadRetryWait
	t.Cleanup(func() {
		UpdateStageDir = oldStageDir
		VerifyCandidate = oldVerifyCandidate
		UpdateDownloadAttempts = oldAttempts
		UpdateDownloadRetryWait = oldRetryWait
	})
	UpdateDownloadAttempts = 1
	VerifyCandidate = func(string, UpdateResponse) error {
		return errors.New("verifier should not run")
	}

	t.Run("checksum", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(payload) }))
		defer server.Close()
		configureManifestTest(t, server.URL)
		UpdateStageDir = t.TempDir()
		candidate := UpdateResponse{Source: MLXPreviewSource, UpdateVersion: "0.35.0", BuildVersion: "35.0.1", AssetName: Installer, UpdateURL: server.URL, ExpectedSize: int64(len(payload)), SHA256: strings.Repeat("0", 64)}
		err := (&Updater{}).DownloadNewRelease(t.Context(), candidate)
		if err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("size", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(payload) }))
		defer server.Close()
		configureManifestTest(t, server.URL)
		UpdateStageDir = t.TempDir()
		candidate := UpdateResponse{Source: MLXPreviewSource, UpdateVersion: "0.35.0", BuildVersion: "35.0.1", AssetName: Installer, UpdateURL: server.URL, ExpectedSize: int64(len(payload) + 1), SHA256: strings.Repeat("0", 64)}
		err := (&Updater{}).DownloadNewRelease(t.Context(), candidate)
		if err == nil || !strings.Contains(err.Error(), "does not match manifest size") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("redirect", func(t *testing.T) {
		target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		defer target.Close()
		server := httptest.NewServer(http.RedirectHandler(target.URL, http.StatusFound))
		defer server.Close()
		configureManifestTest(t, server.URL)
		UpdateStageDir = t.TempDir()
		candidate := UpdateResponse{Source: MLXPreviewSource, UpdateVersion: "0.35.0", BuildVersion: "35.0.1", AssetName: Installer, UpdateURL: server.URL, ExpectedSize: int64(len(payload)), SHA256: strings.Repeat("0", 64)}
		err := (&Updater{}).DownloadNewRelease(t.Context(), candidate)
		if err == nil || !strings.Contains(err.Error(), "redirect host") {
			t.Fatalf("error = %v", err)
		}
	})

	entries, err := os.ReadDir(UpdateStageDir)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) == ".zip" {
			t.Fatalf("failed download staged %s", entry.Name())
		}
	}
}
