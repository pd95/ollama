//go:build windows || darwin

package updater

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestUpdateStagePathRejectsUnsafeFilename(t *testing.T) {
	stageDir := t.TempDir()
	for _, tt := range []struct {
		name     string
		filename string
	}{
		{"empty", ""},
		{"dot", "."},
		{"dotdot", ".."},
		{"posix_parent", "../OllamaSetup.exe"},
		{"windows_parent", `..\OllamaSetup.exe`},
		{"posix_absolute_tmp", "/tmp/OllamaSetup.exe"},
		{"darwin_absolute_app", "/Applications/Ollama.app"},
		{"darwin_bundle_path", "Ollama.app/Contents/MacOS/Ollama"},
		{"darwin_user_download", "~/Downloads/Ollama-darwin.zip"},
		{"windows_absolute", `C:\Users\Public\OllamaSetup.exe`},
		{"colon", "Ollama:Setup.exe"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := updateStagePath(stageDir, "etag", tt.filename); err == nil {
				t.Fatal("expected unsafe filename to be rejected")
			}
		})
	}
}

func TestUpdateStagePathHashesETag(t *testing.T) {
	stageDir := t.TempDir()
	stageFilename, err := updateStagePath(stageDir, `../escaped`, "OllamaSetup.exe")
	if err != nil {
		t.Fatal(err)
	}

	rel, err := filepath.Rel(stageDir, stageFilename)
	if err != nil {
		t.Fatal(err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		t.Fatalf("stage filename escaped stage dir: %s", stageFilename)
	}
	etagDir := filepath.Base(filepath.Dir(stageFilename))
	if etagDir == ".." || etagDir == "escaped" || strings.ContainsAny(etagDir, `/\`) {
		t.Fatalf("stage filename used raw etag path component: %s", stageFilename)
	}
}

func TestIsNewReleaseAvailable(t *testing.T) {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
			// TODO - wire up the redirects to mimic real behavior
		} else {
			slog.Debug("unexpected request", "url", r.URL)
		}
	}))
	defer server.Close()
	slog.Debug("server", "url", server.URL)

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close() // Ensure database is closed
	UpdateCheckURLBase = server.URL + "/update.json"
	updatePresent, resp, err := updater.checkForUpdate(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if !updatePresent {
		t.Fatal("expected update to be available")
	}
	if resp.UpdateVersion != "9.9.9" {
		t.Fatal("unexpected response", "url", resp.UpdateURL, "version", resp.UpdateVersion)
	}
}

func TestAutomaticUpdatesDisabledSkipsStartupCheckButAllowsManualDownload(t *testing.T) {
	oldDisableUpdates := DisableUpdates
	oldUpdateDownloaded := UpdateDownloaded
	oldUpdateCheckURLBase := UpdateCheckURLBase
	oldUpdateCheckInitialDelay := UpdateCheckInitialDelay
	oldUpdateCheckInterval := UpdateCheckInterval
	oldUpdateStageDir := UpdateStageDir
	oldVerifyDownload := VerifyDownload
	defer func() {
		DisableUpdates = oldDisableUpdates
		UpdateDownloaded = oldUpdateDownloaded
		UpdateCheckURLBase = oldUpdateCheckURLBase
		UpdateCheckInitialDelay = oldUpdateCheckInitialDelay
		UpdateCheckInterval = oldUpdateCheckInterval
		UpdateStageDir = oldUpdateStageDir
		VerifyDownload = oldVerifyDownload
	}()

	DisableUpdates = "true"
	UpdateDownloaded = false
	UpdateCheckInitialDelay = time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	UpdateStageDir = t.TempDir()
	VerifyDownload = func(_ string) error { return nil }

	checkCount := atomic.Int32{}
	downloadCount := atomic.Int32{}
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/update.json":
			checkCount.Add(1)
			w.Write([]byte(fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`, server.URL+"/9.9.9/"+Installer)))
		case "/9.9.9/" + Installer:
			downloadCount.Add(1)
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()
	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	updater.StartBackgroundUpdaterChecker(ctx, func(string) error {
		t.Fatal("preview background checker must not activate the tray")
		return nil
	})
	time.Sleep(20 * time.Millisecond)

	if checkCount.Load() != 0 {
		t.Fatalf("automatic updates should not check on startup, got %d requests", checkCount.Load())
	}
	updater.TriggerImmediateCheck()
	time.Sleep(20 * time.Millisecond)
	if checkCount.Load() != 0 {
		t.Fatalf("background triggers should not bypass preview policy, got %d requests", checkCount.Load())
	}

	result, err := updater.CheckForUpdates(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "ready" || result.Version != "9.9.9" {
		t.Fatalf("manual update result = %+v, want ready 9.9.9", result)
	}

	if downloadCount.Load() == 0 {
		t.Fatal("manual update check should download in release build")
	}
	if !UpdateDownloaded {
		t.Fatal("manual update check should mark the update as downloaded")
	}
	if _, err := os.Stat(getStagedUpdate()); err != nil {
		t.Fatalf("manual update check should stage an update bundle: %v", err)
	}
	staged, ok := StagedUpdate()
	if !ok || staged.Version != "9.9.9" {
		t.Fatalf("staged update = %+v, %v; want durable version 9.9.9", staged, ok)
	}
}

func TestManualCheckUpToDate(t *testing.T) {
	oldURL := UpdateCheckURLBase
	defer func() { UpdateCheckURLBase = oldURL }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL

	result, err := (&Updater{}).CheckForUpdates(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "up_to_date" || result.Version != "" {
		t.Fatalf("result = %+v, want up_to_date", result)
	}
}

func TestManualCheckNetworkFailure(t *testing.T) {
	oldURL := UpdateCheckURLBase
	defer func() { UpdateCheckURLBase = oldURL }()

	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	UpdateCheckURLBase = server.URL
	server.Close()

	_, err := (&Updater{}).CheckForUpdates(t.Context())
	if err == nil || !strings.Contains(err.Error(), "check for update") {
		t.Fatalf("error = %v, want network check failure", err)
	}
}

func TestManualCheckFailuresDoNotStageUpdate(t *testing.T) {
	oldURL := UpdateCheckURLBase
	oldStageDir := UpdateStageDir
	oldVerify := VerifyDownload
	oldDownloaded := UpdateDownloaded
	oldRetryWait := UpdateDownloadRetryWait
	defer func() {
		UpdateCheckURLBase = oldURL
		UpdateStageDir = oldStageDir
		VerifyDownload = oldVerify
		UpdateDownloaded = oldDownloaded
		UpdateDownloadRetryWait = oldRetryWait
	}()
	UpdateDownloadRetryWait = time.Millisecond

	for _, tt := range []struct {
		name      string
		checkBody string
		checkCode int
		download  func(http.ResponseWriter, *http.Request)
		verify    func(string) error
		wantError string
	}{
		{
			name:      "malformed metadata",
			checkBody: `{not-json`,
			checkCode: http.StatusOK,
			wantError: "decode update response",
		},
		{
			name:      "update service failure",
			checkBody: `unavailable`,
			checkCode: http.StatusServiceUnavailable,
			wantError: "status 503",
		},
		{
			name:      "download failure",
			checkCode: http.StatusOK,
			download: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusBadGateway)
			},
			verify:    func(string) error { return nil },
			wantError: "update download returned status 502",
		},
		{
			name:      "verification failure",
			checkCode: http.StatusOK,
			download: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("ETag", `"test"`)
				w.WriteHeader(http.StatusOK)
				if r.Method == http.MethodGet {
					_, _ = w.Write([]byte("archive"))
				}
			},
			verify:    func(string) error { return errors.New("signature mismatch") },
			wantError: "signature mismatch",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			UpdateStageDir = t.TempDir()
			UpdateDownloaded = false
			VerifyDownload = tt.verify

			var server *httptest.Server
			server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/update" {
					w.WriteHeader(tt.checkCode)
					if tt.checkCode == http.StatusOK && tt.checkBody == "" {
						_, _ = fmt.Fprintf(w, `{"url":%q}`, server.URL+"/v9.9.9/"+Installer)
					} else {
						_, _ = io.WriteString(w, tt.checkBody)
					}
					return
				}
				if tt.download != nil {
					tt.download(w, r)
					return
				}
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()
			UpdateCheckURLBase = server.URL + "/update"

			_, err := (&Updater{}).CheckForUpdates(t.Context())
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("error = %v, want %q", err, tt.wantError)
			}
			if UpdateDownloaded {
				t.Fatal("failed manual check must not mark an update ready")
			}
			if getStagedUpdate() != "" {
				t.Fatal("failed manual check must not leave a staged update")
			}
		})
	}
}

func TestConcurrentManualChecksAreSerialized(t *testing.T) {
	oldURL := UpdateCheckURLBase
	defer func() { UpdateCheckURLBase = oldURL }()

	var active atomic.Int32
	var maximum atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		current := active.Add(1)
		defer active.Add(-1)
		for {
			observed := maximum.Load()
			if current <= observed || maximum.CompareAndSwap(observed, current) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL

	upd := &Updater{}
	results := make(chan error, 2)
	for range 2 {
		go func() {
			_, err := upd.CheckForUpdates(t.Context())
			results <- err
		}()
	}
	for range 2 {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	if maximum.Load() != 1 {
		t.Fatalf("maximum concurrent update requests = %d, want 1", maximum.Load())
	}
}

func TestDownloadNewReleaseRejectsUnsafeHeaderFilename(t *testing.T) {
	UpdateStageDir = t.TempDir()
	oldInstaller := Installer
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	defer func() {
		Installer = oldInstaller
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
	}()
	Installer = "OllamaSetup.exe"
	UpdateDownloaded = false
	VerifyDownload = func(_ string) error {
		t.Fatal("verification should not run for rejected downloads")
		return nil
	}

	var requestCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requestCount.Add(1)
		w.Header().Set("ETag", `"safe"`)
		w.Header().Set("Content-Disposition", `attachment; filename="../OllamaSetup.exe"`)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	updater := &Updater{}
	err := updater.DownloadNewRelease(t.Context(), UpdateResponse{UpdateURL: server.URL + "/download"})
	if err == nil || !strings.Contains(err.Error(), "unsafe update filename") {
		t.Fatalf("expected unsafe filename error, got %v", err)
	}
	if requestCount.Load() != 1 {
		t.Fatalf("request count = %d, want one GET before rejecting its response metadata", requestCount.Load())
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(UpdateStageDir), "OllamaSetup.exe")); err == nil {
		t.Fatal("download escaped update stage dir")
	}
}

func TestDownloadNewReleaseResumesInterruptedTransfer(t *testing.T) {
	oldStageDir := UpdateStageDir
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	oldAttempts := UpdateDownloadAttempts
	oldRetryWait := UpdateDownloadRetryWait
	defer func() {
		UpdateStageDir = oldStageDir
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
		UpdateDownloadAttempts = oldAttempts
		UpdateDownloadRetryWait = oldRetryWait
	}()
	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	UpdateDownloadAttempts = 3
	UpdateDownloadRetryWait = time.Millisecond

	payload := []byte("complete archive payload")
	cut := 8
	var requestCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		request := requestCount.Add(1)
		w.Header().Set("ETag", `"resumable"`)
		w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, Installer))
		if request == 1 {
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(payload[:cut])
			return
		}
		if got, want := r.Header.Get("Range"), fmt.Sprintf("bytes=%d-", cut); got != want {
			t.Errorf("Range = %q, want %q", got, want)
		}
		if got := r.Header.Get("If-Range"); got != `"resumable"` {
			t.Errorf("If-Range = %q, want resumable ETag", got)
		}
		if got := r.Header.Get("Accept-Encoding"); got != "identity" {
			t.Errorf("Accept-Encoding = %q, want identity", got)
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", cut, len(payload)-1, len(payload)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(payload[cut:])
	}))
	defer server.Close()

	VerifyDownload = func(bundle string) error {
		got, err := os.ReadFile(bundle)
		if err != nil {
			return err
		}
		if !bytes.Equal(got, payload) {
			return fmt.Errorf("downloaded payload = %q, want %q", got, payload)
		}
		return nil
	}

	err := (&Updater{}).DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL:     server.URL + "/download",
		UpdateVersion: "v9.9.9",
	})
	if err != nil {
		t.Fatal(err)
	}
	if requestCount.Load() != 2 {
		t.Fatalf("request count = %d, want initial GET and one ranged retry", requestCount.Load())
	}
	staged, ok := StagedUpdate()
	if !ok || staged.Version != "v9.9.9" {
		t.Fatalf("staged update = %+v, %v; want resumed v9.9.9 archive", staged, ok)
	}
}

func TestDownloadNewReleaseDoesNotRelabelOlderStagedUpdate(t *testing.T) {
	oldStageDir := UpdateStageDir
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	defer func() {
		UpdateStageDir = oldStageDir
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
	}()
	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	VerifyDownload = func(string) error { return nil }

	oldBundle, err := updateStagePath(UpdateStageDir, `"old"`, Installer)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(oldBundle), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(oldBundle, []byte("old archive"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeUpdateMetadata(oldBundle, "v1"); err != nil {
		t.Fatal(err)
	}

	payload := []byte("new archive")
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.Header().Set("ETag", `"new"`)
		w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, Installer))
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	if err := (&Updater{}).DownloadNewRelease(t.Context(), UpdateResponse{
		UpdateURL: server.URL, UpdateVersion: "v2",
	}); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 1 {
		t.Fatalf("download requests = %d, want 1 for newer release", requests.Load())
	}
	staged, ok := StagedUpdate()
	if !ok || staged.Version != "v2" {
		t.Fatalf("staged update = %+v, %v; want v2", staged, ok)
	}
	got, err := os.ReadFile(getStagedUpdate())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("staged payload = %q, want new archive", got)
	}
}

func TestDownloadNewReleaseRestartsAfterInvalidContinuation(t *testing.T) {
	oldStageDir := UpdateStageDir
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	oldRetryWait := UpdateDownloadRetryWait
	defer func() {
		UpdateStageDir = oldStageDir
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
		UpdateDownloadRetryWait = oldRetryWait
	}()
	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	UpdateDownloadRetryWait = time.Millisecond
	payload := []byte("complete archive payload")
	cut := 8
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := requests.Add(1)
		w.Header().Set("ETag", `"stable"`)
		w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, Installer))
		switch n {
		case 1:
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
			_, _ = w.Write(payload[:cut])
		case 2:
			w.Header().Set("Content-Range", fmt.Sprintf("bytes 0-%d/%d", len(payload)-cut-1, len(payload)))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(payload[cut:])
		default:
			if r.Header.Get("Range") != "" {
				t.Errorf("clean retry unexpectedly sent Range %q", r.Header.Get("Range"))
			}
			_, _ = w.Write(payload)
		}
	}))
	defer server.Close()
	VerifyDownload = func(bundle string) error {
		got, err := os.ReadFile(bundle)
		if err != nil {
			return err
		}
		if !bytes.Equal(got, payload) {
			return fmt.Errorf("payload = %q", got)
		}
		return nil
	}
	if err := (&Updater{}).DownloadNewRelease(t.Context(), UpdateResponse{UpdateURL: server.URL, UpdateVersion: "v2"}); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 3 {
		t.Fatalf("requests = %d, want interrupted, invalid continuation, and clean retry", requests.Load())
	}
}

func TestDownloadNewReleaseDoesNotUseRawETagAsPathComponent(t *testing.T) {
	UpdateStageDir = t.TempDir()
	oldInstaller := Installer
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	defer func() {
		Installer = oldInstaller
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
	}()
	Installer = "OllamaSetup.exe"
	UpdateDownloaded = false
	VerifyDownload = func(_ string) error {
		return nil
	}

	payload := []byte("payload")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"../escaped"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write(payload)
		}
	}))
	defer server.Close()

	updater := &Updater{}
	if err := updater.DownloadNewRelease(t.Context(), UpdateResponse{UpdateURL: server.URL + "/download"}); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(filepath.Join(filepath.Dir(UpdateStageDir), "escaped", Installer)); err == nil {
		t.Fatal("download escaped update stage dir via etag")
	}

	entries, err := os.ReadDir(UpdateStageDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected one staged update dir, got %d", len(entries))
	}
	stageFilename := filepath.Join(UpdateStageDir, entries[0].Name(), Installer)
	got, err := os.ReadFile(stageFilename)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatalf("unexpected staged payload %q", got)
	}
}

// stopChecker cancels the background update checker and waits for its
// goroutine to return. Tests must join it before returning: the goroutine
// reads package-level knobs (UpdateCheckURLBase, UpdateCheckInterval, ...)
// that the next test rewrites.
func stopChecker(t *testing.T, cancel context.CancelFunc, done <-chan struct{}) {
	t.Helper()
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Error("background update checker did not stop")
	}
}

// waitDownloadIdle blocks until no download is in flight, so staged-file
// handles close before t.TempDir cleanup removes the stage directory. After
// the context is cancelled a new download can't write, so reaching idle makes
// cleanup race-free.
func (u *Updater) waitDownloadIdle() {
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		u.cancelDownloadLock.Lock()
		idle := u.cancelDownload == nil
		u.cancelDownloadLock.Unlock()
		if idle {
			return
		}
		time.Sleep(time.Millisecond)
	}
}

func TestDownloadIsNotDiscoverableUntilVerified(t *testing.T) {
	oldStageDir := UpdateStageDir
	oldVerify := VerifyDownload
	oldDownloaded := UpdateDownloaded
	defer func() {
		UpdateStageDir = oldStageDir
		VerifyDownload = oldVerify
		UpdateDownloaded = oldDownloaded
	}()
	UpdateStageDir = t.TempDir()
	UpdateDownloaded = false
	verificationStarted := make(chan string, 1)
	releaseVerification := make(chan struct{})
	VerifyDownload = func(bundle string) error {
		verificationStarted <- bundle
		<-releaseVerification
		return nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"atomic"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte("archive"))
		}
	}))
	defer server.Close()

	upd := &Updater{}
	done := make(chan error, 1)
	go func() {
		done <- upd.DownloadNewRelease(t.Context(), UpdateResponse{
			UpdateURL:     server.URL + "/download",
			UpdateVersion: "v9.9.9",
		})
	}()
	partial := <-verificationStarted
	if !strings.HasSuffix(partial, ".partial") {
		t.Fatalf("verification path = %q, want non-discoverable partial", partial)
	}
	if _, ok := StagedUpdate(); ok {
		t.Fatal("partial download became visible as a staged update")
	}
	if _, err := os.Stat(partial); err != nil {
		t.Fatalf("Settings discovery removed partial download: %v", err)
	}
	close(releaseVerification)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if staged, ok := StagedUpdate(); !ok || staged.Version != "v9.9.9" {
		t.Fatalf("published staged update = %+v, %v", staged, ok)
	}
}

func TestStagedUpdateCachesVerificationAndToleratesMalformedMetadata(t *testing.T) {
	oldStageDir := UpdateStageDir
	oldVerify := VerifyDownload
	defer func() {
		UpdateStageDir = oldStageDir
		VerifyDownload = oldVerify
	}()
	UpdateStageDir = t.TempDir()
	bundleDir := filepath.Join(UpdateStageDir, "verified")
	if err := os.MkdirAll(bundleDir, 0o755); err != nil {
		t.Fatal(err)
	}
	bundle := filepath.Join(bundleDir, Installer)
	if err := os.WriteFile(bundle, []byte("archive"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bundleDir, updateMetadataFilename), []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	var verifyCount atomic.Int32
	VerifyDownload = func(got string) error {
		if got != bundle {
			t.Fatalf("verify path = %q, want %q", got, bundle)
		}
		verifyCount.Add(1)
		return nil
	}

	for range 2 {
		staged, ok := StagedUpdate()
		if !ok || staged.Version != "" {
			t.Fatalf("legacy staged update = %+v, %v", staged, ok)
		}
	}
	if verifyCount.Load() != 1 {
		t.Fatalf("verification count = %d, want 1", verifyCount.Load())
	}
}

func TestBackgroundCheckerSkipsAlreadyStagedETagDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	oldVerifyDownload := VerifyDownload
	oldUpdateDownloaded := UpdateDownloaded
	oldUpdateCheckInitialDelay := UpdateCheckInitialDelay
	oldUpdateCheckInterval := UpdateCheckInterval
	oldUpdateCheckURLBase := UpdateCheckURLBase
	defer func() {
		VerifyDownload = oldVerifyDownload
		UpdateDownloaded = oldUpdateDownloaded
		UpdateCheckInitialDelay = oldUpdateCheckInitialDelay
		UpdateCheckInterval = oldUpdateCheckInterval
		UpdateCheckURLBase = oldUpdateCheckURLBase
	}()
	UpdateDownloaded = false
	UpdateCheckInitialDelay = time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond

	var verifyCount atomic.Int32
	VerifyDownload = func(_ string) error {
		verifyCount.Add(1)
		return nil
	}

	getETag := `"download-response-etag"`
	payload := []byte("payload")
	var getCount atomic.Int32
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/update.json":
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		case "/9.9.9/" + Installer:
			w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, Installer))
			switch r.Method {
			case http.MethodGet:
				w.Header().Set("ETag", getETag)
				getCount.Add(1)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(payload)
			default:
				t.Errorf("unexpected request method %s", r.Method)
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
		default:
			t.Errorf("unexpected request path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()
	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	callbacks := make(chan string, 4)
	checkerDone := updater.startBackgroundUpdaterChecker(ctx, func(ver string) error {
		callbacks <- ver
		return nil
	})
	defer updater.waitDownloadIdle()
	defer stopChecker(t, cancel, checkerDone)

	for range 2 {
		select {
		case <-callbacks:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for repeated update checks")
		}
	}
	cancel()

	stageFilename, err := updateStagePath(UpdateStageDir, getETag, Installer)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(stageFilename)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatalf("unexpected staged payload %q", got)
	}

	if getCount.Load() != 1 {
		t.Fatalf("GET count = %d, want 1", getCount.Load())
	}
	if verifyCount.Load() != 1 {
		t.Fatalf("verification count = %d, want 1", verifyCount.Load())
	}
	if !UpdateDownloaded {
		t.Fatal("UpdateDownloaded should stay true for already staged update")
	}
}

func TestBackgoundChecker(t *testing.T) {
	UpdateStageDir = t.TempDir()
	haveUpdate := false
	verified := false
	// Buffered + non-blocking send: the checker keeps calling cb every
	// UpdateCheckInterval, and a blocking send would wedge its goroutine once
	// the test stops receiving.
	done := make(chan int, 1)
	cb := func(ver string) error {
		haveUpdate = true
		select {
		case done <- 0:
		default:
		}
		return nil
	}
	stallTimer := time.NewTimer(5 * time.Second)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	VerifyDownload = func(_ string) error {
		verified = true
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
			// TODO - wire up the redirects to mimic real behavior
		} else if r.URL.Path == "/9.9.9/"+Installer {
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		} else {
			slog.Debug("unexpected request", "url", r.URL)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	checkerDone := updater.startBackgroundUpdaterChecker(ctx, cb)
	defer updater.waitDownloadIdle()
	defer stopChecker(t, cancel, checkerDone)
	select {
	case <-stallTimer.C:
		t.Fatal("stalled")
	case <-done:
		if !haveUpdate {
			t.Fatal("no update received")
		}
		if !verified {
			t.Fatal("unverified")
		}
	}
}

func TestAutoUpdateDisabledSkipsDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	var downloadAttempted atomic.Bool
	var updateAvailable atomic.Bool

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	VerifyDownload = func(_ string) error {
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			downloadAttempted.Store(true)
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	// Ensure auto-update is disabled
	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	cb := func(ver string) error {
		updateAvailable.Store(true)
		return nil
	}

	checkerDone := updater.startBackgroundUpdaterChecker(ctx, cb)
	defer updater.waitDownloadIdle()
	defer stopChecker(t, cancel, checkerDone)
	time.Sleep(30 * time.Millisecond)

	if downloadAttempted.Load() {
		t.Fatal("download should not be attempted when auto-update is disabled")
	}
	if updateAvailable.Load() {
		t.Fatal("discovery without a staged archive should not activate the tray")
	}
}

func TestAutoUpdateReenabledDownloadsUpdate(t *testing.T) {
	UpdateStageDir = t.TempDir()
	var downloadAttempted atomic.Bool
	callbackCalled := make(chan struct{}, 1)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	VerifyDownload = func(_ string) error {
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			downloadAttempted.Store(true)
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	upd := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer upd.Store.Close()

	// Start with auto-update disabled
	settings, err := upd.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := upd.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	cb := func(ver string) error {
		select {
		case callbackCalled <- struct{}{}:
		default:
		}
		return nil
	}

	checkerDone := upd.startBackgroundUpdaterChecker(ctx, cb)
	defer upd.waitDownloadIdle()
	defer stopChecker(t, cancel, checkerDone)

	// Allow several checks while disabled; discovery must not activate the tray.
	time.Sleep(30 * time.Millisecond)
	if downloadAttempted.Load() {
		t.Fatal("download should not happen while auto-update is disabled")
	}
	select {
	case <-callbackCalled:
		t.Fatal("discovery without a staged archive should not activate the tray")
	default:
	}

	// Re-enable auto-update
	settings.AutoUpdateEnabled = true
	if err := upd.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	deadline := time.After(5 * time.Second)
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for !downloadAttempted.Load() {
		select {
		case <-deadline:
			t.Fatal("expected download after re-enabling auto-update")
		case <-ticker.C:
		}
	}
}

func TestCancelOngoingDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	downloadStarted := make(chan struct{})
	downloadCancelled := make(chan struct{})

	ctx := t.Context()
	VerifyDownload = func(_ string) error {
		return nil
	}

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			if r.Method == http.MethodHead {
				w.Header().Set("Content-Length", "1000000")
				w.WriteHeader(http.StatusOK)
				return
			}
			// Signal that download has started
			close(downloadStarted)
			// Wait for cancellation or timeout
			select {
			case <-r.Context().Done():
				close(downloadCancelled)
				return
			case <-time.After(5 * time.Second):
				t.Error("download was not cancelled in time")
			}
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	_, resp, err := updater.checkForUpdate(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// Start download in goroutine
	downloadDone := make(chan struct{})
	go func() {
		defer close(downloadDone)
		_ = updater.DownloadNewRelease(ctx, resp)
	}()

	// Wait for download to start
	select {
	case <-downloadStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("download did not start in time")
	}

	// Cancel the download
	updater.CancelOngoingDownload()

	// Verify cancellation was received
	select {
	case <-downloadCancelled:
		// Success
	case <-time.After(2 * time.Second):
		t.Fatal("download cancellation was not received by server")
	}

	// Wait for the download goroutine to unwind: it drags along a background
	// update-check loop that reads package-level knobs the next test rewrites.
	<-downloadDone
}

func TestTriggerImmediateCheck(t *testing.T) {
	UpdateStageDir = t.TempDir()
	checkCount := atomic.Int32{}
	checkDone := make(chan struct{}, 10)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	// Set a very long interval so only TriggerImmediateCheck causes checks
	UpdateCheckInitialDelay = 1 * time.Millisecond
	UpdateCheckInterval = 1 * time.Hour
	VerifyDownload = func(_ string) error {
		return nil
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			checkCount.Add(1)
			select {
			case checkDone <- struct{}{}:
			default:
			}
			// Return no update available
			w.WriteHeader(http.StatusNoContent)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	cb := func(ver string) error {
		return nil
	}

	checkerDone := updater.startBackgroundUpdaterChecker(ctx, cb)
	defer updater.waitDownloadIdle()
	defer stopChecker(t, cancel, checkerDone)

	// Wait for the initial check that fires after the initial delay
	select {
	case <-checkDone:
	case <-time.After(2 * time.Second):
		t.Fatal("initial check did not happen")
	}

	initialCount := checkCount.Load()

	// Trigger immediate check
	updater.TriggerImmediateCheck()

	// Wait for the triggered check
	select {
	case <-checkDone:
	case <-time.After(2 * time.Second):
		t.Fatal("triggered check did not happen")
	}

	finalCount := checkCount.Load()
	if finalCount <= initialCount {
		t.Fatalf("TriggerImmediateCheck did not cause additional check: initial=%d, final=%d", initialCount, finalCount)
	}
}
