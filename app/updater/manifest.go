//go:build windows || darwin

package updater

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"golang.org/x/mod/semver"
)

const (
	OfficialUpdateSource = "official"
	MLXPreviewSource     = "mlx-preview"

	maxManifestBytes = 64 << 10
	maxArchiveBytes  = 1 << 30
)

var (
	UpdateManifestURL           string
	UpdateSource                string
	UpdateChannel               string
	UpdateRepository            string
	CurrentMarketingVersion     string
	CurrentBuildVersion         string
	ExpectedBundleID            string
	ExpectedAppleTeamID         string
	ExpectedOfficialAppleTeamID = "3MU9H2V9Y9"

	// Tests use local HTTP servers. Production custom update URLs are always
	// required to use HTTPS.
	allowInsecureUpdateURLs bool
)

type updateManifest struct {
	SchemaVersion    int                 `json:"schema_version"`
	Source           string              `json:"source"`
	Channel          string              `json:"channel"`
	Repository       string              `json:"repository"`
	MarketingVersion string              `json:"marketing_version"`
	BuildVersion     string              `json:"build_version"`
	ReleaseTag       string              `json:"release_tag"`
	ReleasePageURL   string              `json:"release_page_url"`
	Platform         string              `json:"platform"`
	Architecture     string              `json:"architecture"`
	Asset            updateManifestAsset `json:"asset"`
}

type updateManifestAsset struct {
	Name      string `json:"name"`
	URL       string `json:"url"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256"`
}

var (
	marketingVersionPattern = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)
	buildVersionPattern     = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)
	bundleIDPattern         = regexp.MustCompile(`^[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)+$`)
	teamIDPattern           = regexp.MustCompile(`^[A-Z0-9]{10}$`)
	releaseTagPattern       = regexp.MustCompile(`^preview/mlx-[A-Za-z0-9._-]+$`)
)

func CustomUpdateSourceEnabled() bool {
	return strings.TrimSpace(UpdateManifestURL) != ""
}

func PrimaryUpdateSource() string {
	if CustomUpdateSourceEnabled() {
		return UpdateSource
	}
	return OfficialUpdateSource
}

func ValidUpdateSource(source string) bool {
	return source == OfficialUpdateSource || (source == MLXPreviewSource && CustomUpdateSourceEnabled())
}

func OfficialReplacementAvailable() bool {
	return runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" &&
		teamIDPattern.MatchString(ExpectedOfficialAppleTeamID)
}

// AutomaticInstallAllowed distinguishes the official updater's historical
// startup installation from the MLX distribution's download-only automation.
func AutomaticInstallAllowed() bool {
	return !CustomUpdateSourceEnabled()
}

func UpdateReleaseFallbackURL() string {
	if CustomUpdateSourceEnabled() {
		return "https://github.com/pd95/ollama/releases"
	}
	return ""
}

func validateCustomUpdateConfiguration() error {
	values := map[string]string{
		"source":                    UpdateSource,
		"channel":                   UpdateChannel,
		"repository":                UpdateRepository,
		"current marketing version": CurrentMarketingVersion,
		"current build version":     CurrentBuildVersion,
		"bundle identifier":         ExpectedBundleID,
		"Apple Team ID":             ExpectedAppleTeamID,
	}
	for name, value := range values {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("custom update configuration is missing %s", name)
		}
	}
	if UpdateSource != MLXPreviewSource {
		return fmt.Errorf("unsupported custom update source %q", UpdateSource)
	}
	if UpdateChannel != "preview" {
		return fmt.Errorf("unsupported custom update channel %q", UpdateChannel)
	}
	if UpdateRepository != "pd95/ollama" {
		return fmt.Errorf("unsupported custom update repository %q", UpdateRepository)
	}
	if !marketingVersionPattern.MatchString(CurrentMarketingVersion) {
		return fmt.Errorf("invalid current marketing version %q", CurrentMarketingVersion)
	}
	if _, err := parseBuildVersion(CurrentBuildVersion); err != nil {
		return fmt.Errorf("invalid current build version: %w", err)
	}
	if !bundleIDPattern.MatchString(ExpectedBundleID) {
		return fmt.Errorf("invalid expected bundle identifier %q", ExpectedBundleID)
	}
	if !teamIDPattern.MatchString(ExpectedAppleTeamID) {
		return fmt.Errorf("invalid expected Apple Team ID %q", ExpectedAppleTeamID)
	}
	manifestURL, err := url.Parse(UpdateManifestURL)
	if err != nil {
		return fmt.Errorf("parse custom update manifest URL: %w", err)
	}
	if err := validateManifestURL(manifestURL); err != nil {
		return err
	}
	return nil
}

func (u *Updater) checkCustomUpdate(ctx context.Context) (bool, UpdateResponse, error) {
	var candidate UpdateResponse
	if err := validateCustomUpdateConfiguration(); err != nil {
		return false, candidate, err
	}

	manifestURL, _ := url.Parse(UpdateManifestURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, manifestURL.String(), nil)
	if err != nil {
		return false, candidate, fmt.Errorf("create update manifest request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Cache-Control", "no-cache")
	client := &http.Client{
		Timeout: UpdateCheckTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return errors.New("too many update manifest redirects")
			}
			if req.URL.Host != manifestURL.Host {
				return fmt.Errorf("unexpected update manifest redirect host %q", req.URL.Host)
			}
			return validateManifestURL(req.URL)
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return false, candidate, fmt.Errorf("check MLX preview update: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 97))
		return false, candidate, fmt.Errorf("MLX preview update manifest returned status %d: %.96s", resp.StatusCode, string(body))
	}

	payload, err := io.ReadAll(io.LimitReader(resp.Body, maxManifestBytes+1))
	if err != nil {
		return false, candidate, fmt.Errorf("read MLX preview update manifest: %w", err)
	}
	if len(payload) > maxManifestBytes {
		return false, candidate, fmt.Errorf("MLX preview update manifest exceeds %d bytes", maxManifestBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var manifest updateManifest
	if err := decoder.Decode(&manifest); err != nil {
		return false, candidate, fmt.Errorf("decode MLX preview update manifest: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return false, candidate, fmt.Errorf("decode MLX preview update manifest: %w", err)
	}
	if err := validateManifest(manifest); err != nil {
		return false, candidate, err
	}

	comparison, err := compareUpdateVersions(manifest.MarketingVersion, manifest.BuildVersion, CurrentMarketingVersion, CurrentBuildVersion)
	if err != nil {
		return false, candidate, err
	}
	if comparison <= 0 {
		return false, candidate, nil
	}

	candidate = UpdateResponse{
		Source:           manifest.Source,
		Channel:          manifest.Channel,
		Repository:       manifest.Repository,
		MarketingVersion: manifest.MarketingVersion,
		BuildVersion:     manifest.BuildVersion,
		ReleaseTag:       manifest.ReleaseTag,
		ReleasePageURL:   manifest.ReleasePageURL,
		Platform:         manifest.Platform,
		Architecture:     manifest.Architecture,
		AssetName:        manifest.Asset.Name,
		ExpectedSize:     manifest.Asset.SizeBytes,
		SHA256:           manifest.Asset.SHA256,
		UpdateURL:        manifest.Asset.URL,
		UpdateVersion:    manifest.MarketingVersion,
	}
	return true, candidate, nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err == io.EOF {
		return nil
	} else if err != nil {
		return err
	}
	return errors.New("manifest contains multiple JSON values")
}

func validateManifestURL(parsed *url.URL) error {
	if parsed == nil || parsed.User != nil || parsed.Host == "" {
		return errors.New("invalid update manifest URL")
	}
	if parsed.Scheme != "https" && !(allowInsecureUpdateURLs && parsed.Scheme == "http") {
		return fmt.Errorf("update manifest URL must use HTTPS: %q", parsed.String())
	}
	if !allowInsecureUpdateURLs && parsed.Hostname() != "pd95.github.io" {
		return fmt.Errorf("unexpected update manifest host %q", parsed.Hostname())
	}
	if (!allowInsecureUpdateURLs && parsed.Port() != "") || parsed.Fragment != "" {
		return fmt.Errorf("invalid update manifest URL %q", parsed.String())
	}
	if !allowInsecureUpdateURLs && parsed.EscapedPath() != "/Ollama/updates/v1/preview/darwin-arm64.json" {
		return fmt.Errorf("unexpected update manifest path %q", parsed.EscapedPath())
	}
	return nil
}

func validateManifest(manifest updateManifest) error {
	if manifest.SchemaVersion != 1 {
		return fmt.Errorf("unsupported update manifest schema %d", manifest.SchemaVersion)
	}
	if manifest.Source != UpdateSource || manifest.Channel != UpdateChannel || manifest.Repository != UpdateRepository {
		return errors.New("update manifest source, channel, or repository does not match this build")
	}
	if manifest.Platform != runtime.GOOS || manifest.Architecture != runtime.GOARCH {
		return fmt.Errorf("update manifest targets %s/%s, expected %s/%s", manifest.Platform, manifest.Architecture, runtime.GOOS, runtime.GOARCH)
	}
	if !marketingVersionPattern.MatchString(manifest.MarketingVersion) {
		return fmt.Errorf("invalid update marketing version %q", manifest.MarketingVersion)
	}
	if _, err := parseBuildVersion(manifest.BuildVersion); err != nil {
		return fmt.Errorf("invalid update build version: %w", err)
	}
	if !releaseTagPattern.MatchString(manifest.ReleaseTag) {
		return errors.New("update manifest has an invalid release tag")
	}
	if manifest.Asset.Name != Installer {
		return fmt.Errorf("unexpected update asset %q", manifest.Asset.Name)
	}
	if manifest.Asset.SizeBytes <= 0 || manifest.Asset.SizeBytes > maxArchiveBytes {
		return fmt.Errorf("update asset size %d is outside the allowed range", manifest.Asset.SizeBytes)
	}
	if len(manifest.Asset.SHA256) != sha256.Size*2 || strings.ToLower(manifest.Asset.SHA256) != manifest.Asset.SHA256 {
		return errors.New("update manifest has an invalid SHA-256")
	}
	if _, err := hex.DecodeString(manifest.Asset.SHA256); err != nil {
		return errors.New("update manifest has an invalid SHA-256")
	}
	assetURL, err := url.Parse(manifest.Asset.URL)
	if err != nil || validateReleaseAssetURL(assetURL) != nil {
		return fmt.Errorf("update manifest has an invalid release asset URL %q", manifest.Asset.URL)
	}
	if path.Base(assetURL.Path) != manifest.Asset.Name {
		return errors.New("update asset URL does not match its declared filename")
	}
	if !allowInsecureUpdateURLs {
		expectedAssetPath := "/" + manifest.Repository + "/releases/download/" + manifest.ReleaseTag + "/" + manifest.Asset.Name
		if assetURL.Path != expectedAssetPath {
			return errors.New("update asset URL does not match its declared release tag")
		}
	}
	releaseURL, err := url.Parse(manifest.ReleasePageURL)
	if err != nil || validateReleasePageURL(releaseURL) != nil {
		return fmt.Errorf("update manifest has an invalid release page URL %q", manifest.ReleasePageURL)
	}
	expectedReleasePath := "/" + manifest.Repository + "/releases/tag/" + manifest.ReleaseTag
	if releaseURL.Path != expectedReleasePath {
		return errors.New("update release page URL does not match its declared release tag")
	}
	return nil
}

func validateReleaseAssetURL(parsed *url.URL) error {
	if parsed == nil || parsed.User != nil || parsed.Host == "" || parsed.Fragment != "" {
		return errors.New("invalid release asset URL")
	}
	if parsed.Scheme != "https" && !(allowInsecureUpdateURLs && parsed.Scheme == "http") {
		return errors.New("release asset URL must use HTTPS")
	}
	if allowInsecureUpdateURLs {
		return nil
	}
	if parsed.Hostname() != "github.com" || parsed.Port() != "" {
		return errors.New("unexpected release asset host")
	}
	prefix := "/" + UpdateRepository + "/releases/download/"
	if !strings.HasPrefix(parsed.Path, prefix) || parsed.RawQuery != "" {
		return errors.New("unexpected release asset path")
	}
	return nil
}

func validateReleasePageURL(parsed *url.URL) error {
	if parsed == nil || parsed.Scheme != "https" || parsed.Hostname() != "github.com" || parsed.Port() != "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return errors.New("invalid release page URL")
	}
	if !strings.HasPrefix(parsed.Path, "/"+UpdateRepository+"/releases/tag/") {
		return errors.New("unexpected release page path")
	}
	return nil
}

func compareUpdateVersions(marketing, build, currentMarketing, currentBuild string) (int, error) {
	if !marketingVersionPattern.MatchString(marketing) || !marketingVersionPattern.MatchString(currentMarketing) {
		return 0, errors.New("marketing versions must contain three numeric components")
	}
	if cmp := semver.Compare("v"+marketing, "v"+currentMarketing); cmp != 0 {
		return cmp, nil
	}
	nextBuild, err := parseBuildVersion(build)
	if err != nil {
		return 0, err
	}
	current, err := parseBuildVersion(currentBuild)
	if err != nil {
		return 0, err
	}
	for i := range nextBuild {
		if nextBuild[i] < current[i] {
			return -1, nil
		}
		if nextBuild[i] > current[i] {
			return 1, nil
		}
	}
	return 0, nil
}

func parseBuildVersion(value string) ([3]uint64, error) {
	var parsed [3]uint64
	if !buildVersionPattern.MatchString(value) {
		return parsed, fmt.Errorf("build version %q must contain three numeric components", value)
	}
	for i, component := range strings.Split(value, ".") {
		n, err := strconv.ParseUint(component, 10, 31)
		if err != nil {
			return parsed, fmt.Errorf("invalid build version %q", value)
		}
		parsed[i] = n
	}
	return parsed, nil
}
