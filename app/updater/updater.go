//go:build windows || darwin

package updater

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/auth"
	"golang.org/x/mod/semver"
)

var (
	UpdateCheckURLBase      = "https://ollama.com/api/update"
	UpdateDownloaded        = false
	UpdateCheckInterval     = 60 * 60 * time.Second
	UpdateCheckInitialDelay = 3 * time.Second // 30 * time.Second
	UpdateCheckTimeout      = 30 * time.Second
	UpdateDownloadTimeout   = 30 * time.Minute
	UpdateDownloadAttempts  = 3
	UpdateDownloadRetryWait = 500 * time.Millisecond

	UpdateStageDir    string
	UpgradeLogFile    string
	UpgradeMarkerFile string
	Installer         string
	UserAgentOS       string
	DisableUpdates    string

	VerifyDownload  func(string) error
	VerifyCandidate func(string, UpdateResponse) error

	readyState struct {
		sync.Mutex
		path    string
		size    int64
		modTime time.Time
	}
)

// TODO - maybe move up to the API package?
type UpdateResponse struct {
	Source           string `json:"source,omitempty"`
	Channel          string `json:"channel,omitempty"`
	Repository       string `json:"repository,omitempty"`
	MarketingVersion string `json:"marketing_version,omitempty"`
	BuildVersion     string `json:"build_version,omitempty"`
	ReleaseTag       string `json:"release_tag,omitempty"`
	ReleasePageURL   string `json:"release_page_url,omitempty"`
	Platform         string `json:"platform,omitempty"`
	Architecture     string `json:"architecture,omitempty"`
	AssetName        string `json:"asset_name,omitempty"`
	ExpectedSize     int64  `json:"size_bytes,omitempty"`
	SHA256           string `json:"sha256,omitempty"`
	UpdateURL        string `json:"url"`
	UpdateVersion    string `json:"version"`
}

const updateMetadataFilename = "update.json"

var ErrInstallCancelled = errors.New("update installation cancelled")

func AutomaticUpdatesDisabled() bool {
	switch strings.ToLower(strings.TrimSpace(DisableUpdates)) {
	case "1", "t", "true", "y", "yes", "on":
		return true
	default:
		return false
	}
}

func (u *Updater) checkForUpdate(ctx context.Context) (bool, UpdateResponse, error) {
	return u.checkForUpdateSource(ctx, PrimaryUpdateSource())
}

func (u *Updater) checkForUpdateSource(ctx context.Context, source string) (bool, UpdateResponse, error) {
	if !ValidUpdateSource(source) {
		return false, UpdateResponse{}, fmt.Errorf("unsupported update source %q", source)
	}
	if source == MLXPreviewSource {
		return u.checkCustomUpdate(ctx)
	}
	if CustomUpdateSourceEnabled() && !OfficialReplacementAvailable() {
		return false, UpdateResponse{}, errors.New("official Ollama installation is unavailable because its signing identity is not pinned")
	}
	var updateResp UpdateResponse

	requestURL, err := url.Parse(UpdateCheckURLBase)
	if err != nil {
		return false, updateResp, fmt.Errorf("parse update check URL: %w", err)
	}

	query := requestURL.Query()
	query.Add("os", runtime.GOOS)
	query.Add("arch", runtime.GOARCH)
	currentVersion := version.Version
	if CustomUpdateSourceEnabled() && CurrentMarketingVersion != "" {
		currentVersion = CurrentMarketingVersion
	}
	query.Add("version", currentVersion)
	query.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))

	// The original macOS app used to use the device ID
	// to check for updates so include it if present
	if runtime.GOOS == "darwin" && u.Store != nil {
		if id, err := u.Store.ID(); err == nil && id != "" {
			query.Add("id", id)
		}
	}

	var signature string

	nonce, err := auth.NewNonce(rand.Reader, 16)
	if err != nil {
		// Don't sign if we haven't yet generated a key pair for the server
		slog.Debug("unable to generate nonce for update check request", "error", err)
	} else {
		query.Add("nonce", nonce)
		requestURL.RawQuery = query.Encode()

		data := []byte(fmt.Sprintf("%s,%s", http.MethodGet, requestURL.RequestURI()))
		signature, err = auth.Sign(ctx, data)
		if err != nil {
			slog.Debug("unable to generate signature for update check request", "error", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		return false, updateResp, fmt.Errorf("create update check request: %w", err)
	}
	if signature != "" {
		req.Header.Set("Authorization", signature)
	}
	ua := fmt.Sprintf("ollama/%s %s Go/%s %s", version.Version, runtime.GOARCH, runtime.Version(), UserAgentOS)
	req.Header.Set("User-Agent", ua)

	slog.Debug("checking for available update", "requestURL", requestURL, "User-Agent", ua)
	client := &http.Client{Timeout: UpdateCheckTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return false, updateResp, fmt.Errorf("check for update: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		slog.Debug("check update response 204 (current version is up to date)")
		return false, updateResp, nil
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, updateResp, fmt.Errorf("read update response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return false, updateResp, fmt.Errorf("update service returned status %d: %.96s", resp.StatusCode, string(body))
	}
	err = json.Unmarshal(body, &updateResp)
	if err != nil {
		return false, updateResp, fmt.Errorf("decode update response: %w", err)
	}
	updateURL, err := url.Parse(updateResp.UpdateURL)
	if err != nil || updateURL.Scheme == "" || updateURL.Host == "" {
		return false, UpdateResponse{}, fmt.Errorf("update response contains invalid download URL %q", updateResp.UpdateURL)
	}
	if CustomUpdateSourceEnabled() {
		if err := validateOfficialAssetURL(updateURL); err != nil {
			return false, UpdateResponse{}, fmt.Errorf("update response contains invalid official download URL: %w", err)
		}
	} else if updateURL.Scheme != "http" && updateURL.Scheme != "https" {
		return false, UpdateResponse{}, fmt.Errorf("update response contains unsupported download URL %q", updateResp.UpdateURL)
	}
	// Extract the version string from the URL in the github release artifact path
	updateResp.UpdateVersion = path.Base(path.Dir(updateResp.UpdateURL))
	if updateResp.UpdateVersion == "." || updateResp.UpdateVersion == "/" || updateResp.UpdateVersion == "" {
		return false, UpdateResponse{}, errors.New("update response is missing a version")
	}

	updateResp.Source = OfficialUpdateSource
	updateResp.Channel = "stable"
	updateResp.Repository = "ollama/ollama"
	updateResp.MarketingVersion = strings.TrimPrefix(updateResp.UpdateVersion, "v")
	updateResp.BuildVersion = updateResp.MarketingVersion
	updateResp.Platform = runtime.GOOS
	updateResp.Architecture = runtime.GOARCH
	updateResp.AssetName = path.Base(updateURL.Path)
	updateResp.ReleasePageURL = "https://github.com/ollama/ollama/releases/tag/" + url.PathEscape(updateResp.UpdateVersion)
	if !marketingVersionPattern.MatchString(updateResp.MarketingVersion) {
		return false, UpdateResponse{}, fmt.Errorf("official update has invalid marketing version %q", updateResp.UpdateVersion)
	}
	baseline := strings.TrimPrefix(currentVersion, "v")
	if !marketingVersionPattern.MatchString(baseline) || semver.Compare("v"+updateResp.MarketingVersion, "v"+baseline) <= 0 {
		return false, UpdateResponse{}, nil
	}
	slog.Info("New update available at " + updateResp.UpdateURL)
	return true, updateResp, nil
}

func (u *Updater) DownloadNewRelease(ctx context.Context, updateResp UpdateResponse) error {
	// Create a cancellable context for this download
	downloadCtx, cancel := context.WithTimeout(ctx, UpdateDownloadTimeout)
	u.cancelDownloadLock.Lock()
	u.cancelDownload = cancel
	u.cancelDownloadLock.Unlock()
	defer func() {
		u.cancelDownloadLock.Lock()
		u.cancelDownload = nil
		u.cancelDownloadLock.Unlock()
		cancel()
	}()

	if staged, ok := StagedUpdateForSource(candidateSource(updateResp)); ok && staged.Version != "" && staged.Version == updateResp.UpdateVersion {
		slog.Info("update already downloaded", "version", staged.Version)
		return nil
	}

	var resp *http.Response
	var err error
	for attempt := 1; attempt <= UpdateDownloadAttempts; attempt++ {
		if attempt > 1 {
			if err := waitForUpdateRetry(downloadCtx, attempt-1); err != nil {
				return err
			}
		}
		resp, err = requestUpdateDownload(downloadCtx, updateResp, 0)
		if err == nil {
			break
		}
		slog.Warn("update download connection failed; retrying", "attempt", attempt, "error", err)
	}
	if err != nil {
		return fmt.Errorf("connect to %s update after %d attempts: %w", candidateSource(updateResp), UpdateDownloadAttempts, err)
	}
	filename := Installer
	if updateResp.AssetName != "" {
		filename = updateResp.AssetName
	}
	validator := resp.Header.Get("etag")
	_, params, mediaErr := mime.ParseMediaType(resp.Header.Get("content-disposition"))
	if mediaErr == nil && params["filename"] != "" {
		if updateResp.Source == MLXPreviewSource && params["filename"] != filename {
			resp.Body.Close()
			return fmt.Errorf("update response filename %q does not match manifest asset %q", params["filename"], filename)
		}
		filename = params["filename"]
	}
	stageValidator := validator
	if updateResp.Source == MLXPreviewSource {
		stageValidator = strings.Join([]string{updateResp.Source, updateResp.MarketingVersion, updateResp.BuildVersion, updateResp.SHA256, validator}, "\x00")
	}
	stageDir := updateSourceStageDir(candidateSource(updateResp))
	stageFilename, err := updateStagePath(stageDir, stageValidator, filename)
	if err != nil {
		resp.Body.Close()
		return err
	}
	cleanupOldDownloads(stageDir)
	if err := os.MkdirAll(filepath.Dir(stageFilename), 0o755); err != nil {
		resp.Body.Close()
		return fmt.Errorf("create update stage directory: %w", err)
	}
	partialFilename := stageFilename + ".partial"
	_ = os.Remove(partialFilename)
	defer os.Remove(partialFilename) //nolint:errcheck

	var downloaded int64
	var downloadErr error
	for attempt := 1; attempt <= UpdateDownloadAttempts; attempt++ {
		if attempt > 1 {
			if err := waitForUpdateRetry(downloadCtx, attempt-1); err != nil {
				return err
			}
			resp, downloadErr = requestUpdateDownload(downloadCtx, updateResp, downloaded, validator)
			if downloadErr != nil {
				slog.Warn("update download retry failed", "attempt", attempt, "error", downloadErr)
				continue
			}
		}

		if downloaded > 0 && resp.StatusCode == http.StatusPartialContent {
			contentRange := resp.Header.Get("Content-Range")
			validatorChanged := validator != "" && resp.Header.Get("ETag") != validator
			if validator == "" || validatorChanged || !strings.HasPrefix(contentRange, fmt.Sprintf("bytes %d-", downloaded)) {
				resp.Body.Close()
				downloadErr = fmt.Errorf("update server returned an invalid continuation response")
				downloaded = 0
				continue
			}
			if updateResp.ExpectedSize > 0 && !strings.HasSuffix(contentRange, fmt.Sprintf("/%d", updateResp.ExpectedSize)) {
				resp.Body.Close()
				downloadErr = fmt.Errorf("update server returned an unexpected continuation size")
				downloaded = 0
				continue
			}
		}

		flags := os.O_WRONLY | os.O_CREATE
		if downloaded > 0 && resp.StatusCode == http.StatusPartialContent {
			flags |= os.O_APPEND
		} else {
			flags |= os.O_TRUNC
			downloaded = 0
		}
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
			resp.Body.Close()
			downloadErr = fmt.Errorf("update download returned status %d", resp.StatusCode)
			continue
		}
		fp, openErr := os.OpenFile(partialFilename, flags, 0o600)
		if openErr != nil {
			resp.Body.Close()
			return fmt.Errorf("open partial update: %w", openErr)
		}
		var reader io.Reader = resp.Body
		if updateResp.ExpectedSize > 0 {
			remaining := updateResp.ExpectedSize - downloaded
			if remaining < 0 {
				remaining = 0
			}
			reader = io.LimitReader(resp.Body, remaining+1)
		}
		n, copyErr := io.Copy(fp, reader)
		downloaded += n
		closeErr := fp.Close()
		resp.Body.Close()
		if copyErr == nil && closeErr == nil {
			if updateResp.ExpectedSize > 0 && downloaded != updateResp.ExpectedSize {
				downloadErr = fmt.Errorf("downloaded update size %d does not match manifest size %d", downloaded, updateResp.ExpectedSize)
				if downloaded > updateResp.ExpectedSize {
					downloaded = 0
				}
				slog.Warn("update download size mismatch; retrying", "attempt", attempt, "error", downloadErr)
				continue
			}
			downloadErr = nil
			break
		}
		if copyErr != nil {
			downloadErr = fmt.Errorf("download interrupted after %d bytes: %w", downloaded, copyErr)
			if validator == "" {
				downloaded = 0
			}
		} else {
			downloadErr = fmt.Errorf("close partial update: %w", closeErr)
		}
		slog.Warn("update download interrupted; retrying", "attempt", attempt, "bytes", downloaded, "error", downloadErr)
	}
	if downloadErr != nil {
		return fmt.Errorf("download %s update after %d attempts: %w", candidateSource(updateResp), UpdateDownloadAttempts, downloadErr)
	}
	slog.Info("new update downloaded", "partial", partialFilename, "bytes", downloaded)

	if err := downloadCtx.Err(); err != nil {
		return err
	}
	if err := verifyCandidateChecksum(partialFilename, updateResp); err != nil {
		return fmt.Errorf("verify downloaded update checksum: %w", err)
	}
	if err := verifyDownloadedCandidate(partialFilename, updateResp); err != nil {
		return fmt.Errorf("verify downloaded update: %w", err)
	}
	if err := downloadCtx.Err(); err != nil {
		return err
	}
	if err := writeUpdateCandidateMetadata(partialFilename, updateResp); err != nil {
		return err
	}
	if err := downloadCtx.Err(); err != nil {
		_ = os.Remove(filepath.Join(filepath.Dir(stageFilename), updateMetadataFilename))
		return err
	}
	if err := os.Rename(partialFilename, stageFilename); err != nil {
		_ = os.Remove(filepath.Join(filepath.Dir(stageFilename), updateMetadataFilename))
		return fmt.Errorf("publish verified update: %w", err)
	}
	markReadyUpdate(stageFilename)
	slog.Info("verified update staged", "bundle", stageFilename)
	UpdateDownloaded = true
	return nil
}

func requestUpdateDownload(ctx context.Context, candidate UpdateResponse, offset int64, validator ...string) (*http.Response, error) {
	initialURL, parseErr := url.Parse(candidate.UpdateURL)
	if parseErr != nil {
		return nil, fmt.Errorf("parse update download URL: %w", parseErr)
	}
	if candidate.Source == MLXPreviewSource {
		if err := validateReleaseAssetURL(initialURL); err != nil {
			return nil, fmt.Errorf("invalid update download URL: %w", err)
		}
	} else if CustomUpdateSourceEnabled() {
		if err := validateOfficialAssetURL(initialURL); err != nil {
			return nil, fmt.Errorf("invalid official update download URL: %w", err)
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, candidate.UpdateURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create update download request: %w", err)
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		if len(validator) > 0 && validator[0] != "" {
			req.Header.Set("If-Range", validator[0])
		}
	}
	req.Header.Set("Accept-Encoding", "identity")
	client := http.DefaultClient
	if candidate.Source == MLXPreviewSource || (candidate.Source == OfficialUpdateSource && CustomUpdateSourceEnabled()) {
		client = &http.Client{
			Timeout: UpdateDownloadTimeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 5 {
					return errors.New("too many update download redirects")
				}
				if req.URL.Scheme != "https" && !(allowInsecureUpdateURLs && req.URL.Scheme == "http") {
					return errors.New("update download redirect must use HTTPS")
				}
				if allowInsecureUpdateURLs {
					if req.URL.Host != initialURL.Host {
						return fmt.Errorf("unexpected update download redirect host %q", req.URL.Host)
					}
					return nil
				}
				if req.URL.User != nil || req.URL.Port() != "" || req.URL.Hostname() != "release-assets.githubusercontent.com" {
					return fmt.Errorf("unexpected update download redirect host %q", req.URL.Hostname())
				}
				return nil
			},
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		var urlErr *url.Error
		if errors.As(err, &urlErr) {
			return nil, fmt.Errorf("update download connection failed: %w", urlErr.Err)
		}
		return nil, fmt.Errorf("update download connection failed: %w", err)
	}
	return resp, nil
}

func waitForUpdateRetry(ctx context.Context, retry int) error {
	timer := time.NewTimer(time.Duration(retry) * UpdateDownloadRetryWait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func writeUpdateMetadata(bundle, version string) error {
	return writeUpdateCandidateMetadata(bundle, UpdateResponse{Source: OfficialUpdateSource, UpdateVersion: version})
}

func writeUpdateCandidateMetadata(bundle string, candidate UpdateResponse) error {
	if candidate.Source == "" {
		candidate.Source = OfficialUpdateSource
	}
	payload, err := json.Marshal(candidate)
	if err != nil {
		return fmt.Errorf("encode update metadata: %w", err)
	}
	metadataPath := filepath.Join(filepath.Dir(bundle), updateMetadataFilename)
	tmpPath := metadataPath + ".tmp"
	if err := os.WriteFile(tmpPath, payload, 0o600); err != nil {
		return fmt.Errorf("write update metadata: %w", err)
	}
	if err := os.Rename(tmpPath, metadataPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("commit update metadata: %w", err)
	}
	return nil
}

func candidateSource(candidate UpdateResponse) string {
	if candidate.Source == "" {
		return OfficialUpdateSource
	}
	return candidate.Source
}

func verifyDownloadedCandidate(bundle string, candidate UpdateResponse) error {
	if (candidate.Source == MLXPreviewSource || (candidate.Source == OfficialUpdateSource && CustomUpdateSourceEnabled())) && VerifyCandidate != nil {
		return VerifyCandidate(bundle, candidate)
	}
	if VerifyDownload == nil {
		return errors.New("platform update verifier is unavailable")
	}
	return VerifyDownload(bundle)
}

func validateOfficialAssetURL(parsed *url.URL) error {
	if parsed == nil || parsed.User != nil || parsed.Scheme == "" || parsed.Host == "" {
		return errors.New("invalid official asset URL")
	}
	if parsed.Scheme != "https" && !(allowInsecureUpdateURLs && parsed.Scheme == "http") {
		return errors.New("official asset URL must use HTTPS")
	}
	if allowInsecureUpdateURLs {
		return nil
	}
	if parsed.Hostname() != "github.com" || parsed.Port() != "" || parsed.Fragment != "" {
		return fmt.Errorf("unexpected official asset host %q", parsed.Hostname())
	}
	parts := strings.Split(strings.Trim(parsed.EscapedPath(), "/"), "/")
	if len(parts) != 6 || parts[0] != "ollama" || parts[1] != "ollama" || parts[2] != "releases" || parts[3] != "download" ||
		!strings.HasPrefix(parts[4], "v") || parts[5] != "Ollama-darwin.zip" {
		return fmt.Errorf("unexpected official asset path %q", parsed.EscapedPath())
	}
	return nil
}

func verifyCandidateChecksum(bundle string, candidate UpdateResponse) error {
	if candidate.SHA256 == "" {
		if candidate.Source == MLXPreviewSource {
			return errors.New("MLX preview update is missing a SHA-256")
		}
		return nil
	}
	fp, err := os.Open(bundle)
	if err != nil {
		return err
	}
	defer fp.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, fp); err != nil {
		return err
	}
	if actual := hex.EncodeToString(hash.Sum(nil)); actual != candidate.SHA256 {
		return fmt.Errorf("SHA-256 mismatch: got %s", actual)
	}
	return nil
}

func updateStagePath(stageDir, etag, filename string) (string, error) {
	filename, err := safeUpdateFilename(filename)
	if err != nil {
		return "", err
	}

	stageDir, err = filepath.Abs(stageDir)
	if err != nil {
		return "", fmt.Errorf("resolve update stage dir: %w", err)
	}

	stageFilename := filepath.Join(stageDir, updateStageETagDir(etag), filename)
	if err := ensurePathInDir(stageDir, stageFilename); err != nil {
		return "", err
	}

	return stageFilename, nil
}

func safeUpdateFilename(filename string) (string, error) {
	filename = strings.TrimSpace(filename)
	if filename == "" {
		return "", errors.New("missing update filename")
	}
	if filename == "." || filename == ".." ||
		filepath.IsAbs(filename) || path.IsAbs(filename) ||
		strings.ContainsAny(filename, `/\:`) ||
		filepath.Base(filename) != filename || path.Base(filename) != filename {
		return "", fmt.Errorf("unsafe update filename %q", filename)
	}
	return filename, nil
}

func updateStageETagDir(etag string) string {
	etag = strings.Trim(strings.TrimSpace(etag), "\"")
	if etag == "" {
		slog.Debug("no etag detected, falling back to filename based dedup")
		return "_"
	}

	sum := sha256.Sum256([]byte(etag))
	return hex.EncodeToString(sum[:])
}

func ensurePathInDir(dir, name string) error {
	rel, err := filepath.Rel(dir, name)
	if err != nil {
		return fmt.Errorf("resolve update staging path: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return fmt.Errorf("update staging path escapes stage dir: %s", name)
	}
	return nil
}

func cleanupOldDownloads(stageDir string) {
	files, err := os.ReadDir(stageDir)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		// Expected behavior on first run
		return
	} else if err != nil {
		slog.Warn(fmt.Sprintf("failed to list stage dir: %s", err))
		return
	}
	for _, file := range files {
		fullname := filepath.Join(stageDir, file.Name())
		slog.Debug("cleaning up old download: " + fullname)
		err = os.RemoveAll(fullname)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to cleanup stale update download %s", err))
		}
	}
}

func updateSourceStageDir(source string) string {
	if !CustomUpdateSourceEnabled() {
		return UpdateStageDir
	}
	return filepath.Join(UpdateStageDir, source)
}

func stagedUpdatePath(source, extension string) string {
	if !ValidUpdateSource(source) {
		return ""
	}
	patterns := []string{filepath.Join(updateSourceStageDir(source), "*", "*."+extension)}
	// Admit official archives staged by versions predating source-qualified
	// directories so an upgrade does not strand a previously verified update.
	if source == OfficialUpdateSource {
		patterns = append(patterns, filepath.Join(UpdateStageDir, "*", "*."+extension))
	}
	for _, pattern := range patterns {
		files, err := filepath.Glob(pattern)
		if err != nil {
			slog.Debug("failed to lookup downloads", "pattern", pattern, "error", err)
			continue
		}
		for _, file := range files {
			candidate, err := readUpdateCandidateMetadata(file)
			if err == nil && candidateSource(candidate) == source {
				return file
			}
			// A malformed candidate in a source-owned directory must still be
			// returned so ReadyUpdatePendingForSource can delete it. Never infer
			// ownership for a legacy shared directory in a custom build.
			if err != nil && (!CustomUpdateSourceEnabled() || strings.HasPrefix(file, updateSourceStageDir(source)+string(filepath.Separator))) {
				return file
			}
		}
	}
	return ""
}

// ReadyUpdatePending reports whether a staged update exists and still passes
// platform verification. Invalid staged files are removed so they can never
// activate a restart action on a later launch.
func ReadyUpdatePending() bool {
	return ReadyUpdatePendingForSource(PrimaryUpdateSource())
}

func ReadyUpdatePendingForSource(source string) bool {
	bundle := getStagedUpdateForSource(source)
	if bundle == "" {
		return false
	}
	if readyUpdateUnchanged(bundle) {
		return true
	}
	candidate, err := readUpdateCandidateMetadata(bundle)
	if err != nil {
		slog.Warn("removing update with invalid staged metadata", "bundle", bundle, "error", err)
		removeStagedUpdate(bundle)
		return false
	}
	if err := verifyCandidateChecksum(bundle, candidate); err != nil {
		slog.Warn("removing update with invalid staged checksum", "bundle", bundle, "error", err)
		removeStagedUpdate(bundle)
		return false
	}
	if err := verifyDownloadedCandidate(bundle, candidate); err != nil {
		slog.Warn("removing invalid staged update", "bundle", bundle, "error", err)
		removeStagedUpdate(bundle)
		return false
	}
	markReadyUpdate(bundle)
	UpdateDownloaded = true
	return true
}

func removeStagedUpdate(bundle string) {
	_ = os.Remove(bundle)
	_ = os.Remove(filepath.Join(filepath.Dir(bundle), updateMetadataFilename))
	forgetReadyUpdate(bundle)
	UpdateDownloaded = false
}

func markReadyUpdate(bundle string) {
	info, err := os.Stat(bundle)
	if err != nil {
		return
	}
	readyState.Lock()
	defer readyState.Unlock()
	readyState.path = bundle
	readyState.size = info.Size()
	readyState.modTime = info.ModTime()
}

func readyUpdateUnchanged(bundle string) bool {
	info, err := os.Stat(bundle)
	if err != nil {
		return false
	}
	readyState.Lock()
	defer readyState.Unlock()
	return readyState.path == bundle && readyState.size == info.Size() && readyState.modTime.Equal(info.ModTime())
}

func forgetReadyUpdate(bundle string) {
	readyState.Lock()
	defer readyState.Unlock()
	if readyState.path == bundle {
		readyState.path = ""
		readyState.size = 0
		readyState.modTime = time.Time{}
	}
}

// StagedUpdate returns the durable ready state used to restore Settings after
// it is reopened. Archives staged by older builds may not have version metadata.
func StagedUpdate() (ManualUpdateResult, bool) {
	return StagedUpdateForSource(PrimaryUpdateSource())
}

func StagedUpdateForSource(source string) (ManualUpdateResult, bool) {
	if !ReadyUpdatePendingForSource(source) {
		return ManualUpdateResult{}, false
	}
	bundle := getStagedUpdateForSource(source)
	candidate, err := readUpdateCandidateMetadata(bundle)
	if err != nil {
		return ManualUpdateResult{}, false
	}
	return ManualUpdateResult{
		Status:         "ready",
		Version:        candidate.UpdateVersion,
		BuildVersion:   candidate.BuildVersion,
		Source:         candidateSource(candidate),
		Channel:        candidate.Channel,
		ReleasePageURL: candidate.ReleasePageURL,
	}, true
}

func readUpdateCandidateMetadata(bundle string) (UpdateResponse, error) {
	metadataPath := filepath.Join(filepath.Dir(bundle), updateMetadataFilename)
	fp, err := os.Open(metadataPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// Official updater archives from older builds did not carry metadata.
			return UpdateResponse{Source: OfficialUpdateSource}, nil
		}
		return UpdateResponse{}, fmt.Errorf("open staged update metadata: %w", err)
	}
	defer fp.Close()
	payload, err := io.ReadAll(io.LimitReader(fp, maxManifestBytes+1))
	if err != nil {
		return UpdateResponse{}, fmt.Errorf("read staged update metadata: %w", err)
	}
	if len(payload) > maxManifestBytes {
		return UpdateResponse{}, errors.New("staged update metadata is too large")
	}
	var candidate UpdateResponse
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&candidate); err != nil {
		return UpdateResponse{}, fmt.Errorf("decode staged update metadata: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return UpdateResponse{}, fmt.Errorf("decode staged update metadata: %w", err)
	}
	if candidate.Source == "" {
		candidate.Source = OfficialUpdateSource
	}
	if candidate.Source == MLXPreviewSource {
		if candidate.UpdateVersion == "" || candidate.BuildVersion == "" || candidate.SHA256 == "" || candidate.AssetName != Installer {
			return UpdateResponse{}, errors.New("staged MLX preview metadata is incomplete")
		}
	}
	return candidate, nil
}

type Updater struct {
	Store              *store.Store
	cancelDownload     context.CancelFunc
	cancelDownloadLock sync.Mutex
	checkNow           chan bool
	operationLock      sync.Mutex
}

type ManualUpdateResult struct {
	Status         string `json:"status"`
	Version        string `json:"version,omitempty"`
	BuildVersion   string `json:"buildVersion,omitempty"`
	Source         string `json:"source,omitempty"`
	Channel        string `json:"channel,omitempty"`
	ReleasePageURL string `json:"releasePageUrl,omitempty"`
	NewlyStaged    bool   `json:"-"`
}

// CheckForUpdates checks the official update service and stages any available
// release before reporting it as ready. Calls are serialized with background
// update work so only one check or download can run at a time.
func (u *Updater) CheckForUpdates(ctx context.Context) (ManualUpdateResult, error) {
	return u.CheckForUpdatesForSource(ctx, PrimaryUpdateSource())
}

func (u *Updater) CheckForUpdatesForSource(ctx context.Context, source string) (ManualUpdateResult, error) {
	u.operationLock.Lock()
	defer u.operationLock.Unlock()
	if !ValidUpdateSource(source) {
		return ManualUpdateResult{}, fmt.Errorf("unsupported update source %q", source)
	}
	if staged, ok := StagedUpdateForSource(source); ok {
		return staged, nil
	}

	available, resp, err := u.checkForUpdateSource(ctx, source)
	if err != nil {
		return ManualUpdateResult{}, err
	}
	if !available {
		channel := "stable"
		if source == MLXPreviewSource {
			channel = UpdateChannel
		}
		return ManualUpdateResult{Status: "up_to_date", Source: source, Channel: channel}, nil
	}
	if err := u.DownloadNewRelease(ctx, resp); err != nil {
		return ManualUpdateResult{}, fmt.Errorf("stage %s update %s: %w", candidateSource(resp), resp.UpdateVersion, err)
	}
	return ManualUpdateResult{
		Status:         "ready",
		Version:        resp.UpdateVersion,
		BuildVersion:   resp.BuildVersion,
		Source:         candidateSource(resp),
		Channel:        resp.Channel,
		ReleasePageURL: resp.ReleasePageURL,
		NewlyStaged:    true,
	}, nil
}

// CancelOngoingDownload cancels any currently running download
func (u *Updater) CancelOngoingDownload() {
	u.cancelDownloadLock.Lock()
	defer u.cancelDownloadLock.Unlock()
	if u.cancelDownload != nil {
		slog.Info("cancelling ongoing update download")
		u.cancelDownload()
		u.cancelDownload = nil
	}
}

// TriggerImmediateCheck signals the background checker to check for updates immediately
func (u *Updater) TriggerImmediateCheck() {
	if u.checkNow != nil {
		select {
		case u.checkNow <- true:
		default:
			// Check already pending, no need to queue another
		}
	}
}

func (u *Updater) StartBackgroundUpdaterChecker(ctx context.Context, cb func(string) error) {
	u.startBackgroundUpdaterChecker(ctx, cb)
}

func (u *Updater) startBackgroundUpdaterChecker(ctx context.Context, cb func(string) error) <-chan struct{} {
	u.checkNow = make(chan bool, 1)
	if !AutomaticUpdatesDisabled() {
		u.checkNow <- false // Trigger first check after initial delay
	} else {
		slog.Info("automatic update startup check disabled by build")
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		// Don't blast an update message immediately after startup
		initialDelay := time.NewTimer(UpdateCheckInitialDelay)
		defer initialDelay.Stop()
		select {
		case <-ctx.Done():
			return
		case <-initialDelay.C:
		}
		slog.Info("beginning update checker", "interval", UpdateCheckInterval)
		ticker := time.NewTicker(UpdateCheckInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				slog.Debug("stopping background update checker")
				return
			case <-u.checkNow:
				// Immediate check triggered
			case <-ticker.C:
				// Regular interval check
			}

			// Preview builds only check when explicitly requested from Settings.
			if AutomaticUpdatesDisabled() {
				slog.Debug("automatic update check disabled by build")
				continue
			}

			u.operationLock.Lock()
			// Always check for updates
			available, resp, err := u.checkForUpdate(ctx)
			if err != nil {
				slog.Warn("failed to check for update", "error", err)
				u.operationLock.Unlock()
				continue
			}
			if !available {
				u.operationLock.Unlock()
				continue
			}

			if AutomaticUpdatesDisabled() {
				// Manual preview checks use CheckForUpdates directly and never enter
				// the background queue.
				u.operationLock.Unlock()
				continue
			}

			// Update is available - check if auto-update is enabled for downloading
			settings, err := u.Store.Settings()
			if err != nil {
				slog.Error("failed to load settings", "error", err)
				u.operationLock.Unlock()
				continue
			}

			if !settings.AutoUpdateEnabled {
				// Discovery alone must not expose a restart action.
				slog.Debug("update available but auto-update disabled", "version", resp.UpdateVersion)
				u.operationLock.Unlock()
				continue
			}

			// Auto-update is enabled - download
			err = u.DownloadNewRelease(ctx, resp)
			if err != nil {
				slog.Error("failed to download new release", "error", err)
				u.operationLock.Unlock()
				continue
			}
			u.operationLock.Unlock()

			// Download successful - show tray notification
			err = cb(resp.UpdateVersion)
			if err != nil {
				slog.Warn("failed to register update available with tray", "error", err)
			}
		}
	}()
	return done
}
