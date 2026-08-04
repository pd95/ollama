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

	VerifyDownload func(string) error

	readyState struct {
		sync.Mutex
		path    string
		size    int64
		modTime time.Time
	}
)

// TODO - maybe move up to the API package?
type UpdateResponse struct {
	UpdateURL     string `json:"url"`
	UpdateVersion string `json:"version"`
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
	var updateResp UpdateResponse

	requestURL, err := url.Parse(UpdateCheckURLBase)
	if err != nil {
		return false, updateResp, fmt.Errorf("parse update check URL: %w", err)
	}

	query := requestURL.Query()
	query.Add("os", runtime.GOOS)
	query.Add("arch", runtime.GOARCH)
	currentVersion := version.Version
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
	if updateURL.Scheme != "http" && updateURL.Scheme != "https" {
		return false, UpdateResponse{}, fmt.Errorf("update response contains unsupported download URL %q", updateResp.UpdateURL)
	}
	// Extract the version string from the URL in the github release artifact path
	updateResp.UpdateVersion = path.Base(path.Dir(updateResp.UpdateURL))
	if updateResp.UpdateVersion == "." || updateResp.UpdateVersion == "/" || updateResp.UpdateVersion == "" {
		return false, UpdateResponse{}, errors.New("update response is missing a version")
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

	if staged, ok := StagedUpdate(); ok && staged.Version != "" && staged.Version == updateResp.UpdateVersion {
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
		resp, err = requestUpdateDownload(downloadCtx, updateResp.UpdateURL, 0)
		if err == nil {
			break
		}
		slog.Warn("update download connection failed; retrying", "attempt", attempt, "error", err)
	}
	if err != nil {
		return fmt.Errorf("connect to official update after %d attempts: %w", UpdateDownloadAttempts, err)
	}
	filename := Installer
	validator := resp.Header.Get("etag")
	_, params, mediaErr := mime.ParseMediaType(resp.Header.Get("content-disposition"))
	if mediaErr == nil && params["filename"] != "" {
		filename = params["filename"]
	}
	stageFilename, err := updateStagePath(UpdateStageDir, validator, filename)
	if err != nil {
		resp.Body.Close()
		return err
	}
	cleanupOldDownloads(UpdateStageDir)
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
			resp, downloadErr = requestUpdateDownload(downloadCtx, updateResp.UpdateURL, downloaded, validator)
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
		fp, openErr := os.OpenFile(partialFilename, flags, 0o755)
		if openErr != nil {
			resp.Body.Close()
			return fmt.Errorf("open partial update: %w", openErr)
		}
		n, copyErr := io.Copy(fp, resp.Body)
		downloaded += n
		closeErr := fp.Close()
		resp.Body.Close()
		if copyErr == nil && closeErr == nil {
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
		return fmt.Errorf("download official update after %d attempts: %w", UpdateDownloadAttempts, downloadErr)
	}
	slog.Info("new update downloaded", "partial", partialFilename, "bytes", downloaded)

	if err := downloadCtx.Err(); err != nil {
		return err
	}
	if err := VerifyDownload(partialFilename); err != nil {
		return fmt.Errorf("verify downloaded update: %w", err)
	}
	if err := downloadCtx.Err(); err != nil {
		return err
	}
	if err := writeUpdateMetadata(partialFilename, updateResp.UpdateVersion); err != nil {
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

func requestUpdateDownload(ctx context.Context, downloadURL string, offset int64, validator ...string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, downloadURL, nil)
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
	resp, err := http.DefaultClient.Do(req)
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
	metadata := struct {
		Version string `json:"version"`
	}{Version: version}
	payload, err := json.Marshal(metadata)
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

// ReadyUpdatePending reports whether a staged update exists and still passes
// platform verification. Invalid staged files are removed so they can never
// activate a restart action on a later launch.
func ReadyUpdatePending() bool {
	bundle := getStagedUpdate()
	if bundle == "" {
		return false
	}
	if readyUpdateUnchanged(bundle) {
		return true
	}
	if err := VerifyDownload(bundle); err != nil {
		slog.Warn("removing invalid staged update", "bundle", bundle, "error", err)
		_ = os.Remove(bundle)
		forgetReadyUpdate(bundle)
		UpdateDownloaded = false
		return false
	}
	markReadyUpdate(bundle)
	UpdateDownloaded = true
	return true
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
	if !ReadyUpdatePending() {
		return ManualUpdateResult{}, false
	}
	bundle := getStagedUpdate()
	metadataPath := filepath.Join(filepath.Dir(bundle), updateMetadataFilename)
	var metadata struct {
		Version string `json:"version"`
	}
	if fp, err := os.Open(metadataPath); err == nil {
		payload, readErr := io.ReadAll(io.LimitReader(fp, 4097))
		_ = fp.Close()
		if readErr != nil {
			slog.Warn("failed to read staged update metadata", "error", readErr)
		} else if len(payload) > 4096 {
			slog.Warn("staged update metadata is too large")
		} else if err := json.Unmarshal(payload, &metadata); err != nil {
			slog.Warn("failed to decode staged update metadata", "error", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		slog.Warn("failed to read staged update metadata", "error", err)
	}
	return ManualUpdateResult{Status: "ready", Version: metadata.Version}, true
}

type Updater struct {
	Store              *store.Store
	cancelDownload     context.CancelFunc
	cancelDownloadLock sync.Mutex
	checkNow           chan bool
	operationLock      sync.Mutex
}

type ManualUpdateResult struct {
	Status      string `json:"status"`
	Version     string `json:"version,omitempty"`
	NewlyStaged bool   `json:"-"`
}

// CheckForUpdates checks the official update service and stages any available
// release before reporting it as ready. Calls are serialized with background
// update work so only one check or download can run at a time.
func (u *Updater) CheckForUpdates(ctx context.Context) (ManualUpdateResult, error) {
	u.operationLock.Lock()
	defer u.operationLock.Unlock()
	if staged, ok := StagedUpdate(); ok {
		return staged, nil
	}

	available, resp, err := u.checkForUpdate(ctx)
	if err != nil {
		return ManualUpdateResult{}, err
	}
	if !available {
		return ManualUpdateResult{Status: "up_to_date"}, nil
	}
	if err := u.DownloadNewRelease(ctx, resp); err != nil {
		return ManualUpdateResult{}, fmt.Errorf("stage official Ollama %s: %w", resp.UpdateVersion, err)
	}
	return ManualUpdateResult{Status: "ready", Version: resp.UpdateVersion, NewlyStaged: true}, nil
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
