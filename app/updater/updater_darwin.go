package updater

// #cgo CFLAGS: -x objective-c
// #cgo LDFLAGS: -framework Webkit -framework Cocoa -framework LocalAuthentication -framework ServiceManagement -framework Security
// #include "updater_darwin.h"
// typedef const char cchar_t;
import "C"

import (
	"archive/zip"
	"debug/macho"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

const updateArchiveRoot = "Ollama.app"

const (
	maxArchiveEntries       = 100000
	maxArchiveExpandedBytes = 4 << 30
)

type bundleEntryScope int

const (
	bundleEntryRelative bundleEntryScope = iota
	bundleEntryWithArchiveRoot
)

var (
	appBackupDir   string
	SystemWidePath = "/Applications/Ollama.app"
)

var BundlePath = func() string {
	if bundle := alreadyMoved(); bundle != "" {
		return bundle
	}

	exe, err := os.Executable()
	if err != nil {
		return ""
	}

	// We also install this binary in Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel
	if filepath.Base(exe) == "Squirrel" &&
		filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exe)))))) == "Contents" {
		return filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exe))))))
	}

	// Make sure we're in a proper macOS app bundle structure (Contents/MacOS)
	if filepath.Base(filepath.Dir(exe)) != "MacOS" ||
		filepath.Base(filepath.Dir(filepath.Dir(exe))) != "Contents" {
		return ""
	}

	return filepath.Dir(filepath.Dir(filepath.Dir(exe)))
}()

func init() {
	VerifyDownload = verifyDownload
	VerifyCandidate = verifyDownloadCandidate
	Installer = "Ollama-darwin.zip"
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	var uts unix.Utsname
	if err := unix.Uname(&uts); err == nil {
		sysname := unix.ByteSliceToString(uts.Sysname[:])
		release := unix.ByteSliceToString(uts.Release[:])
		UserAgentOS = fmt.Sprintf("%s/%s", sysname, release)
	} else {
		slog.Warn("unable to determine OS version", "error", err)
		UserAgentOS = "Darwin"
	}

	// TODO handle failure modes here, and developer mode better...

	// Executable = Ollama.app/Contents/MacOS/Ollama

	UpgradeLogFile = filepath.Join(home, ".ollama", "logs", "upgrade.log")

	cacheDir, err := os.UserCacheDir()
	if err != nil {
		slog.Warn("unable to determine user cache dir, falling back to tmpdir", "error", err)
		cacheDir = os.TempDir()
	}
	appDataDir := filepath.Join(cacheDir, "ollama")
	UpgradeMarkerFile = filepath.Join(appDataDir, "upgraded")
	appBackupDir = filepath.Join(appDataDir, "backup")
	UpdateStageDir = filepath.Join(appDataDir, "updates")
}

func DoUpgrade(interactive bool) error {
	return DoUpgradeSource(PrimaryUpdateSource(), interactive)
}

func DoUpgradeSource(source string, interactive bool) error {
	// TODO use UpgradeLogFile to record the upgrade details from->to version, etc.

	bundle := getStagedUpdateForSource(source)
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}
	// Always verify at the point of use. Ready-state verification is cached for
	// display, but must not authorize a later installation.
	candidate, err := readUpdateCandidateMetadata(bundle)
	if err != nil {
		removeStagedUpdate(bundle)
		return fmt.Errorf("staged update metadata failed: %w", err)
	}
	if candidateSource(candidate) != source {
		return fmt.Errorf("staged update source %q does not match requested source %q", candidateSource(candidate), source)
	}
	// Hold the archive open from verification through extraction. The staged
	// pathname is writable by the current user, so reopening it after
	// verification would allow another process to substitute different bytes.
	archiveFile, err := os.Open(bundle)
	if err != nil {
		removeStagedUpdate(bundle)
		return fmt.Errorf("unable to open upgrade bundle %s: %w", bundle, err)
	}
	defer archiveFile.Close()
	heldArchivePath := fmt.Sprintf("/dev/fd/%d", archiveFile.Fd())
	if err := verifyCandidateChecksum(heldArchivePath, candidate); err != nil {
		removeStagedUpdate(bundle)
		return fmt.Errorf("staged update checksum failed: %w", err)
	}
	if err := verifyDownloadedCandidate(heldArchivePath, candidate); err != nil {
		removeStagedUpdate(bundle)
		return fmt.Errorf("staged update verification failed: %w", err)
	}
	archiveInfo, err := archiveFile.Stat()
	if err != nil {
		return fmt.Errorf("unable to inspect upgrade bundle %s: %w", bundle, err)
	}
	r, err := zip.NewReader(archiveFile, archiveInfo.Size())
	if err != nil {
		return fmt.Errorf("unable to open upgrade bundle %s: %w", bundle, err)
	}

	slog.Info("starting upgrade", "app", BundlePath, "update", bundle, "pid", os.Getpid(), "log", UpgradeLogFile)

	// TODO - in the future, consider shutting down the backend server now to give it
	// time to drain connections and stop allowing new connections while we perform the
	// actual upgrade to reduce the overall time to complete
	contentsName := filepath.Join(BundlePath, "Contents")
	appBackup := filepath.Join(appBackupDir, "Ollama.app")
	contentsOldName := filepath.Join(appBackup, "Contents")

	// Verify old doesn't exist yet
	if _, err := os.Stat(contentsOldName); err == nil {
		slog.Error("prior upgrade failed", "backup", contentsOldName)
		return fmt.Errorf("prior upgrade failed - please upgrade manually by installing the bundle")
	}
	if err := os.MkdirAll(appBackupDir, 0o755); err != nil {
		return fmt.Errorf("unable to create backup dir %s: %w", appBackupDir, err)
	}

	if err := validateBundleArchive(r.File); err != nil {
		return err
	}

	slog.Debug("temporarily staging old version", "staging", appBackup)
	if err := os.Rename(BundlePath, appBackup); err != nil {
		if !interactive {
			// We don't want to prompt for permission if we're attempting to upgrade at startup
			return fmt.Errorf("unable to upgrade in non-interactive mode with permission problems: %w", err)
		}
		// TODO actually inspect the error and look for permission problems before trying chown
		slog.Warn("unable to backup old version due to permission problems, changing ownership", "error", err)
		u, err := user.Current()
		if err != nil {
			return err
		}
		if !chownWithAuthorization(u.Username) {
			return fmt.Errorf("unable to change permissions to complete upgrade")
		}
		if err := os.Rename(BundlePath, appBackup); err != nil {
			return fmt.Errorf("unable to perform upgrade - failed to stage old version: %w", err)
		}
	}

	// Get ready to try to unwind a partial upgrade failure during unzip
	// If something goes wrong, we attempt to put the old version back.
	anyFailures := false
	defer func() {
		if anyFailures {
			slog.Warn("upgrade failures detected, attempting to revert")
			if err := os.RemoveAll(BundlePath); err != nil {
				slog.Warn("failed to remove partial upgrade", "path", BundlePath, "error", err)
				// At this point, we're basically hosed and the user will need to re-install
				return
			}
			if err := os.Rename(appBackup, BundlePath); err != nil {
				slog.Error("failed to revert to prior version", "path", contentsName, "error", err)
			}
		}
	}()

	// Bundle contents Ollama.app/Contents/...
	links := []*zip.File{}
	for _, f := range r.File {
		s := strings.SplitN(f.Name, "/", 2)
		if len(s) < 2 || s[1] == "" {
			slog.Debug("skipping", "file", f.Name)
			continue
		}
		name := s[1]
		if strings.HasSuffix(name, "/") {
			d, err := bundleEntryPath(BundlePath, name, bundleEntryRelative)
			if err != nil {
				anyFailures = true
				return err
			}
			err = os.MkdirAll(d, 0o755)
			if err != nil {
				anyFailures = true
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			// Defer links to the end
			links = append(links, f)
			continue
		}

		destName, err := bundleEntryPath(BundlePath, name, bundleEntryRelative)
		if err != nil {
			anyFailures = true
			return err
		}
		if err := extractBundleFile(f, destName, name); err != nil {
			anyFailures = true
			return err
		}
	}
	for _, f := range links {
		s := strings.SplitN(f.Name, "/", 2) // Strip off Ollama.app/
		if len(s) < 2 || s[1] == "" {
			slog.Debug("skipping link", "file", f.Name)
			continue
		}
		name := s[1]
		src, err := f.Open()
		if err != nil {
			anyFailures = true
			return err
		}
		buf, err := io.ReadAll(src)
		if err != nil {
			anyFailures = true
			return err
		}
		link := string(buf)
		if link == "" {
			anyFailures = true
			return fmt.Errorf("bundle contains empty symlink %s", f.Name)
		}
		if filepath.IsAbs(link) {
			anyFailures = true
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		if !validBundleLinkTarget(name, link, bundleEntryRelative) {
			anyFailures = true
			return fmt.Errorf("bundle contains invalid symlink %s -> %s", f.Name, link)
		}
		destName, err := bundleEntryPath(BundlePath, name, bundleEntryRelative)
		if err != nil {
			anyFailures = true
			return err
		}
		if err = os.Symlink(link, destName); err != nil {
			anyFailures = true
			return err
		}
	}

	f, err := os.OpenFile(UpgradeMarkerFile, os.O_RDONLY|os.O_CREATE, 0o666)
	if err != nil {
		slog.Warn("unable to create marker file", "file", UpgradeMarkerFile, "error", err)
	}
	f.Close()
	// Make sure to remove the staged download now that we succeeded so we don't inadvertently try again.
	cleanupOldDownloads(UpdateStageDir)

	return nil
}

func DoPostUpgradeCleanup() error {
	slog.Debug("post upgrade cleanup", "backup", appBackupDir)
	err := os.RemoveAll(appBackupDir)
	if err != nil {
		return err
	}
	slog.Debug("post upgrade cleanup", "old", UpgradeMarkerFile)
	return os.Remove(UpgradeMarkerFile)
}

func verifyDownload(bundle string) error {
	return verifyDownloadCandidate(bundle, UpdateResponse{Source: OfficialUpdateSource})
}

func verifyDownloadCandidate(bundle string, candidate UpdateResponse) error {
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}
	slog.Debug("verifying update", "bundle", bundle)

	// Extract zip file into a temporary location so we can run the cert verification routines
	dir, err := os.MkdirTemp("", "ollama_update_verify")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	r, err := zip.OpenReader(bundle)
	if err != nil {
		return fmt.Errorf("unable to open upgrade bundle %s: %w", bundle, err)
	}
	defer r.Close()
	if err := validateBundleArchive(r.File); err != nil {
		return err
	}
	links := []*zip.File{}
	for _, f := range r.File {
		if strings.HasSuffix(f.Name, "/") {
			d, err := bundleEntryPath(dir, f.Name, bundleEntryWithArchiveRoot)
			if err != nil {
				return err
			}
			err = os.MkdirAll(d, 0o755)
			if err != nil {
				return fmt.Errorf("failed to mkdir %s: %w", d, err)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			// Defer links to the end
			links = append(links, f)
			continue
		}
		destName, err := bundleEntryPath(dir, f.Name, bundleEntryWithArchiveRoot)
		if err != nil {
			return err
		}
		if err := extractBundleFile(f, destName, f.Name); err != nil {
			return err
		}
	}
	for _, f := range links {
		src, err := f.Open()
		if err != nil {
			return err
		}
		buf, err := io.ReadAll(src)
		if err != nil {
			return err
		}
		link := string(buf)
		if link == "" {
			return fmt.Errorf("bundle contains empty symlink %s", f.Name)
		}
		if filepath.IsAbs(link) {
			return fmt.Errorf("bundle contains absolute symlink %s -> %s", f.Name, link)
		}
		if !validBundleLinkTarget(f.Name, link, bundleEntryWithArchiveRoot) {
			return fmt.Errorf("bundle contains invalid symlink %s -> %s", f.Name, link)
		}
		destName, err := bundleEntryPath(dir, f.Name, bundleEntryWithArchiveRoot)
		if err != nil {
			return err
		}
		if err = os.Symlink(link, destName); err != nil {
			return err
		}
	}

	extractedBundle := filepath.Join(dir, updateArchiveRoot)
	if candidate.Source == MLXPreviewSource {
		if candidate.Architecture != "arm64" {
			return fmt.Errorf("unsupported MLX preview architecture %q", candidate.Architecture)
		}
		if err := verifyBundleArchitecture(extractedBundle, candidate.Architecture); err != nil {
			return err
		}
		if err := verifyExtractedBundleWithPolicy(extractedBundle, ExpectedBundleID, candidate.MarketingVersion, candidate.BuildVersion, ExpectedAppleTeamID, true); err != nil {
			return fmt.Errorf("signature verification failed: %s", err)
		}
		if err := verifyGatekeeperAssessment(extractedBundle); err != nil {
			return err
		}
		return nil
	}
	if candidate.Source == OfficialUpdateSource && candidate.MarketingVersion != "" {
		if candidate.Architecture != "arm64" {
			return fmt.Errorf("unsupported official Ollama architecture %q", candidate.Architecture)
		}
		if err := verifyBundleArchitecture(extractedBundle, candidate.Architecture); err != nil {
			return err
		}
		if err := verifyExtractedBundleWithPolicy(extractedBundle, "com.electron.ollama", candidate.MarketingVersion, candidate.BuildVersion, ExpectedOfficialAppleTeamID, true); err != nil {
			return fmt.Errorf("official signature verification failed: %s", err)
		}
		if err := verifyGatekeeperAssessment(extractedBundle); err != nil {
			return err
		}
		return nil
	}
	if err := verifyExtractedBundle(extractedBundle); err != nil {
		return fmt.Errorf("signature verification failed: %s", err)
	}
	return nil
}

func verifyGatekeeperAssessment(bundle string) error {
	cmd := exec.Command("/usr/sbin/spctl", "--assess", "--type", "execute", "--verbose=2", bundle)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("Gatekeeper assessment failed: %s: %w", strings.TrimSpace(string(output)), err)
	}
	return nil
}

func validateBundleArchive(files []*zip.File) error {
	if len(files) == 0 || len(files) > maxArchiveEntries {
		return fmt.Errorf("bundle archive contains %d entries; limit is %d", len(files), maxArchiveEntries)
	}
	seen := make(map[string]os.FileMode, len(files))
	folded := make(map[string]string, len(files))
	var expanded uint64
	for _, file := range files {
		if len(file.Name) > 1024 {
			return fmt.Errorf("bundle path exceeds 1024 bytes: %s", file.Name[:128])
		}
		cleanName := filepath.Clean(filepath.FromSlash(file.Name))
		if _, err := bundleEntryPath(".", file.Name, bundleEntryWithArchiveRoot); err != nil {
			return err
		}
		mode := file.Mode()
		if mode.IsDir() != strings.HasSuffix(file.Name, "/") {
			return fmt.Errorf("bundle entry type does not match its path: %s", file.Name)
		}
		if mode.Type() != 0 && !mode.IsDir() && mode&os.ModeSymlink == 0 {
			return fmt.Errorf("bundle contains unsupported entry type: %s", file.Name)
		}
		if previous, ok := seen[cleanName]; ok {
			return fmt.Errorf("bundle contains duplicate or conflicting path %s (%s and %s)", file.Name, previous, mode)
		}
		foldedName := strings.ToLower(cleanName)
		if previous, ok := folded[foldedName]; ok {
			return fmt.Errorf("bundle contains case-conflicting paths %s and %s", previous, file.Name)
		}
		seen[cleanName] = mode
		folded[foldedName] = file.Name
		if !mode.IsDir() {
			expanded += file.UncompressedSize64
			if expanded > maxArchiveExpandedBytes {
				return fmt.Errorf("bundle expanded size exceeds %d bytes", maxArchiveExpandedBytes)
			}
		}
	}
	for name := range seen {
		for parent := filepath.Dir(name); parent != "." && parent != name; parent = filepath.Dir(parent) {
			if parentMode, ok := seen[parent]; ok && !parentMode.IsDir() {
				if parentMode&os.ModeSymlink != 0 {
					return fmt.Errorf("bundle path %s traverses symlink entry %s", name, parent)
				}
				return fmt.Errorf("bundle path %s traverses non-directory entry %s", name, parent)
			}
		}
	}
	return nil
}

func bundleEntryPath(root, name string, scope bundleEntryScope) (string, error) {
	cleanName := filepath.Clean(filepath.FromSlash(name))
	if !filepath.IsLocal(cleanName) {
		return "", fmt.Errorf("bundle contains invalid path: %s", name)
	}
	if scope == bundleEntryWithArchiveRoot && cleanName != updateArchiveRoot &&
		!strings.HasPrefix(cleanName, updateArchiveRoot+string(os.PathSeparator)) {
		return "", fmt.Errorf("bundle contains invalid path: %s", name)
	}
	return filepath.Join(root, cleanName), nil
}

func extractBundleFile(f *zip.File, destName, name string) error {
	src, err := f.Open()
	if err != nil {
		return fmt.Errorf("failed to open bundle file %s: %w", name, err)
	}
	defer src.Close()

	d := filepath.Dir(destName)
	if _, err := os.Stat(d); err != nil {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return fmt.Errorf("failed to mkdir %s: %w", d, err)
		}
	}

	destFile, err := os.OpenFile(destName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return fmt.Errorf("failed to open output file %s: %w", destName, err)
	}
	defer destFile.Close()

	written, err := io.Copy(destFile, io.LimitReader(src, int64(f.UncompressedSize64)+1))
	if err != nil {
		return fmt.Errorf("failed to open extract file %s: %w", destName, err)
	}
	if written != int64(f.UncompressedSize64) {
		return fmt.Errorf("bundle file %s expanded to an unexpected size", name)
	}
	return nil
}

func validBundleLinkTarget(name, link string, scope bundleEntryScope) bool {
	cleanTarget := filepath.Clean(filepath.Join(filepath.Dir(filepath.FromSlash(name)), filepath.FromSlash(link)))
	if !filepath.IsLocal(cleanTarget) {
		return false
	}
	return scope == bundleEntryRelative || cleanTarget == updateArchiveRoot ||
		strings.HasPrefix(cleanTarget, updateArchiveRoot+string(os.PathSeparator))
}

// If we detect an upgrade bundle, attempt to upgrade at startup
func DoUpgradeAtStartup() error {
	if AutomaticUpdatesDisabled() || !AutomaticInstallAllowed() {
		return fmt.Errorf("automatic updates disabled by build")
	}

	bundle := getStagedUpdate()
	if bundle == "" {
		return fmt.Errorf("failed to lookup downloads")
	}

	if BundlePath == "" {
		return fmt.Errorf("unable to upgrade at startup, app in development mode")
	}

	slog.Info("performing update at startup", "bundle", bundle)
	return DoUpgrade(false)
}

func getStagedUpdate() string {
	return getStagedUpdateForSource(PrimaryUpdateSource())
}

func getStagedUpdateForSource(source string) string {
	return stagedUpdatePath(source, "zip")
}

func IsUpdatePending() bool {
	return getStagedUpdate() != ""
}

func chownWithAuthorization(user string) bool {
	u := C.CString(user)
	defer C.free(unsafe.Pointer(u))
	return bool(C.chownWithAuthorization(u))
}

func verifyExtractedBundle(path string) error {
	return verifyExtractedBundleWithPolicy(path, "", "", "", "", false)
}

func verifyExtractedBundleWithPolicy(path, bundleID, marketingVersion, buildVersion, teamID string, requirePinnedIdentity bool) error {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	bid := C.CString(bundleID)
	defer C.free(unsafe.Pointer(bid))
	marketing := C.CString(marketingVersion)
	defer C.free(unsafe.Pointer(marketing))
	build := C.CString(buildVersion)
	defer C.free(unsafe.Pointer(build))
	team := C.CString(teamID)
	defer C.free(unsafe.Pointer(team))
	resp := C.verifyExtractedBundle(p, bid, marketing, build, team, C.bool(requirePinnedIdentity))
	if resp == nil {
		return nil
	}
	defer C.free(unsafe.Pointer(resp))
	return errors.New(C.GoString(resp))
}

func verifyBundleArchitecture(bundle, architecture string) error {
	executable := filepath.Join(bundle, "Contents", "MacOS", "Ollama")
	if fat, err := macho.OpenFat(executable); err == nil {
		defer fat.Close()
		for _, arch := range fat.Arches {
			if architecture == "arm64" && arch.Cpu == macho.CpuArm64 {
				return nil
			}
		}
		return fmt.Errorf("bundle executable does not contain %s", architecture)
	}
	thin, err := macho.Open(executable)
	if err != nil {
		return fmt.Errorf("inspect bundle architecture: %w", err)
	}
	defer thin.Close()
	if architecture == "arm64" && thin.Cpu == macho.CpuArm64 {
		return nil
	}
	return fmt.Errorf("bundle executable architecture %s does not match %s", thin.Cpu, architecture)
}

//export goLogInfo
func goLogInfo(msg *C.cchar_t) {
	slog.Info(C.GoString(msg))
}

//export goLogDebug
func goLogDebug(msg *C.cchar_t) {
	slog.Debug(C.GoString(msg))
}

func alreadyMoved() string {
	// Respect users intent if they chose "keep" vs. "replace" when dragging to Applications
	installedAppPaths, err := filepath.Glob(filepath.Join(
		strings.TrimSuffix(SystemWidePath, filepath.Ext(SystemWidePath))+"*"+filepath.Ext(SystemWidePath),
		"Contents", "MacOS", "Ollama"))
	if err != nil {
		slog.Warn("failed to lookup installed app paths", "error", err)
		return ""
	}
	exe, err := os.Executable()
	if err != nil {
		slog.Warn("failed to resolve executable", "error", err)
		return ""
	}
	self, err := os.Stat(exe)
	if err != nil {
		slog.Warn("failed to stat running executable", "path", exe, "error", err)
		return ""
	}
	selfSys := self.Sys().(*syscall.Stat_t)
	for _, installedAppPath := range installedAppPaths {
		app, err := os.Stat(installedAppPath)
		if err != nil {
			slog.Debug("failed to stat installed app path", "path", installedAppPath, "error", err)
			continue
		}
		appSys := app.Sys().(*syscall.Stat_t)

		if appSys.Ino == selfSys.Ino {
			return filepath.Dir(filepath.Dir(filepath.Dir(installedAppPath)))
		}
	}
	return ""
}
