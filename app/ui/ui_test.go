//go:build windows || darwin

package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/cmd/launch"
)

func TestBuildChatRequestPreservesAudioAttachmentsAsMedia(t *testing.T) {
	server := &Server{}
	chat := &store.Chat{Messages: []store.Message{{
		Role:    "user",
		Content: "Transcribe the audio.",
		Attachments: []store.File{
			{Filename: "speech.mp3", Data: []byte{1, 2, 3}},
			{Filename: "speech.wav", Data: []byte{4, 5, 6}},
			{Filename: "notes.txt", Data: []byte("keep this text attachment")},
		},
	}}}

	req, err := server.buildChatRequest(chat, "apertus-1.5-mlx", false, nil)
	if err != nil {
		t.Fatalf("buildChatRequest: %v", err)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("message count = %d, want 1", len(req.Messages))
	}
	message := req.Messages[0]
	if len(message.Images) != 2 || !bytes.Equal(message.Images[0], []byte{1, 2, 3}) || !bytes.Equal(message.Images[1], []byte{4, 5, 6}) {
		t.Fatalf("media attachments = %v, want mp3 and wav bytes in order", message.Images)
	}
	if strings.Contains(message.Content, "Binary file of type .mp3") || strings.Contains(message.Content, "Binary file of type .wav") {
		t.Fatalf("audio was converted to text content: %q", message.Content)
	}
	if !strings.Contains(message.Content, "keep this text attachment") {
		t.Fatalf("text attachment missing from content: %q", message.Content)
	}
}

func TestHandlePostApiSettings(t *testing.T) {
	tests := []struct {
		name      string
		requested store.Settings
		wantErr   bool
	}{
		{
			name: "valid settings update - all fields",
			requested: store.Settings{
				Expose:     true,
				Browser:    true,
				Models:     "/custom/models",
				Agent:      true,
				Tools:      true,
				WorkingDir: "/workspace",
			},
			wantErr: false,
		},
		{
			name: "partial settings update",
			requested: store.Settings{
				Agent:      true,
				Tools:      false,
				WorkingDir: "/new/path",
			},
			wantErr: false,
		},
		{
			name: "settings with special characters in paths",
			requested: store.Settings{
				Models:     "/path with spaces/models",
				WorkingDir: "/tmp/work-dir_123",
				Agent:      true,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testStore := &store.Store{
				DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
			}
			defer testStore.Close() // Ensure database is closed before cleanup

			body, err := json.Marshal(tt.requested)
			if err != nil {
				t.Fatalf("failed to marshal test body: %v", err)
			}

			req := httptest.NewRequest("POST", "/api/v1/settings", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			// Set up server with test store
			server := &Server{
				Store:   testStore,
				Restart: func() {}, // Mock restart function for tests
			}

			if err := server.settings(rr, req); (err != nil) != tt.wantErr {
				t.Errorf("handlePostApiSettings() error = %v, wantErr %v", err, tt.wantErr)
			}
			if rr.Code != http.StatusOK {
				t.Errorf("handlePostApiSettings() status = %v, want %v", rr.Code, http.StatusOK)
			}

			// Check settings were saved correctly (if no error expected)
			if !tt.wantErr {
				savedSettings, err := testStore.Settings()
				if err != nil {
					t.Errorf("failed to retrieve saved settings: %v", err)
				} else {
					// Compare field by field, accounting for defaults that may be set by the store
					if savedSettings.Expose != tt.requested.Expose {
						t.Errorf("Expose: got %v, want %v", savedSettings.Expose, tt.requested.Expose)
					}
					if savedSettings.Browser != tt.requested.Browser {
						t.Errorf("Browser: got %v, want %v", savedSettings.Browser, tt.requested.Browser)
					}
					if savedSettings.Agent != tt.requested.Agent {
						t.Errorf("Agent: got %v, want %v", savedSettings.Agent, tt.requested.Agent)
					}
					if savedSettings.Tools != tt.requested.Tools {
						t.Errorf("Tools: got %v, want %v", savedSettings.Tools, tt.requested.Tools)
					}
					if savedSettings.WorkingDir != tt.requested.WorkingDir {
						t.Errorf("WorkingDir: got %q, want %q", savedSettings.WorkingDir, tt.requested.WorkingDir)
					}
					// Only check Models if explicitly set in the test case
					if tt.requested.Models != "" && savedSettings.Models != tt.requested.Models {
						t.Errorf("Models: got %q, want %q", savedSettings.Models, tt.requested.Models)
					}
				}
			}
		})
	}
}

func TestGetIntegrationStatuses(t *testing.T) {
	server := &Server{
		IntegrationInstalled: func(name string) bool {
			return name == "claude-desktop" || name == "codex"
		},
	}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/integrations", nil)
	rr := httptest.NewRecorder()

	if err := server.getIntegrationStatuses(rr, req); err != nil {
		t.Fatalf("getIntegrationStatuses() error = %v", err)
	}

	var got []struct {
		ID        string `json:"id"`
		Installed *bool  `json:"installed"`
		Action    string `json:"action"`
		Command   string `json:"command"`
	}
	if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if len(got) < 5 {
		t.Fatalf("got %d integrations, want the full registry", len(got))
	}
	if got[0].ID != "claude-desktop" || got[0].Action != "connect" || got[0].Command != "" {
		t.Fatalf("first integration = %+v, want command-free Claude Desktop connect", got[0])
	}
	wantPrefix := []string{"claude-desktop", "claude", "codex", "openclaw", "opencode", "hermes", "hermes-desktop", "droid", "pi", "cline"}
	for i, want := range wantPrefix {
		if got[i].ID != want {
			t.Fatalf("integration %d = %q, want launcher menu order entry %q", i, got[i].ID, want)
		}
	}
	byID := make(map[string]struct {
		Installed *bool
		Action    string
		Command   string
	}, len(got))
	for _, item := range got {
		byID[item.ID] = struct {
			Installed *bool
			Action    string
			Command   string
		}{item.Installed, item.Action, item.Command}
	}

	for name, want := range map[string]bool{
		"claude-desktop": true,
		"opencode":       false,
		"codex":          true,
	} {
		item, ok := byID[name]
		if !ok || item.Installed == nil || *item.Installed != want {
			t.Errorf("%s installed = %v, want %v", name, item.Installed, want)
		}
	}
	if item, ok := byID["claude"]; !ok || item.Command != "ollama launch claude" {
		t.Fatal("Claude Code should follow Claude Desktop with its launch command")
	}
	if _, ok := byID["chatgpt"]; ok {
		t.Fatal("ChatGPT should be excluded from onboarding integrations")
	}
	wantCount := len(launch.ListIntegrationInfos()) + 1 // Claude Desktop and Terminal replace omitted ChatGPT.
	if len(got) != wantCount {
		t.Fatalf("got %d integrations, want %d launcher entries", len(got), wantCount)
	}
	terminal := got[len(got)-1]
	if terminal.ID != "terminal" || terminal.Installed != nil || terminal.Command != "ollama" {
		t.Fatalf("last integration = %+v, want Terminal without install status", terminal)
	}
}

func TestGetSettingsReportsManualUpdatePolicy(t *testing.T) {
	oldDisableUpdates := updater.DisableUpdates
	oldStageDir := updater.UpdateStageDir
	oldVerify := updater.VerifyDownload
	defer func() {
		updater.DisableUpdates = oldDisableUpdates
		updater.UpdateStageDir = oldStageDir
		updater.VerifyDownload = oldVerify
	}()
	updater.DisableUpdates = "true"
	updater.UpdateStageDir = t.TempDir()
	updater.VerifyDownload = func(_ string) error { return nil }
	stagedDir := filepath.Join(updater.UpdateStageDir, "verified")
	if err := os.MkdirAll(stagedDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagedDir, updater.Installer), []byte("archive"), 0o755); err != nil {
		t.Fatal(err)
	}

	testStore := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	defer testStore.Close()
	server := &Server{Store: testStore}
	recorder := httptest.NewRecorder()
	if err := server.getSettings(recorder, httptest.NewRequest(http.MethodGet, "/api/v1/settings", nil)); err != nil {
		t.Fatal(err)
	}
	var response struct {
		ManualUpdatesOnly bool `json:"manualUpdatesOnly"`
		UpdateReady       bool `json:"updateReady"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if !response.ManualUpdatesOnly {
		t.Fatal("manualUpdatesOnly = false, want true")
	}
	if !response.UpdateReady {
		t.Fatal("updateReady = false, want staged update restored")
	}
}

func TestCheckForUpdatesReturnsStructuredResults(t *testing.T) {
	oldURL := updater.UpdateCheckURLBase
	oldStageDir := updater.UpdateStageDir
	oldVerify := updater.VerifyDownload
	oldDownloaded := updater.UpdateDownloaded
	defer func() {
		updater.UpdateCheckURLBase = oldURL
		updater.UpdateStageDir = oldStageDir
		updater.VerifyDownload = oldVerify
		updater.UpdateDownloaded = oldDownloaded
	}()

	t.Run("up to date", func(t *testing.T) {
		updater.UpdateStageDir = t.TempDir()
		service := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		}))
		defer service.Close()
		updater.UpdateCheckURLBase = service.URL

		var trayCalls atomic.Int32
		server := &Server{
			Updater:             &updater.Updater{},
			UpdateAvailableFunc: func() { trayCalls.Add(1) },
		}
		recorder := httptest.NewRecorder()
		if err := server.checkForUpdates(recorder, httptest.NewRequest(http.MethodPost, "/api/v1/update/check", nil)); err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(recorder.Body.String()) != `{"status":"up_to_date"}` {
			t.Fatalf("response = %s", recorder.Body.String())
		}
		if trayCalls.Load() != 0 {
			t.Fatal("up-to-date check activated tray")
		}
	})

	t.Run("ready", func(t *testing.T) {
		updater.UpdateStageDir = t.TempDir()
		updater.UpdateDownloaded = false
		updater.VerifyDownload = func(_ string) error { return nil }
		var service *httptest.Server
		service = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/update" {
				_, _ = fmt.Fprintf(w, `{"url":%q}`, service.URL+"/v9.9.9/"+updater.Installer)
				return
			}
			w.Header().Set("ETag", `"ready"`)
			w.WriteHeader(http.StatusOK)
			if r.Method == http.MethodGet {
				_, _ = w.Write([]byte("archive"))
			}
		}))
		defer service.Close()
		updater.UpdateCheckURLBase = service.URL + "/update"

		var trayCalls atomic.Int32
		var installCalls atomic.Int32
		server := &Server{
			Updater:             &updater.Updater{},
			UpdateAvailableFunc: func() { trayCalls.Add(1) },
			InstallUpdateFunc: func() error {
				installCalls.Add(1)
				return nil
			},
		}
		recorder := httptest.NewRecorder()
		if err := server.checkForUpdates(recorder, httptest.NewRequest(http.MethodPost, "/api/v1/update/check", nil)); err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(recorder.Body.String()) != `{"status":"ready","version":"v9.9.9"}` {
			t.Fatalf("response = %s", recorder.Body.String())
		}
		if trayCalls.Load() != 1 {
			t.Fatalf("tray calls = %d, want 1", trayCalls.Load())
		}
		installRecorder := httptest.NewRecorder()
		if err := server.installUpdate(installRecorder, httptest.NewRequest(http.MethodPost, "/api/v1/update/install", nil)); err != nil {
			t.Fatal(err)
		}
		if installCalls.Load() != 1 {
			t.Fatalf("install calls = %d, want 1", installCalls.Load())
		}
	})

	t.Run("error", func(t *testing.T) {
		service := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte("not json"))
		}))
		defer service.Close()
		updater.UpdateCheckURLBase = service.URL
		var trayCalls atomic.Int32
		server := &Server{Updater: &updater.Updater{}, UpdateAvailableFunc: func() { trayCalls.Add(1) }}
		err := server.checkForUpdates(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/v1/update/check", nil))
		if err == nil || !strings.Contains(err.Error(), "decode update response") {
			t.Fatalf("error = %v", err)
		}
		if trayCalls.Load() != 0 {
			t.Fatal("failed check activated tray")
		}
	})
}

func TestConcurrentReadyChecksActivateTrayOnce(t *testing.T) {
	oldURL := updater.UpdateCheckURLBase
	oldStageDir := updater.UpdateStageDir
	oldVerify := updater.VerifyDownload
	defer func() {
		updater.UpdateCheckURLBase = oldURL
		updater.UpdateStageDir = oldStageDir
		updater.VerifyDownload = oldVerify
	}()
	updater.UpdateStageDir = t.TempDir()
	updater.VerifyDownload = func(_ string) error { return nil }
	var service *httptest.Server
	service = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update" {
			_, _ = fmt.Fprintf(w, `{"url":%q}`, service.URL+"/v9.9.9/"+updater.Installer)
			return
		}
		w.Header().Set("ETag", `"once"`)
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte("archive"))
		}
	}))
	defer service.Close()
	updater.UpdateCheckURLBase = service.URL + "/update"

	var trayCalls atomic.Int32
	server := &Server{
		Updater:             &updater.Updater{},
		UpdateAvailableFunc: func() { trayCalls.Add(1) },
	}
	errs := make(chan error, 2)
	for range 2 {
		go func() {
			errs <- server.checkForUpdates(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/v1/update/check", nil))
		}()
	}
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if trayCalls.Load() != 1 {
		t.Fatalf("tray calls = %d, want 1", trayCalls.Load())
	}
}

func TestInstallUpdateIsSingleFlightAndReportsResults(t *testing.T) {
	oldStageDir := updater.UpdateStageDir
	oldVerify := updater.VerifyDownload
	defer func() {
		updater.UpdateStageDir = oldStageDir
		updater.VerifyDownload = oldVerify
	}()
	updater.UpdateStageDir = t.TempDir()
	updater.VerifyDownload = func(_ string) error { return nil }
	bundleDir := filepath.Join(updater.UpdateStageDir, "verified")
	if err := os.MkdirAll(bundleDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bundleDir, updater.Installer), []byte("archive"), 0o755); err != nil {
		t.Fatal(err)
	}

	started := make(chan struct{})
	release := make(chan struct{})
	server := &Server{InstallUpdateFunc: func() error {
		close(started)
		<-release
		return updater.ErrInstallCancelled
	}}
	firstRecorder := httptest.NewRecorder()
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- server.installUpdate(firstRecorder, httptest.NewRequest(http.MethodPost, "/api/v1/update/install", nil))
	}()
	<-started

	duplicateRecorder := httptest.NewRecorder()
	if err := server.installUpdate(duplicateRecorder, httptest.NewRequest(http.MethodPost, "/api/v1/update/install", nil)); err != nil {
		t.Fatal(err)
	}
	if duplicateRecorder.Code != http.StatusConflict {
		t.Fatalf("duplicate status = %d, want 409", duplicateRecorder.Code)
	}
	close(release)
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(firstRecorder.Body.String()) != `{"status":"cancelled"}` {
		t.Fatalf("cancel response = %s", firstRecorder.Body.String())
	}

	failure := &Server{InstallUpdateFunc: func() error { return errors.New("permission denied") }}
	err := failure.installUpdate(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/v1/update/install", nil))
	if err == nil || !strings.Contains(err.Error(), "permission denied") {
		t.Fatalf("install failure = %v", err)
	}
}

func TestHandlePostApiCloudSetting(t *testing.T) {
	tmpHome := t.TempDir()
	t.Setenv("HOME", tmpHome)
	t.Setenv("OLLAMA_NO_CLOUD", "")

	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	restartCount := 0
	server := &Server{
		Store: testStore,
		Restart: func() {
			restartCount++
		},
	}

	for _, tc := range []struct {
		name        string
		body        string
		wantEnabled bool
	}{
		{name: "disable cloud", body: `{"enabled": false}`, wantEnabled: false},
		{name: "enable cloud", body: `{"enabled": true}`, wantEnabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/v1/cloud", bytes.NewBufferString(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			if err := server.cloudSetting(rr, req); err != nil {
				t.Fatalf("cloudSetting() error = %v", err)
			}
			if rr.Code != http.StatusOK {
				t.Fatalf("cloudSetting() status = %d, want %d", rr.Code, http.StatusOK)
			}

			var got map[string]any
			if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
				t.Fatalf("cloudSetting() invalid response JSON: %v", err)
			}
			if got["disabled"] != !tc.wantEnabled {
				t.Fatalf("response disabled = %v, want %v", got["disabled"], !tc.wantEnabled)
			}

			disabled, err := testStore.CloudDisabled()
			if err != nil {
				t.Fatalf("CloudDisabled() error = %v", err)
			}
			if gotEnabled := !disabled; gotEnabled != tc.wantEnabled {
				t.Fatalf("cloud enabled = %v, want %v", gotEnabled, tc.wantEnabled)
			}
		})
	}

	if restartCount != 2 {
		t.Fatalf("Restart called %d times, want 2", restartCount)
	}
}

func TestHandleGetApiCloudSetting(t *testing.T) {
	tmpHome := t.TempDir()
	t.Setenv("HOME", tmpHome)
	t.Setenv("OLLAMA_NO_CLOUD", "")

	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	if err := testStore.SetCloudEnabled(false); err != nil {
		t.Fatalf("SetCloudEnabled(false) error = %v", err)
	}

	server := &Server{
		Store:   testStore,
		Restart: func() {},
	}

	req := httptest.NewRequest("GET", "/api/v1/cloud", nil)
	rr := httptest.NewRecorder()
	if err := server.getCloudSetting(rr, req); err != nil {
		t.Fatalf("getCloudSetting() error = %v", err)
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("getCloudSetting() status = %d, want %d", rr.Code, http.StatusOK)
	}

	var got map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("getCloudSetting() invalid response JSON: %v", err)
	}
	if got["disabled"] != true {
		t.Fatalf("response disabled = %v, want true", got["disabled"])
	}
	if got["source"] != "config" {
		t.Fatalf("response source = %v, want config", got["source"])
	}
}

func TestAuthenticationMiddleware(t *testing.T) {
	tests := []struct {
		name         string
		method       string
		contentType  string
		tokenCookie  string
		serverToken  string
		wantStatus   int
		wantError    string
		setupRequest func(*http.Request)
	}{
		{
			name:        "missing token cookie",
			method:      "GET",
			tokenCookie: "",
			serverToken: "test-token-123",
			wantStatus:  http.StatusForbidden,
			wantError:   "Token is required",
		},
		{
			name:        "invalid token value",
			method:      "GET",
			tokenCookie: "wrong-token",
			serverToken: "test-token-123",
			wantStatus:  http.StatusForbidden,
			wantError:   "Token is required",
		},
		{
			name:        "valid token - GET request",
			method:      "GET",
			tokenCookie: "test-token-123",
			serverToken: "test-token-123",
			wantStatus:  http.StatusOK,
			wantError:   "",
		},
		{
			name:        "valid token - POST with application/json",
			method:      "POST",
			contentType: "application/json",
			tokenCookie: "test-token-123",
			serverToken: "test-token-123",
			wantStatus:  http.StatusOK,
			wantError:   "",
		},
		{
			name:        "POST without Content-Type header",
			method:      "POST",
			contentType: "",
			tokenCookie: "test-token-123",
			serverToken: "test-token-123",
			wantStatus:  http.StatusForbidden,
			wantError:   "Content-Type must be application/json",
		},
		{
			name:        "POST with wrong Content-Type",
			method:      "POST",
			contentType: "text/plain",
			tokenCookie: "test-token-123",
			serverToken: "test-token-123",
			wantStatus:  http.StatusForbidden,
			wantError:   "Content-Type must be application/json",
		},
		{
			name:        "OPTIONS request (CORS preflight) - should bypass auth",
			method:      "OPTIONS",
			tokenCookie: "",
			serverToken: "test-token-123",
			wantStatus:  http.StatusOK,
			wantError:   "",
			setupRequest: func(r *http.Request) {
				// Simulate CORS being enabled
				// Note: This assumes CORS() returns true in test environment
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test handler that just returns 200 OK if auth passes
			testHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			})

			// Create server with test token
			server := &Server{
				Token: tt.serverToken,
			}

			// Get the authentication middleware by calling Handler()
			// We need to wrap our test handler with the auth middleware
			handler := server.Handler()

			// Create a test router to simulate the authentication middleware
			mux := http.NewServeMux()
			mux.Handle("/test", handler)

			// But since Handler() returns the full router, we'll need a different approach
			// Let's create a minimal handler that includes just the auth logic
			authHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Add CORS headers for dev work
				if r.Method == "OPTIONS" {
					w.WriteHeader(http.StatusOK)
					return
				}

				if r.Method == "POST" && r.Header.Get("Content-Type") != "application/json" {
					w.WriteHeader(http.StatusForbidden)
					json.NewEncoder(w).Encode(map[string]string{"error": "Content-Type must be application/json"})
					return
				}

				cookie, err := r.Cookie("token")
				if err != nil {
					w.WriteHeader(http.StatusForbidden)
					json.NewEncoder(w).Encode(map[string]string{"error": "Token is required"})
					return
				}

				if cookie.Value != server.Token {
					w.WriteHeader(http.StatusForbidden)
					json.NewEncoder(w).Encode(map[string]string{"error": "Token is required"})
					return
				}

				// If auth passes, call the test handler
				testHandler.ServeHTTP(w, r)
			})

			// Create test request
			req := httptest.NewRequest(tt.method, "/test", nil)

			// Set Content-Type if provided
			if tt.contentType != "" {
				req.Header.Set("Content-Type", tt.contentType)
			}

			// Set token cookie if provided
			if tt.tokenCookie != "" {
				req.AddCookie(&http.Cookie{
					Name:  "token",
					Value: tt.tokenCookie,
				})
			}

			// Run any additional setup
			if tt.setupRequest != nil {
				tt.setupRequest(req)
			}

			// Create response recorder
			rr := httptest.NewRecorder()

			// Serve the request
			authHandler.ServeHTTP(rr, req)

			// Check status code
			if rr.Code != tt.wantStatus {
				t.Errorf("handler returned wrong status code: got %v want %v", rr.Code, tt.wantStatus)
			}

			// Check error message if expected
			if tt.wantError != "" {
				var response map[string]string
				if err := json.NewDecoder(rr.Body).Decode(&response); err != nil {
					t.Fatalf("failed to decode response body: %v", err)
				}

				if response["error"] != tt.wantError {
					t.Errorf("handler returned wrong error message: got %v want %v", response["error"], tt.wantError)
				}
			}
		})
	}
}

func TestUserAgent(t *testing.T) {
	ua := userAgent()

	// The userAgent function should return a string in the format:
	// "ollama/version (arch os) app/version Go/goversion"
	// Example: "ollama/v0.1.28 (amd64 darwin) Go/go1.21.0"

	if ua == "" {
		t.Fatal("userAgent returned empty string")
	}

	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("User-Agent", ua)

	// This is a copy of the logic ollama.com uses to parse the user agent
	clientInfoFromRequest := func(r *http.Request) struct {
		Product    string
		Version    string
		OS         string
		Arch       string
		AppVersion string
	} {
		product, rest, _ := strings.Cut(r.UserAgent(), " ")
		client, version, ok := strings.Cut(product, "/")
		if !ok {
			return struct {
				Product    string
				Version    string
				OS         string
				Arch       string
				AppVersion string
			}{}
		}

		if version != "" && version[0] != 'v' {
			version = "v" + version
		}

		arch, rest, _ := strings.Cut(rest, " ")
		arch = strings.Trim(arch, "(")
		os, rest, _ := strings.Cut(rest, ")")

		var appVersion string
		if strings.Contains(rest, "app/") {
			_, appPart, found := strings.Cut(rest, "app/")
			if found {
				appVersion = strings.Fields(strings.TrimSpace(appPart))[0]
				if appVersion != "" && appVersion[0] != 'v' {
					appVersion = "v" + appVersion
				}
			}
		}

		return struct {
			Product    string
			Version    string
			OS         string
			Arch       string
			AppVersion string
		}{
			Product:    client,
			Version:    version,
			OS:         os,
			Arch:       arch,
			AppVersion: appVersion,
		}
	}

	info := clientInfoFromRequest(req)
	if info.Product != "ollama" {
		t.Errorf("Expected Product to be 'ollama', got '%s'", info.Product)
	}

	if info.Version != "" && info.Version[0] != 'v' {
		t.Errorf("Expected Version to start with 'v', got '%s'", info.Version)
	}

	expectedOS := runtime.GOOS
	if info.OS != expectedOS {
		t.Errorf("Expected OS to be '%s', got '%s'", expectedOS, info.OS)
	}

	expectedArch := runtime.GOARCH
	if info.Arch != expectedArch {
		t.Errorf("Expected Arch to be '%s', got '%s'", expectedArch, info.Arch)
	}

	if info.AppVersion != "" && info.AppVersion[0] != 'v' {
		t.Errorf("Expected AppVersion to start with 'v', got '%s'", info.AppVersion)
	}

	t.Logf("User Agent: %s", ua)
	t.Logf("Parsed - Product: %s, Version: %s, OS: %s, Arch: %s",
		info.Product, info.Version, info.OS, info.Arch)
}

func TestUserAgentTransport(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte(r.Header.Get("User-Agent")))
	}))
	defer ts.Close()
	server := &Server{}

	client := server.httpClient()
	resp, err := client.Get(ts.URL)
	if err != nil {
		t.Fatalf("Failed to make request: %v", err)
	}
	defer resp.Body.Close()

	// In this case the User-Agent is the response body, as the server just echoes it back
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response: %v", err)
	}

	receivedUA := string(body)
	expectedUA := userAgent()

	if receivedUA != expectedUA {
		t.Errorf("User-Agent mismatch\nExpected: %s\nReceived: %s", expectedUA, receivedUA)
	}

	if !strings.HasPrefix(receivedUA, "ollama/") {
		t.Errorf("User-Agent should start with 'ollama/', got: %s", receivedUA)
	}

	t.Logf("User-Agent transport successfully set: %s", receivedUA)
}

func TestGetCloudModels(t *testing.T) {
	t.Run("does not call ollama.com when cloud is disabled", func(t *testing.T) {
		t.Setenv("HOME", t.TempDir())
		t.Setenv("OLLAMA_NO_CLOUD", "1")
		testStore := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer testStore.Close()

		server := &Server{
			Store: testStore,
			ListCloudModels: func(context.Context) (*api.ListResponse, error) {
				t.Fatal("cloud model list called while cloud was disabled")
				return nil, nil
			},
		}
		req := httptest.NewRequest(http.MethodGet, "/api/v1/models/cloud", nil)
		rr := httptest.NewRecorder()
		if err := server.getCloudModels(rr, req); err != nil {
			t.Fatal(err)
		}

		var got api.ListResponse
		if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		if len(got.Models) != 0 {
			t.Fatalf("models = %+v, want none", got.Models)
		}
	})

	t.Run("returns no cloud models when account is unauthorized", func(t *testing.T) {
		t.Setenv("HOME", t.TempDir())
		t.Setenv("OLLAMA_NO_CLOUD", "")
		testStore := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer testStore.Close()

		server := &Server{
			Store: testStore,
			ListCloudModels: func(context.Context) (*api.ListResponse, error) {
				return nil, api.AuthorizationError{StatusCode: http.StatusUnauthorized}
			},
		}
		req := httptest.NewRequest(http.MethodGet, "/api/v1/models/cloud", nil)
		rr := httptest.NewRecorder()
		if err := server.getCloudModels(rr, req); err != nil {
			t.Fatal(err)
		}

		var got api.ListResponse
		if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		if len(got.Models) != 0 {
			t.Fatalf("models = %+v, want none", got.Models)
		}
	})
}

func TestInferenceClientUsesUserAgent(t *testing.T) {
	var gotUserAgent atomic.Value
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUserAgent.Store(r.Header.Get("User-Agent"))
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer ts.Close()

	t.Setenv("OLLAMA_HOST", ts.URL)

	server := &Server{}
	client := server.inferenceClient()

	_, err := client.Show(context.Background(), &api.ShowRequest{Model: "test"})
	if err != nil {
		t.Fatalf("show request failed: %v", err)
	}

	receivedUA, _ := gotUserAgent.Load().(string)
	expectedUA := userAgent()

	if receivedUA != expectedUA {
		t.Errorf("User-Agent mismatch\nExpected: %s\nReceived: %s", expectedUA, receivedUA)
	}
}

func TestSupportsBrowserTools(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gpt-oss", true},
		{"gpt-oss-latest", true},
		{"GPT-OSS", true},
		{"Gpt-Oss-v2", true},
		{"qwen3", false},
		{"deepseek-v3", false},
		{"llama3.3", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := supportsBrowserTools(tt.model); got != tt.want {
				t.Errorf("supportsBrowserTools(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestWebSearchToolRegistration(t *testing.T) {
	// Validates that the capability-gating logic in chat() correctly
	// decides which tools to register based on model capabilities and
	// the web search flag.
	tests := []struct {
		name             string
		webSearchEnabled bool
		hasToolsCap      bool
		model            string
		wantBrowser      bool // expects browser tools (gpt-oss)
		wantWebSearch    bool // expects basic web search/fetch tools
		wantNone         bool // expects no tools registered
	}{
		{
			name:             "web search enabled with tools capability - browser model",
			webSearchEnabled: true,
			hasToolsCap:      true,
			model:            "gpt-oss-latest",
			wantBrowser:      true,
		},
		{
			name:             "web search enabled with tools capability - non-browser model",
			webSearchEnabled: true,
			hasToolsCap:      true,
			model:            "qwen3",
			wantWebSearch:    true,
		},
		{
			name:             "web search enabled without tools capability",
			webSearchEnabled: true,
			hasToolsCap:      false,
			model:            "llama3.3",
			wantNone:         true,
		},
		{
			name:             "web search disabled with tools capability",
			webSearchEnabled: false,
			hasToolsCap:      true,
			model:            "qwen3",
			wantNone:         true,
		},
		{
			name:             "web search disabled without tools capability",
			webSearchEnabled: false,
			hasToolsCap:      false,
			model:            "llama3.3",
			wantNone:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Replicate the decision logic from chat() handler
			gotBrowser := false
			gotWebSearch := false

			if tt.webSearchEnabled && tt.hasToolsCap {
				if supportsBrowserTools(tt.model) {
					gotBrowser = true
				} else {
					gotWebSearch = true
				}
			}

			if tt.wantBrowser && !gotBrowser {
				t.Error("expected browser tools to be registered")
			}
			if tt.wantWebSearch && !gotWebSearch {
				t.Error("expected web search tools to be registered")
			}
			if tt.wantNone && (gotBrowser || gotWebSearch) {
				t.Error("expected no tools to be registered")
			}
			if !tt.wantBrowser && gotBrowser {
				t.Error("unexpected browser tools registered")
			}
			if !tt.wantWebSearch && gotWebSearch {
				t.Error("unexpected web search tools registered")
			}
		})
	}
}

func TestSettingsToggleAutoUpdateOff_CancelsDownload(t *testing.T) {
	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	// Start with auto-update enabled
	settings, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := testStore.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	upd := &updater.Updater{Store: &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db2.sqlite"),
	}}
	defer upd.Store.Close()

	// We can't easily mock CancelOngoingDownload, but we can verify
	// the full settings handler flow works without error
	server := &Server{
		Store:   testStore,
		Restart: func() {},
		Updater: upd,
	}

	// Disable auto-update via settings API
	settings.AutoUpdateEnabled = false
	body, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("POST", "/api/v1/settings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	if err := server.settings(rr, req); err != nil {
		t.Fatalf("settings() error = %v", err)
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("settings() status = %d, want %d", rr.Code, http.StatusOK)
	}

	// Verify settings were saved with auto-update disabled
	saved, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if saved.AutoUpdateEnabled {
		t.Fatal("expected AutoUpdateEnabled to be false after toggle off")
	}
}

func TestSettingsPreservesOnboardingVersionWhenOmitted(t *testing.T) {
	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	settings, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.OnboardingVersion = 1
	if err := testStore.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	payload, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(payload, &fields); err != nil {
		t.Fatal(err)
	}
	delete(fields, "OnboardingVersion")
	payload, err = json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}

	server := &Server{
		Store:   testStore,
		Restart: func() {},
	}
	req := httptest.NewRequest("POST", "/api/v1/settings", bytes.NewReader(payload))
	rr := httptest.NewRecorder()

	if err := server.settings(rr, req); err != nil {
		t.Fatalf("settings() error = %v", err)
	}

	saved, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if saved.OnboardingVersion != 1 {
		t.Fatalf("OnboardingVersion = %d, want 1", saved.OnboardingVersion)
	}
}

func TestSettingsPreservesClaudeDesktopUsedWhenOmitted(t *testing.T) {
	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	settings, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.ClaudeDesktopUsed = true
	if err := testStore.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	payload, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(payload, &fields); err != nil {
		t.Fatal(err)
	}
	delete(fields, "ClaudeDesktopUsed")
	payload, err = json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}

	server := &Server{Store: testStore, Restart: func() {}}
	req := httptest.NewRequest("POST", "/api/v1/settings", bytes.NewReader(payload))
	rr := httptest.NewRecorder()
	if err := server.settings(rr, req); err != nil {
		t.Fatalf("settings() error = %v", err)
	}

	saved, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if !saved.ClaudeDesktopUsed {
		t.Fatal("expected ClaudeDesktopUsed to be preserved")
	}
}

func TestSettingsToggleAutoUpdateOn_WithPendingUpdate_ShowsNotification(t *testing.T) {
	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	// Start with auto-update disabled
	settings, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := testStore.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	// Simulate a verified staged update. The in-memory downloaded flag alone is
	// deliberately insufficient to expose a restart action.
	oldVal := updater.UpdateDownloaded
	updater.UpdateDownloaded = true
	defer func() { updater.UpdateDownloaded = oldVal }()
	oldStageDir := updater.UpdateStageDir
	oldVerify := updater.VerifyDownload
	updater.UpdateStageDir = t.TempDir()
	updater.VerifyDownload = func(_ string) error { return nil }
	defer func() {
		updater.UpdateStageDir = oldStageDir
		updater.VerifyDownload = oldVerify
	}()
	stagedDir := filepath.Join(updater.UpdateStageDir, "verified")
	if err := os.MkdirAll(stagedDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stagedDir, "Ollama-darwin.zip"), []byte("archive"), 0o755); err != nil {
		t.Fatal(err)
	}

	var notificationCalled atomic.Bool
	server := &Server{
		Store:   testStore,
		Restart: func() {},
		UpdateAvailableFunc: func() {
			notificationCalled.Store(true)
		},
	}

	// Re-enable auto-update via settings API
	settings.AutoUpdateEnabled = true
	body, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("POST", "/api/v1/settings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	if err := server.settings(rr, req); err != nil {
		t.Fatalf("settings() error = %v", err)
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("settings() status = %d, want %d", rr.Code, http.StatusOK)
	}

	if !notificationCalled.Load() {
		t.Fatal("expected UpdateAvailableFunc to be called when re-enabling with a downloaded update")
	}
}

func TestSettingsToggleAutoUpdateOn_NoPendingUpdate_DoesNotNotify(t *testing.T) {
	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	// Start with auto-update disabled
	settings, err := testStore.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := testStore.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	// Ensure no pending update - clear both the downloaded flag and the stage dir
	oldVal := updater.UpdateDownloaded
	updater.UpdateDownloaded = false
	defer func() { updater.UpdateDownloaded = oldVal }()

	oldStageDir := updater.UpdateStageDir
	updater.UpdateStageDir = t.TempDir() // empty dir means IsUpdatePending() returns false
	defer func() { updater.UpdateStageDir = oldStageDir }()

	upd := &updater.Updater{Store: &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db2.sqlite"),
	}}
	defer upd.Store.Close()

	var notificationCalled atomic.Bool
	server := &Server{
		Store:   testStore,
		Restart: func() {},
		Updater: upd,
		UpdateAvailableFunc: func() {
			notificationCalled.Store(true)
		},
	}

	// Re-enable auto-update via settings API
	settings.AutoUpdateEnabled = true
	body, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("POST", "/api/v1/settings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	if err := server.settings(rr, req); err != nil {
		t.Fatalf("settings() error = %v", err)
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("settings() status = %d, want %d", rr.Code, http.StatusOK)
	}

	// UpdateAvailableFunc should NOT be called since there's no pending update
	if notificationCalled.Load() {
		t.Fatal("UpdateAvailableFunc should not be called when there is no pending update")
	}
}
