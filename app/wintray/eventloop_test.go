//go:build windows

package wintray

import "testing"

type lifecycleApp struct {
	running    bool
	onboarding bool
	runPath    string
	showCall   bool
}

func (a *lifecycleApp) UIRun(path string)  { a.runPath = path }
func (a *lifecycleApp) UIShow()            { a.showCall = true }
func (a *lifecycleApp) UITerminate()       {}
func (a *lifecycleApp) UIRunning() bool    { return a.running }
func (a *lifecycleApp) UIOnboarding() bool { return a.onboarding }
func (a *lifecycleApp) Quit()              {}
func (a *lifecycleApp) DoUpdate()          {}

func TestFocusUICreatesOrShowsWindow(t *testing.T) {
	tests := []struct {
		name       string
		running    bool
		onboarding bool
		wantRun    string
		wantShow   bool
	}{
		{name: "creates selected home when tray only", wantRun: "/"},
		{name: "routes existing window to selected home", running: true, wantRun: "/"},
		{name: "preserves onboarding", running: true, onboarding: true, wantShow: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			app := &lifecycleApp{running: tt.running, onboarding: tt.onboarding}
			focusUI(app)
			if app.runPath != tt.wantRun {
				t.Errorf("run path = %q, want %q", app.runPath, tt.wantRun)
			}
			if app.showCall != tt.wantShow {
				t.Errorf("show called = %v, want %v", app.showCall, tt.wantShow)
			}
		})
	}
}
