package ui

import "testing"

func TestChatThinkValue(t *testing.T) {
	tests := []struct {
		name      string
		think     any
		wantValue any
		wantNil   bool
	}{
		{name: "unset", wantNil: true},
		{name: "enabled", think: true, wantValue: true},
		{name: "disabled", think: false, wantValue: false},
		{name: "level", think: "medium", wantValue: "medium"},
		{name: "none", think: "none", wantNil: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := chatThinkValue(tt.think)
			if tt.wantNil {
				if got != nil {
					t.Fatalf("chatThinkValue() = %#v, want nil", got)
				}
				return
			}

			if got == nil {
				t.Fatal("chatThinkValue() = nil, want explicit value")
			}
			if got.Value != tt.wantValue {
				t.Fatalf("chatThinkValue().Value = %#v, want %#v", got.Value, tt.wantValue)
			}
		})
	}
}
