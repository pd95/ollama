package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCustomApplyPatchHistoryPreservesPortableReasoning(t *testing.T) {
	for _, precedingCall := range []string{"", `{"type":"function_call","call_id":"read","name":"read_file","arguments":"{}"},`} {
		t.Run(precedingCall, func(t *testing.T) {
			body := []byte(`{"input":[` + precedingCall + `
				{"type":"reasoning","encrypted_content":"opaque ciphertext","summary":[{"type":"summary_text","text":"Plan the edit"}]},
				{"type":"custom_tool_call","call_id":"patch","name":"apply_patch","input":"*** Begin Patch\n*** End Patch\n"},
				{"type":"custom_tool_call_output","call_id":"patch","output":[{"type":"input_text","text":"Done"}]},
				{"role":"user","content":"continue"}
			]}`)
			var request ResponsesRequest
			if err := json.Unmarshal(body, &request); err != nil {
				t.Fatal(err)
			}
			chat, err := FromResponsesRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			if len(chat.Messages) != 3 || chat.Messages[0].Role != "assistant" || chat.Messages[0].Thinking != "Plan the edit" {
				t.Fatalf("portable reasoning/history = %#v", chat.Messages)
			}
			calls := chat.Messages[0].ToolCalls
			wantCalls := 1
			if precedingCall != "" {
				wantCalls = 2
			}
			if len(calls) != wantCalls || calls[len(calls)-1].ID != "patch" {
				t.Fatalf("reasoning must stay with its tool-call group: %#v", calls)
			}
			if chat.Messages[1].Role != "tool" || chat.Messages[1].ToolCallID != "patch" || chat.Messages[1].Content != "Done" || chat.Messages[2].Role != "user" {
				t.Fatalf("call/output continuation = %#v", chat.Messages)
			}
		})
	}
}

func TestResponsesMixedCustomHistoryPreservesApplyPatchIdentity(t *testing.T) {
	body := []byte(`{"model":"test","input":[
		{"type":"custom_tool_call","call_id":"generic","namespace":"editor","name":"apply_patch","input":"  雪\n"},
		{"type":"function_call","call_id":"ordinary","name":"read_file","arguments":"{}"},
		{"type":"custom_tool_call","call_id":"patch","name":"apply_patch","input":"*** Begin Patch\n*** End Patch\n"},
		{"type":"custom_tool_call_output","call_id":"generic","output":[{"type":"input_text","text":"read\n"},{"type":"input_text","text":"雪  "}]},
		{"type":"function_call_output","call_id":"ordinary","output":"ordinary"},
		{"type":"custom_tool_call_output","call_id":"patch","output":[{"type":"input_text","text":"Done\n"}]}
	]}`)
	var request ResponsesRequest
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	chat, err := FromResponsesRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 4 || len(chat.Messages[0].ToolCalls) != 3 {
		t.Fatalf("mixed history = %#v", chat.Messages)
	}
	calls := chat.Messages[0].ToolCalls
	seen := make(map[string]bool)
	for _, call := range calls {
		if seen[call.Function.Name] {
			t.Fatalf("history identities collide: %#v", calls)
		}
		seen[call.Function.Name] = true
	}
	if input, _ := calls[0].Function.Arguments.Get("input"); input != "  雪\n" {
		t.Fatalf("generic input changed: %#v", input)
	}
	if input, _ := calls[2].Function.Arguments.Get("input"); input != "*** Begin Patch\n*** End Patch\n" {
		t.Fatalf("patch input changed: %#v", input)
	}
	for i, want := range []string{"read\n雪  ", "ordinary", "Done\n"} {
		if chat.Messages[i+1].Content != want {
			t.Fatalf("output %d = %q, want %q", i, chat.Messages[i+1].Content, want)
		}
	}
	items := ResponsesFunctionCallOutputItems(request, "history_", calls)
	if items[0].Type != "function_call" || items[1].Type != "function_call" || items[2].Type != "custom_tool_call" {
		t.Fatalf("history broadened custom output identity: %#v", items)
	}
	ordinary := request.Input.Items[1].(ResponsesFunctionCall)
	ordinary.Name = "editor.apply_patch"
	request.Input.Items[1] = ordinary
	if _, err := FromResponsesRequest(request); err == nil || !strings.Contains(err.Error(), "collision") {
		t.Fatalf("ambiguous generic-history identity error = %v", err)
	}
	ordinary.Name = "read_file"
	request.Input.Items[1] = ordinary
	request.Tools = []ResponsesTool{{Type: "custom", Name: "read_note"}}
	if _, err := FromResponsesRequest(request); err == nil {
		t.Fatal("generic history must not authorize another custom declaration")
	}
}

func TestCompactionRoundTripsMixedCustomHistory(t *testing.T) {
	body := []byte(`{"model":"test","input":[
		{"type":"custom_tool_call","call_id":"generic","namespace":"workspace","name":"read_note","input":"  雪\n"},
		{"type":"custom_tool_call_output","call_id":"generic","output":[{"type":"input_text","text":"note\n"}]},
		{"type":"custom_tool_call","call_id":"patch","name":"apply_patch","input":"*** Begin Patch\n*** End Patch\n"},
		{"type":"custom_tool_call_output","call_id":"patch","output":[{"type":"input_text","text":"Done\n"}]},
		{"role":"user","content":"continue"}
	]}`)
	plan, err := PrepareStandaloneCompaction(body)
	if err != nil {
		t.Fatal(err)
	}
	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "summary", "retain_item_ids": []string{"item_000001", "item_000003"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if len(payload.CustomCallIDs) != 1 || !payload.CustomCallIDs["patch"] {
		t.Fatalf("custom identities = %#v", payload.CustomCallIDs)
	}
	items, err := payloadToResponsesItems(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 6 {
		t.Fatalf("retained pair items = %s", items)
	}
	for i, want := range []string{"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"} {
		if got := rawInputItemType(items[i+2]); got != want {
			t.Fatalf("retained item %d type = %q, want %q", i, got, want)
		}
	}
	if payload.Retained[1].Content != "note\n" || payload.Retained[3].Content != "Done\n" {
		t.Fatalf("retained outputs changed: %#v", payload.Retained)
	}
	continued, err := json.Marshal(map[string]any{"model": "test", "input": items})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareStandaloneCompaction(continued); err != nil {
		t.Fatalf("second compaction rejected mixed retained history: %v", err)
	}
}

func TestCompactionMixedHistoryStillRejectsMismatchedCustomPairs(t *testing.T) {
	for _, pair := range []string{
		`{"type":"custom_tool_call","call_id":"patch","name":"apply_patch","input":"*** Begin Patch\n*** End Patch\n"},{"type":"function_call_output","call_id":"patch","output":"Done"}`,
		`{"type":"function_call","call_id":"patch","name":"apply_patch","arguments":"{}"},{"type":"custom_tool_call_output","call_id":"patch","output":"Done"}`,
	} {
		body := []byte(`{"model":"test","input":[{"type":"custom_tool_call","call_id":"generic","name":"read_note","input":"note"},{"type":"custom_tool_call_output","call_id":"generic","output":"read"},` + pair + `]}`)
		if _, err := PrepareStandaloneCompaction(body); err == nil || !strings.Contains(err.Error(), "kinds differ") {
			t.Fatalf("mismatched custom pair error = %v", err)
		}
	}
}
