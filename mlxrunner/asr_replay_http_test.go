package mlxrunner

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen3ASRReplayHTTPGenerationLimit(t *testing.T) {
	err := fmt.Errorf("ASR replay: %w", ErrASRGenerationLimit)
	w := httptest.NewRecorder()
	writeASRReplayError(w, err)
	var body map[string]string
	if decodeErr := json.Unmarshal(w.Body.Bytes(), &body); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if w.Code != http.StatusUnprocessableEntity || w.Header().Get("Content-Type") != "application/json" || len(body) != 2 || body["error"] != err.Error() || body["code"] != "asr_generation_limit" {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
}

func TestQwen3ASRReplayHTTPUnclassifiedErrorsUnchanged(t *testing.T) {
	for _, test := range []struct {
		err    error
		status int
	}{
		{errors.New("invalid packet"), http.StatusBadRequest},
		{errors.New(ErrASRGenerationLimit.Error()), http.StatusBadRequest},
		{&api.StatusError{StatusCode: http.StatusUnprocessableEntity, ErrorMessage: "other backend failure"}, http.StatusUnprocessableEntity},
	} {
		w := httptest.NewRecorder()
		writeASRReplayError(w, test.err)
		if w.Code != test.status || strings.TrimSpace(w.Body.String()) != test.err.Error() || strings.Contains(w.Body.String(), `"code"`) {
			t.Fatalf("%v: %d %s", test.err, w.Code, w.Body)
		}
	}
}
