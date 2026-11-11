package mlxrunner

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"

	"github.com/ollama/ollama/mlxrunner/model/qwen3asr"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

// Only the private loopback runner mux receives these routes. The public Ollama
// transcription API and renderer contracts remain unchanged.
func (r *Runner) installASRReplayRoutes(ctx context.Context, mux *http.ServeMux) (func(), error) {
	if _, ok := r.Model.(*qwen3asr.Model); !ok {
		return func() {}, nil
	}
	registry, err := newASRReplayRegistry(ctx, r.newASRReplaySession)
	if err != nil {
		return nil, err
	}
	registerASRReplayRoutes(mux, registry)
	cleanup := func() { _ = registry.shutdown(context.Background()) }
	return cleanup, nil
}

func registerASRReplayRoutes(mux *http.ServeMux, registry *asrReplayRegistry) {
	mux.HandleFunc("POST /v1/asr/replay/reserve", func(w http.ResponseWriter, r *http.Request) {
		id, until, err := registry.reserve()
		if err != nil {
			writeASRReplayError(w, err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"id": id, "begin_deadline_ms": until.UnixMilli()})
	})
	type command struct {
		ID     string        `json:"id"`
		Config stream.Config `json:"config"`
		Offset int           `json:"offset"`
		PCM    []float32     `json:"pcm"`
	}
	for _, action := range []string{"begin", "push", "seed", "finish", "close"} {
		mux.HandleFunc("POST /v1/asr/replay/"+action, func(w http.ResponseWriter, r *http.Request) {
			limit := int64(256 << 10)
			if action == "seed" {
				limit = 4 << 20
			}
			r.Body = http.MaxBytesReader(w, r.Body, limit)
			decoder := json.NewDecoder(r.Body)
			decoder.DisallowUnknownFields()
			var cmd command
			if err := decoder.Decode(&cmd); err != nil {
				writeASRReplayError(w, err)
				return
			}
			if err := decoder.Decode(new(any)); err != io.EOF {
				writeASRReplayError(w, errors.New("expected one replay command"))
				return
			}
			if _, err := hex.DecodeString(cmd.ID); len(cmd.ID) != 48 || err != nil {
				writeASRReplayError(w, fmt.Errorf("invalid replay reservation ID"))
				return
			}
			var result any = map[string]bool{"ok": true}
			var err error
			switch action {
			case "begin":
				err = registry.begin(cmd.ID, cmd.Config)
			case "push", "seed", "finish":
				stop := context.AfterFunc(r.Context(), func() { registry.cancel(cmd.ID) })
				defer stop()
				if action == "push" {
					result, err = registry.push(cmd.ID, cmd.Offset, cmd.PCM)
				} else if action == "seed" {
					result, err = registry.seed(cmd.ID, cmd.PCM)
				} else {
					result, err = registry.finish(cmd.ID)
				}
			case "close":
				err = registry.closeAndWait(r.Context(), cmd.ID)
				result = map[string]bool{"drained": err == nil}
			}
			if err != nil {
				writeASRReplayError(w, err)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(result)
		})
	}
}

func writeASRReplayError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrASRGenerationLimit) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error(), "code": asrGenerationLimitCode})
		return
	}
	status := http.StatusBadRequest
	var oversized *http.MaxBytesError
	var backend *api.StatusError
	switch {
	case errors.As(err, &oversized):
		status = http.StatusRequestEntityTooLarge
	case errors.As(err, &backend):
		status = backend.StatusCode
	case errors.Is(err, errReplayCapacity), errors.Is(err, stream.ErrBusy):
		status = http.StatusTooManyRequests
	case errors.Is(err, errReplayUnknown):
		status = http.StatusNotFound
	case errors.Is(err, stream.ErrClosed):
		status = http.StatusGone
	case errors.Is(err, errReplayConflict), errors.Is(err, errReplayNotStarted), errors.Is(err, errReplayInstance):
		status = http.StatusConflict
	case errors.Is(err, errReplayShutdown):
		status = http.StatusServiceUnavailable
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		status = http.StatusRequestTimeout
	}
	http.Error(w, err.Error(), status)
}
