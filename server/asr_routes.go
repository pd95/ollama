package server

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

const asrSessionPath = "/api/experimental/audio/sessions"

func (s *Server) registerASRSessionRoutes(r *gin.Engine) {
	if !experimentEnabled("asr-streaming") {
		return
	}
	s.asrSessions = newASRSessionManager(context.Background(), s.acquireASRBackend)
	registerASRSessionRoutes(r, s.asrSessions)
}

func (s *Server) closeASRSessions() {
	if s.asrSessions != nil {
		s.asrSessions.stopParent()
		s.asrSessions.closeAll()
	}
}

func registerASRSessionRoutes(r *gin.Engine, m *asrSessionManager) {
	registerASRWindowRoutes(r, m)
	r.POST(asrSessionPath, func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 32<<10)
		var request struct {
			Model            string          `json:"model"`
			Language         string          `json:"language"`
			Prompt           string          `json:"prompt"`
			UpdateIntervalMS json.RawMessage `json:"update_interval_ms"`
		}
		decoder := json.NewDecoder(c.Request.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil {
			writeASRInputError(c, err)
			return
		}
		if err := decoder.Decode(new(any)); err != io.EOF {
			writeASRSessionError(c, asrInputError("expected one audio session request"))
			return
		}
		if request.Model == "" {
			writeASRSessionError(c, asrInputError("model is required"))
			return
		}
		language, ok := map[string]string{"": "", "auto": "", "en": "English", "de": "German"}[request.Language]
		if !ok {
			writeASRSessionError(c, asrInputError("language must be en, de or auto"))
			return
		}
		updateSamples := asrstream.UpdateSamples
		if len(request.UpdateIntervalMS) != 0 {
			var interval *int
			if err := json.Unmarshal(request.UpdateIntervalMS, &interval); err != nil || interval == nil || (*interval != 500 && *interval != 1000 && *interval != 2000) {
				writeASRSessionError(c, asrInputError("update_interval_ms must be 500, 1000 or 2000"))
				return
			}
			updateSamples = *interval * asrstream.SampleRate / 1000
		}
		p, err := m.create(c.Request.Context(), request.Model, asrstream.Config{Language: language, Context: request.Prompt, UpdateSamples: updateSamples})
		if err != nil {
			writeASRSessionError(c, err)
			return
		}
		stop := context.AfterFunc(c.Request.Context(), p.startClose)
		defer stop()
		writeASRSessionResponse(c, http.StatusCreated, p, map[string]any{"id": p.id, "audio_format": "pcm_s16le", "sample_rate": asrstream.SampleRate, "channels": 1, "max_packet_samples": p.packetSamples, "max_audio_samples": asrstream.MaxSamples, "update_samples": p.updateSamples, "update_interval_ms": p.updateSamples * 1000 / asrstream.SampleRate, "idle_timeout_ms": m.policy.idle.Milliseconds(), "session_timeout_ms": m.policy.absolute.Milliseconds()})
	})
	for _, action := range []string{"audio", "finish"} {
		r.POST(asrSessionPath+"/:id/"+action, func(c *gin.Context) {
			p, err := lookupASRSession(c, m)
			if err != nil {
				writeASRSessionError(c, err)
				return
			}
			if !p.transport.TryLock() {
				writeASRSessionError(c, errASRSessionBusy)
				return
			}
			defer p.transport.Unlock()
			stop := context.AfterFunc(c.Request.Context(), p.startClose)
			defer stop()
			var pcm []float32
			var offset int
			if action == "audio" {
				contentType, _, err := mime.ParseMediaType(c.GetHeader("Content-Type"))
				if err != nil || contentType != "application/octet-stream" {
					writeASRSessionError(c, asrInputError("audio packets require application/octet-stream PCM16"))
					return
				}
				rawOffset := c.Query("offset_samples")
				value, err := strconv.ParseUint(rawOffset, 10, 32)
				if err != nil || strconv.FormatUint(value, 10) != rawOffset || value > asrstream.MaxSamples {
					writeASRSessionError(c, asrInputError("invalid offset_samples"))
					return
				}
				offset = int(value)
				controller := http.NewResponseController(c.Writer)
				_ = controller.SetReadDeadline(time.Now().Add(m.policy.idle))
				defer func() { _ = controller.SetReadDeadline(time.Time{}) }()
				c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, int64(2*p.packetSamples))
				data, err := io.ReadAll(c.Request.Body)
				_ = controller.SetReadDeadline(time.Time{}) // Upload deadline must not cover inference.
				if err != nil {
					writeASRInputError(c, err)
					return
				}
				if len(data) == 0 || len(data)%2 != 0 {
					writeASRSessionError(c, asrInputError("audio packet must contain whole PCM16 samples"))
					return
				}
				pcm = make([]float32, len(data)/2)
				for i := range pcm {
					pcm[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:]))) / 32768
				}
			}
			accepted, u, err := p.operation(c.Request.Context(), offset, pcm, action == "finish")
			if err != nil {
				writeASRSessionError(c, err)
				return
			}
			if action == "finish" {
				if err := m.closeAndWait(c.Request.Context(), p.id); err != nil {
					writeASRSessionError(c, err)
					return
				}
			}
			response := map[string]any{"accepted_samples": accepted}
			if u != nil {
				response["audio_end_sample"] = u.AudioEnd
				response["revision"] = u.Revision
				response["text"] = u.Text
				response["stable_text"] = u.StableText
				response["draft_text"] = u.DraftText
				response["language"] = u.Language
				response["final"] = u.Final
			}
			writeASRSessionResponse(c, http.StatusOK, p, response)
		})
	}
	r.DELETE(asrSessionPath+"/:id", func(c *gin.Context) {
		if !validASRPublicID(c.Param("id")) {
			writeASRSessionError(c, errASRSessionUnknown)
			return
		}
		if err := m.closeAndWait(c.Request.Context(), c.Param("id")); err != nil {
			writeASRSessionError(c, err)
			return
		}
		c.Status(http.StatusNoContent)
	})
}

func validASRPublicID(id string) bool {
	_, err := hex.DecodeString(id)
	return len(id) == 32 && err == nil && id == strings.ToLower(id)
}

func lookupASRSession(c *gin.Context, m *asrSessionManager) (*asrPublicSession, error) {
	id := c.Param("id")
	if !validASRPublicID(id) {
		return nil, errASRSessionUnknown
	}
	m.mu.Lock()
	p := m.active[id]
	m.mu.Unlock()
	if p == nil {
		return nil, errASRSessionUnknown
	}
	return p, nil
}

func writeASRSessionResponse(c *gin.Context, status int, p *asrPublicSession, response any) {
	c.Header("Content-Type", "application/json")
	c.Status(status)
	if err := json.NewEncoder(c.Writer).Encode(response); err != nil {
		p.startClose()
	}
}

func writeASRSessionError(c *gin.Context, err error) {
	if errors.Is(err, mlxrunner.ErrASRGenerationLimit) {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": err.Error(), "code": "asr_generation_limit"})
		return
	}
	status := http.StatusInternalServerError
	var input asrInputError
	var oversized *http.MaxBytesError
	var backend *api.StatusError
	switch {
	case errors.As(err, &input):
		status = http.StatusBadRequest
	case errors.Is(err, errASRSessionUnknown), errors.Is(err, errASRSessionClosed):
		status = http.StatusNotFound
	case errors.Is(err, errASRSessionBusy), errors.Is(err, errASRSessionCapacity):
		status = http.StatusTooManyRequests
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		status = http.StatusRequestTimeout
	case errors.As(err, &oversized):
		status = http.StatusRequestEntityTooLarge
	case errors.As(err, &backend):
		status = backend.StatusCode
	}
	c.JSON(status, gin.H{"error": err.Error()})
}

func writeASRInputError(c *gin.Context, err error) {
	var oversized *http.MaxBytesError
	if errors.As(err, &oversized) {
		writeASRSessionError(c, err)
		return
	}
	writeASRSessionError(c, asrInputError(err.Error()))
}
