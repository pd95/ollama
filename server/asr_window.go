package server

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	asrstream "github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

const asrWindowPath = "/api/experimental/audio/windows"

// Kept separate from ordinary replay backends: only fresh-window requests need
// this optional native operation. Admission and leases still belong to m.create.
type asrWindowBackend interface {
	SeedASR(context.Context, string, []float32) (*asrstream.Update, error)
}

type asrWindowRequest struct {
	model, language, prompt string
	start                   int64
	pcm                     []float32
}

func registerASRWindowRoutes(r *gin.Engine, m *asrSessionManager) {
	r.POST(asrWindowPath, func(c *gin.Context) {
		controller := http.NewResponseController(c.Writer)
		_ = controller.SetReadDeadline(time.Now().Add(m.policy.idle))
		request, err := readASRWindow(c)
		_ = controller.SetReadDeadline(time.Time{})
		if err != nil {
			writeASRInputError(c, err)
			return
		}
		p, err := m.create(c.Request.Context(), request.model, asrstream.Config{Language: request.language, Context: request.prompt})
		if err != nil {
			writeASRSessionError(c, err)
			return
		}
		stop := context.AfterFunc(c.Request.Context(), p.startClose)
		defer stop()
		update, workErr := p.window(c.Request.Context(), request.pcm)
		// Success is published only after the same positive cleanup barrier as
		// sessions. Cancellation may return sooner, but cleanup retains the lease.
		closeErr := m.closeAndWait(c.Request.Context(), p.id)
		if closeErr != nil {
			writeASRSessionError(c, closeErr)
			return
		}
		if workErr != nil {
			writeASRSessionError(c, workErr)
			return
		}
		c.JSON(http.StatusOK, gin.H{"text": update.Text, "language": update.Language, "source_start_sample": request.start, "source_end_sample": request.start + int64(len(request.pcm)), "audio_samples": len(request.pcm), "sample_rate": asrstream.SampleRate, "fresh": true})
	})
}

func readASRWindow(c *gin.Context) (asrWindowRequest, error) {
	var request asrWindowRequest
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 600<<10)
	reader, err := c.Request.MultipartReader()
	if err != nil {
		return request, err
	}
	seen := map[string]bool{}
	var start string
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return request, err
		}
		name := part.FormName()
		if seen[name] {
			return request, asrInputError("duplicate audio window field")
		}
		seen[name] = true
		limit := int64(4096)
		switch name {
		case "model", "language", "prompt", "start_sample":
			if part.FileName() != "" {
				return request, asrInputError("audio window metadata must be text fields")
			}
		case "file":
			limit = 2 * asrstream.MaxSeedSamples
		default:
			return request, asrInputError("unknown audio window field")
		}
		data, err := io.ReadAll(io.LimitReader(part, limit+1))
		if err != nil {
			return request, err
		}
		if int64(len(data)) > limit {
			if name != "file" {
				return request, asrInputError("audio window metadata must fit within 4096 bytes")
			}
			return request, &http.MaxBytesError{Limit: limit}
		}
		switch name {
		case "model":
			request.model = string(data)
		case "language":
			language, ok := map[string]string{"": "", "auto": "", "en": "English", "de": "German"}[string(data)]
			if !ok {
				return request, asrInputError("language must be en, de or auto")
			}
			request.language = language
		case "prompt":
			request.prompt = string(data)
		case "start_sample":
			start = string(data)
		case "file":
			if len(data)%2 != 0 || len(data)/2 < 201 {
				return request, asrInputError("audio window requires 201 to 256000 whole PCM16 samples")
			}
			request.pcm = make([]float32, len(data)/2)
			for i := range request.pcm {
				request.pcm[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:]))) / 32768
			}
		}
	}
	if request.model == "" || !seen["file"] || len(request.prompt) > 4096 || !utf8.ValidString(request.prompt) {
		return request, asrInputError("audio window requires model, PCM file and valid UTF-8 prompt within 4096 bytes")
	}
	if seen["start_sample"] {
		value, err := strconv.ParseInt(start, 10, 64)
		if err != nil || value < 0 || strconv.FormatInt(value, 10) != start || value > math.MaxInt64-int64(len(request.pcm)) {
			return request, asrInputError("invalid audio window start_sample")
		}
		request.start = value
	}
	return request, nil
}

func (p *asrPublicSession) window(ctx context.Context, pcm []float32) (asrstream.Update, error) {
	p.mu.Lock()
	if p.state != "active" || p.busy || p.accepted != 0 {
		p.mu.Unlock()
		return asrstream.Update{}, errASRSessionClosed
	}
	p.busy = true
	p.idle.Stop()
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.busy = false
		p.mu.Unlock()
		p.startClose()
	}()
	backend, ok := p.backend.(asrWindowBackend)
	if !ok {
		return asrstream.Update{}, errors.New("native runner does not support fresh audio windows")
	}
	u, err := backend.SeedASR(ctx, p.privateID, pcm)
	if err != nil {
		return asrstream.Update{}, err
	}
	if u == nil || u.Final || u.Revision != 1 || u.AudioEnd != len(pcm) || !utf8.ValidString(u.Text) || len(u.Text) > 32768 {
		return asrstream.Update{}, errors.New("invalid ASR window seed acknowledgment")
	}
	final, err := p.backend.FinishASR(ctx, p.privateID)
	if err != nil {
		return asrstream.Update{}, err
	}
	if err := ctx.Err(); err != nil {
		return asrstream.Update{}, err
	}
	if !final.Final || final.Revision != 2 || final.AudioEnd != len(pcm) || final.Text != u.Text || final.Language != u.Language {
		return asrstream.Update{}, errors.New("invalid ASR window final acknowledgment")
	}
	p.mu.Lock()
	active := p.state == "active"
	p.mu.Unlock()
	if !active {
		return asrstream.Update{}, errASRSessionClosed
	}
	return final, nil
}
