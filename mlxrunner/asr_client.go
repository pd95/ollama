package mlxrunner

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/qwen3asr/stream"
)

// ASRProcessDone observes this exact subprocess, not a health/status snapshot.
// This optional ASR extension deliberately does not expand llm.LlamaServer.
func (c *Client) ASRProcessDone() <-chan struct{} { return c.done }

func (c *Client) ReserveASR(ctx context.Context) (string, time.Time, error) {
	var out struct {
		ID       string `json:"id"`
		Deadline int64  `json:"begin_deadline_ms"`
	}
	if err := c.asrCommand(ctx, "reserve", struct{}{}, &out); err != nil {
		return "", time.Time{}, err
	}
	if _, err := hex.DecodeString(out.ID); err != nil || len(out.ID) != 48 || strings.ToLower(out.ID) != out.ID || out.Deadline <= 0 {
		return "", time.Time{}, errors.New("invalid ASR reservation acknowledgment")
	}
	return out.ID, time.UnixMilli(out.Deadline), nil
}

func (c *Client) BeginASR(ctx context.Context, id string, config stream.Config) error {
	var out struct {
		OK bool `json:"ok"`
	}
	if err := c.asrCommand(ctx, "begin", struct {
		ID     string        `json:"id"`
		Config stream.Config `json:"config"`
	}{id, config}, &out); err != nil {
		return err
	}
	if !out.OK {
		return errors.New("ASR begin was not acknowledged")
	}
	return nil
}

func (c *Client) PushASR(ctx context.Context, id string, offset int, pcm []float32) (*stream.Update, error) {
	var out *stream.Update
	err := c.asrCommand(ctx, "push", struct {
		ID     string    `json:"id"`
		Offset int       `json:"offset"`
		PCM    []float32 `json:"pcm"`
	}{id, offset, pcm}, &out)
	return out, err
}

func (c *Client) FinishASR(ctx context.Context, id string) (stream.Update, error) {
	var out *stream.Update
	if err := c.asrCommand(ctx, "finish", map[string]string{"id": id}, &out); err != nil {
		return stream.Update{}, err
	}
	if out == nil || !out.Final {
		return stream.Update{}, errors.New("invalid ASR final acknowledgment")
	}
	return *out, nil
}

// SeedASR decodes one bounded fresh window without incremental backfill.
func (c *Client) SeedASR(ctx context.Context, id string, pcm []float32) (*stream.Update, error) {
	var out *stream.Update
	err := c.asrCommand(ctx, "seed", struct {
		ID  string    `json:"id"`
		PCM []float32 `json:"pcm"`
	}{id, pcm}, &out)
	return out, err
}

func (c *Client) CloseASR(ctx context.Context, id string) error {
	var out struct {
		Drained bool `json:"drained"`
	}
	if err := c.asrCommand(ctx, "close", map[string]string{"id": id}, &out); err != nil {
		return err
	}
	if !out.Drained {
		return errors.New("ASR cleanup was not acknowledged")
	}
	return nil
}

func (c *Client) asrCommand(ctx context.Context, action string, body, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/asr/replay/%s", c.port, action), bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	const maxResponse = 1 << 20
	data, err = io.ReadAll(io.LimitReader(resp.Body, maxResponse+1))
	if err != nil {
		return err
	}
	if len(data) > maxResponse {
		return errors.New("ASR response exceeds limit")
	}
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusUnprocessableEntity {
			if message, ok := asrGenerationLimitMessage(data); ok {
				return fmt.Errorf("%s: %w", message, ErrASRGenerationLimit)
			}
		}
		return &api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: strings.TrimSpace(string(data))}
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return fmt.Errorf("invalid ASR acknowledgment: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return errors.New("expected one ASR acknowledgment")
	}
	return nil
}

// Parse the error envelope token by token so duplicate keys and case aliases
// cannot turn a contradictory response into a recoverable generation limit.
func asrGenerationLimitMessage(data []byte) (string, bool) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return "", false
	}
	var message, code string
	var sawError, sawCode bool
	for decoder.More() {
		token, err = decoder.Token()
		if err != nil {
			return "", false
		}
		key, ok := token.(string)
		if !ok {
			return "", false
		}
		token, err = decoder.Token()
		if err != nil {
			return "", false
		}
		value, ok := token.(string)
		if !ok {
			return "", false
		}
		switch key {
		case "error":
			if sawError {
				return "", false
			}
			message, sawError = value, true
		case "code":
			if sawCode {
				return "", false
			}
			code, sawCode = value, true
		default:
			return "", false
		}
	}
	token, err = decoder.Token()
	if err != nil || token != json.Delim('}') || !sawError || !sawCode || strings.TrimSpace(message) == "" || code != asrGenerationLimitCode || decoder.Decode(new(any)) != io.EOF {
		return "", false
	}
	return message, true
}
