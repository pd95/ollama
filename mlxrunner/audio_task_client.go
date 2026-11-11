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

	"github.com/ollama/ollama/api"
)

// AudioTaskProcessDone is the exact owned subprocess exit, never cached health.
func (c *Client) AudioTaskProcessDone() <-chan struct{} { return c.done }

func (c *Client) ReserveAudioTask(ctx context.Context) (string, error) {
	var out struct {
		ID       string `json:"id"`
		Deadline int64  `json:"begin_deadline_ms"`
	}
	if err := c.audioTaskCommand(ctx, "reserve", struct{}{}, &out); err != nil {
		return "", err
	}
	if _, err := hex.DecodeString(out.ID); err != nil || len(out.ID) != 48 || out.ID != strings.ToLower(out.ID) || out.Deadline <= 0 {
		return "", errors.New("invalid audio task reservation")
	}
	return out.ID, nil
}

func (c *Client) RunAudioTask(ctx context.Context, input AudioTaskRequest) (AudioTaskResult, error) {
	var out audioTaskReply
	if err := c.audioTaskCommand(ctx, "run", input, &out); err != nil {
		return AudioTaskResult{}, err
	}
	if out.ID != input.ID || !out.Drained || out.Result == nil {
		return AudioTaskResult{}, errors.New("invalid audio task completion acknowledgment")
	}
	return *out.Result, nil
}

func (c *Client) CloseAudioTask(ctx context.Context, id string) error {
	var out audioTaskReply
	if err := c.audioTaskCommand(ctx, "close", AudioTaskRequest{ID: id}, &out); err != nil {
		return err
	}
	if out.ID != id || !out.Drained {
		return errors.New("audio task cleanup was not acknowledged")
	}
	return nil
}

func (c *Client) audioTaskCommand(ctx context.Context, action string, input, output any) error {
	data, err := json.Marshal(input)
	if err != nil {
		return err
	}
	if len(data) > maxAudioTaskBody {
		return errors.New("native audio task request exceeds limit")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/audio/task/%s", c.port, action), bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err = io.ReadAll(io.LimitReader(resp.Body, (1<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 1<<20 {
		return errors.New("audio task response exceeds limit")
	}
	if resp.StatusCode != http.StatusOK {
		return &api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: strings.TrimSpace(string(data))}
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(output); err != nil {
		return fmt.Errorf("invalid audio task response: %w", err)
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return errors.New("expected one audio task response")
	}
	return nil
}
