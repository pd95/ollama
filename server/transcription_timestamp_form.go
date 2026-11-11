package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
	"golang.org/x/text/language"
)

const maxTimestampUploadBytes = maxAudioUploadBytes

func timestampUploadTooLarge() error {
	return &api.StatusError{StatusCode: http.StatusRequestEntityTooLarge, ErrorMessage: "word timestamps accept WAV or MP3 uploads up to 512 MiB"}
}

func (f *audioTextForm) parseTimestampOptions(r *http.Request, task renderers.AudioTask) error {
	f.alignmentModel = strings.TrimSpace(r.FormValue("alignment_model"))
	values := r.MultipartForm.Value["timestamp_granularities[]"]
	if len(values) > 0 {
		if task != renderers.AudioTranscribe {
			return errors.New("timestamp granularities are only supported for transcriptions")
		}
		if f.responseFormat != "verbose_json" {
			return errors.New("timestamp granularities require response_format=verbose_json")
		}
	}
	for _, value := range values {
		switch value {
		case "word":
			f.wordTimestamps = true
		case "segment":
			f.segmentTimestamps = true
		default:
			return fmt.Errorf("unsupported timestamp granularity %q, use word or segment", value)
		}
	}
	if f.stream && f.segmentTimestamps && !f.wordTimestamps {
		return errors.New("streaming segment timestamps require word timestamps")
	}
	if f.alignmentModel != "" && !f.wordTimestamps {
		return errors.New("alignment_model requires word timestamps")
	}
	name := renderers.LanguageName(f.request.Source)
	if f.wordTimestamps && f.request.Source != language.Und && name != "English" && name != "German" {
		return errors.New("word timestamps currently support English, German or automatic language detection")
	}
	return nil
}

func spoolTimestampUpload(ctx context.Context, source io.Reader) (path string, err error) {
	file, err := os.CreateTemp("", "ollama-word-timestamps-*")
	if err != nil {
		return "", err
	}
	defer func() {
		closeErr := file.Close()
		if err == nil {
			err = closeErr
		}
		if err != nil {
			os.Remove(file.Name())
		}
	}()
	reader := &audioUploadReader{Reader: io.LimitReader(source, maxTimestampUploadBytes+1), check: ctx.Err}
	n, err := io.Copy(file, reader)
	if err != nil {
		return "", err
	}
	if n > maxTimestampUploadBytes {
		return "", timestampUploadTooLarge()
	}
	if n == 0 {
		return "", errors.New("audio file is empty")
	}
	return file.Name(), nil
}

func (f audioTextForm) removeTimestampUpload() {
	if f.timestampUpload != "" {
		_ = os.Remove(f.timestampUpload)
	}
}
