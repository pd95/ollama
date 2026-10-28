//go:build windows || darwin

package main

import (
	"slices"
	"testing"
)

func TestAttachmentFileExtensions(t *testing.T) {
	for _, images := range []bool{false, true} {
		for _, audio := range []bool{false, true} {
			exts := attachmentFileExtensions(images, audio)
			for _, ext := range []string{"png", "jpg", "jpeg", "webp"} {
				if slices.Contains(exts, ext) != images {
					t.Errorf("images=%t audio=%t: picker permission for %s", images, audio, ext)
				}
			}
			for _, ext := range []string{"mp3", "wav"} {
				if slices.Contains(exts, ext) != audio {
					t.Errorf("images=%t audio=%t: picker permission for %s", images, audio, ext)
				}
			}
			for _, ext := range []string{"txt", "pdf", "docx", "md", "go"} {
				if !slices.Contains(exts, ext) {
					t.Errorf("text document %s must remain selectable", ext)
				}
			}
		}
	}
}
