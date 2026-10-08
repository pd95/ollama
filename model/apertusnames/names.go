// Package apertusnames maps API tool names to identifiers in Apertus tool prompts.
package apertusnames

import (
	"fmt"
	"strings"
)

// Encode preserves ordinary identifiers and encodes dots and dollar signs so
// namespaced API names remain distinct from flat names in Apertus declarations.
func Encode(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("invalid apertus tool name %q", name)
	}
	var out strings.Builder
	segmentStart := true
	for i := range len(name) {
		c := name[i]
		switch {
		case c == '.':
			if segmentStart {
				return "", fmt.Errorf("invalid apertus tool name %q", name)
			}
			out.WriteString("$2E")
			segmentStart = true
		case c == '$':
			out.WriteString("$24")
			segmentStart = false
		case c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || !segmentStart && c >= '0' && c <= '9':
			out.WriteByte(c)
			segmentStart = false
		default:
			return "", fmt.Errorf("invalid apertus tool name %q", name)
		}
	}
	if segmentStart {
		return "", fmt.Errorf("invalid apertus tool name %q", name)
	}
	return out.String(), nil
}
