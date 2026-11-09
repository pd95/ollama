package chatterbox

import (
	"strings"

	"golang.org/x/text/unicode/norm"
)

func normalizedV3Text(text, language string) string {
	text = strings.Join(strings.Fields(text), " ")
	for _, pair := range [][2]string{{"...", ", "}, {"…", ", "}, {":", ","}, {" - ", ", "}, {";", ", "}, {"—", "-"}, {"–", "-"}, {" ,", ","}} {
		text = strings.ReplaceAll(text, pair[0], pair[1])
	}
	text = strings.TrimRight(text, " ")
	if !strings.HasSuffix(text, ".") && !strings.HasSuffix(text, "!") && !strings.HasSuffix(text, "?") && !strings.HasSuffix(text, "-") && !strings.HasSuffix(text, ",") && !strings.HasSuffix(text, "、") && !strings.HasSuffix(text, "，") && !strings.HasSuffix(text, "。") && !strings.HasSuffix(text, "？") && !strings.HasSuffix(text, "！") {
		text += "."
	}
	return "[" + language + "]" + strings.ReplaceAll(norm.NFKD.String(text), " ", "[SPACE]")
}
