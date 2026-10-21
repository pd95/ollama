package ui

import "github.com/ollama/ollama/api"

func chatThinkValue(think any) *api.ThinkValue {
	switch value := think.(type) {
	case bool:
		// Preserve false so models whose default is thinking can be
		// explicitly switched off when tools are enabled.
		return &api.ThinkValue{Value: value}
	case string:
		if value != "" && value != "none" {
			return &api.ThinkValue{Value: value}
		}
	}

	return nil
}
