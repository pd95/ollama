package voxtraltts

import schema "github.com/ollama/ollama/mlxrunner/model/voxtraltts/config"

const Architecture = schema.Architecture

type (
	config     = schema.Config
	dimensions = schema.Dimensions
)

func parseConfig(raw []byte) (*config, error) { return schema.Parse(raw) }
