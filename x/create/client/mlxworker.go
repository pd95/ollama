package client

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var (
	createMLXWorkerOnce sync.Once
	createMLXWorker     *mlxthread.Thread
	createMLXWorkerErr  error
)

func mlxWorker() (*mlxthread.Thread, error) {
	createMLXWorkerOnce.Do(func() {
		createMLXWorker, createMLXWorkerErr = mlxthread.Start("mlx-create", func() error {
			if err := mlx.CheckInit(); err != nil {
				return fmt.Errorf("MLX not available: %w", err)
			}
			if mlx.GPUIsAvailable() {
				mlx.SetDefaultDeviceGPU()
				slog.Debug("create MLX worker initialized", "MLX version", mlx.Version(), "device", "gpu")
			} else {
				slog.Debug("create MLX worker initialized", "MLX version", mlx.Version(), "device", "cpu")
			}
			return nil
		})
	})
	return createMLXWorker, createMLXWorkerErr
}

func runOnMLXWorker(fn func() error) error {
	worker, err := mlxWorker()
	if err != nil {
		return err
	}
	return worker.Do(context.Background(), fn)
}
