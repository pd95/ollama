// Package chatterbox implements Chatterbox speech networks in native MLX.
package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type weights struct {
	tensors map[string]*mlx.Array
	factory model.LinearFactory
	err     error
}

func (w *weights) get(name string) *mlx.Array {
	a := w.tensors[name]
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Chatterbox missing tensor %s", name)
	}
	return a
}

func (w *weights) linear(name string) nn.LinearLayer {
	a := w.factory.Make(name)
	if a == nil && w.err == nil {
		w.err = fmt.Errorf("Chatterbox missing linear %s", name)
	}
	return a
}

func (w *weights) norm(name string, eps float32) *nn.LayerNorm {
	return &nn.LayerNorm{Weight: w.get(name + ".weight"), Bias: w.get(name + ".bias"), Eps: eps}
}

type conv struct {
	weight, bias                      *mlx.Array
	stride, padding, dilation, groups int32
}

func (w *weights) conv(name string, stride, padding, dilation, groups int32) conv {
	return conv{weight: w.get(name + ".weight"), bias: w.tensors[name+".bias"], stride: stride, padding: padding, dilation: dilation, groups: groups}
}

func (c conv) forward(x *mlx.Array) *mlx.Array {
	return mlx.Conv1d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)
}

func exactGELU(x *mlx.Array) *mlx.Array {
	return mlx.Div(mlx.Mul(x, mlx.Add(mlx.FromValue(float32(1)).AsType(x.DType()), mlx.Erf(mlx.Div(x, mlx.FromValue(float32(math.Sqrt2)).AsType(x.DType()))))), mlx.FromValue(float32(2)).AsType(x.DType()))
}
