package gemma4

import (
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

func TestValidateGemma4PackedLinearShape(t *testing.T) {
	tests := []struct {
		name      string
		weight    []int
		scales    []int
		want      []int
		groupSize int
		bits      int
		mode      string
		wantErr   bool
	}{
		{name: "nvfp4", weight: []int{3840, 80}, scales: []int{3840, 40}, want: []int{3840, 640}, groupSize: 16, bits: 4, mode: "nvfp4"},
		{name: "mxfp8", weight: []int{768, 192}, scales: []int{768, 24}, want: []int{768, 768}, groupSize: 32, bits: 8, mode: "mxfp8"},
		{name: "wrong output", weight: []int{767, 96}, scales: []int{768, 48}, want: []int{768, 768}, groupSize: 16, bits: 4, mode: "nvfp4", wantErr: true},
		{name: "wrong packed input", weight: []int{768, 95}, scales: []int{768, 48}, want: []int{768, 768}, groupSize: 16, bits: 4, mode: "nvfp4", wantErr: true},
		{name: "wrong scales", weight: []int{768, 96}, scales: []int{768, 47}, want: []int{768, 768}, groupSize: 16, bits: 4, mode: "nvfp4", wantErr: true},
		{name: "unsupported bits", weight: []int{768, 96}, scales: []int{768, 48}, want: []int{768, 768}, groupSize: 16, bits: 2, mode: "nvfp4", wantErr: true},
		{name: "missing mode", weight: []int{768, 96}, scales: []int{768, 48}, want: []int{768, 768}, groupSize: 16, bits: 4, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateGemma4PackedLinearShape(tt.weight, tt.scales, tt.want, tt.groupSize, tt.bits, tt.mode)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateGemma4PackedLinearShape() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGemma4QuantizedMediaLinearForward(t *testing.T) {
	for _, quantType := range []string{"int4", "nvfp4", "mxfp8"} {
		t.Run(quantType, func(t *testing.T) {
			useMLXTestThread(t)
			groupSize, bits, mode := model.QuantizationParams(quantType)
			weight := mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 64, 64), 0.125)
			packed, scales, qbias := mlx.Quantize(weight, groupSize, bits, mode)
			tensors := map[string]*mlx.Array{
				"media.projection.weight":       packed,
				"media.projection.weight_scale": scales,
				"media.projection.bias":         mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 64), 0.25),
			}
			if qbias != nil {
				tensors["media.projection.weight_qbias"] = qbias
			}
			tensorQuant := map[string]*model.TensorQuantInfo{
				"media.projection.weight": {QuantType: quantType, GroupSize: groupSize},
			}
			if err := validateGemma4LinearWeight(tensors, "media.projection", []int{64, 64}, 0, 0, "", tensorQuant); err != nil {
				t.Fatal(err)
			}
			validScales := scales
			if mode == "affine" {
				tensors["media.projection.weight_scale"] = mlx.Zeros(mlx.DTypeUint8, scales.Dims()...)
				err := validateGemma4LinearWeight(tensors, "media.projection", []int{64, 64}, 0, 0, "", tensorQuant)
				if err == nil || !strings.Contains(err.Error(), "unsupported dtype") {
					t.Fatalf("affine uint8 scales error = %v", err)
				}
				tensors["media.projection.weight_scale"] = validScales
				delete(tensors, "media.projection.weight_qbias")
				err = validateGemma4LinearWeight(tensors, "media.projection", []int{64, 64}, 0, 0, "", tensorQuant)
				if err == nil || !strings.Contains(err.Error(), "weight_qbias") {
					t.Fatalf("missing affine qbias error = %v", err)
				}
				tensors["media.projection.weight_qbias"] = qbias
			} else {
				tensors["media.projection.weight_scale"] = mlx.Zeros(mlx.DTypeBFloat16, scales.Dims()...)
				err := validateGemma4LinearWeight(tensors, "media.projection", []int{64, 64}, 0, 0, "", tensorQuant)
				if err == nil || !strings.Contains(err.Error(), "want uint8") {
					t.Fatalf("%s floating scales error = %v", mode, err)
				}
				tensors["media.projection.weight_scale"] = validScales
				tensors["media.projection.weight_qbias"] = mlx.Zeros(mlx.DTypeBFloat16, scales.Dims()...)
				err = validateGemma4LinearWeight(tensors, "media.projection", []int{64, 64}, 0, 0, "", tensorQuant)
				if err == nil || !strings.Contains(err.Error(), "unexpected") {
					t.Fatalf("%s qbias error = %v", mode, err)
				}
				delete(tensors, "media.projection.weight_qbias")
			}
			linear, ok := model.NewLinearFactory(tensors, 0, 0, "", tensorQuant).Make("media.projection").(*nn.QuantizedLinear)
			if !ok {
				t.Fatal("quantized media linear was not constructed")
			}
			if linear.Bias != tensors["media.projection.bias"] || linear.QBiases != qbias {
				t.Fatal("learned and quantization biases were not kept separate")
			}
			output := linear.Forward(mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 1, 64), 1))
			mlx.Eval(output)
			if got := output.Dims(); len(got) != 2 || got[0] != 1 || got[1] != 64 {
				t.Fatalf("quantized media output shape = %v, want [1 64]", got)
			}
			floatOutput := output.AsType(mlx.DTypeFloat32)
			mlx.Eval(floatOutput)
			for i, value := range floatOutput.Floats() {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("quantized media output[%d] is non-finite: %g", i, value)
				}
			}
		})
	}
}
