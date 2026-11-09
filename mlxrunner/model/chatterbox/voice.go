package chatterbox

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type voiceLSTM struct{ wx, wh, bias *mlx.Array }

func (l voiceLSTM) forward(x *mlx.Array) *mlx.Array {
	x = l.bias.Addmm(x, mlx.Transpose(l.wx, 1, 0), 1, 1)
	var hidden, cell *mlx.Array
	all := make([]*mlx.Array, x.Dim(1))
	for step := range all {
		gates := mlx.Squeeze(x.Slice(mlx.Slice(), mlx.Slice(step, step+1), mlx.Slice()), 1)
		if hidden != nil {
			gates = gates.Addmm(hidden, mlx.Transpose(l.wh, 1, 0), 1, 1)
		}
		gate := func(i int) *mlx.Array { return gates.Slice(mlx.Slice(), mlx.Slice(i*256, (i+1)*256)) }
		i, f, g, o := gate(0).Sigmoid(), gate(1).Sigmoid(), gate(2).Tanh(), gate(3).Sigmoid()
		if cell == nil {
			cell = mlx.Mul(i, g)
		} else {
			cell = mlx.Add(mlx.Mul(f, cell), mlx.Mul(i, g))
		}
		hidden = mlx.Mul(o, cell.Tanh())
		all[step] = hidden
	}
	return mlx.Stack(all, 1)
}

type voiceEncoder struct {
	epsilon      float32
	turbo        bool
	layers       []voiceLSTM
	project      nn.LinearLayer
	bank, window *mlx.Array
	derived      []*mlx.Array
}

func loadVoiceEncoder(w *weights) *voiceEncoder {
	v := &voiceEncoder{project: w.linear("ve.proj"), bank: nn.SlaneyMelFilters(16000, 400, 40, 8000)}
	for i := range 3 {
		p := fmt.Sprintf("ve.lstm.layers.%d", i)
		v.layers = append(v.layers, voiceLSTM{w.get(p + ".Wx"), w.get(p + ".Wh"), w.get(p + ".bias")})
	}
	window := make([]float32, 400)
	for i := range window {
		window[i] = float32(.5 * (1 - math.Cos(2*math.Pi*float64(i)/399)))
	}
	v.window = mlx.FromValues(window, 400)
	v.derived = []*mlx.Array{v.bank, v.window}
	return v
}

// VoiceEncoder's energy trim applies only to its speaker example, never to
// generated audio or the reference tokens/mels used by T3 and S3Gen.
func trimVoiceReference(pcm []float32) []float32 {
	frames := 1 + (len(pcm)-2048)/512
	if len(pcm) < 2048 || frames <= 0 {
		return pcm
	}
	rms := make([]float64, frames)
	var peak float64
	for i := range rms {
		var sum float64
		for _, v := range pcm[i*512 : i*512+2048] {
			sum += float64(v) * float64(v)
		}
		rms[i] = math.Sqrt(sum / 2048)
		peak = math.Max(peak, rms[i])
	}
	threshold := math.Max(peak, 1e-10) / 10
	first, last := frames, -1
	for i, value := range rms {
		if math.Max(value, 1e-10) >= threshold {
			first = min(first, i)
			last = i
		}
	}
	if last < first {
		return pcm
	}
	return pcm[first*512 : min((last+1)*512+2048, len(pcm))]
}

func (v *voiceEncoder) mels(pcm *mlx.Array) *mlx.Array {
	padded := mlx.Pad(pcm, []int{0}, []int{200}, []int{200}, mlx.FromValue(float32(0)), "reflect")
	frames := padded.AsStrided([]int{1 + pcm.Size()/160, 400}, []int{160, 1}, 0)
	power := mlx.RFFT(mlx.Mul(frames, v.window), 400, -1).Abs().Power(mlx.FromValue(2))
	if v.turbo {
		return mlx.MatmulCPU(v.bank, mlx.Transpose(power, 1, 0))
	}
	return mlx.Transpose(mlx.Matmul(power, mlx.Transpose(v.bank, 1, 0)), 1, 0)
}

func (v *voiceEncoder) partials(mels *mlx.Array) *mlx.Array {
	// Resemble's rate=1.3 default: round((16000/1.3)/160) = 77 frames.
	const step = 77
	const window = 160
	frames := mels.Dim(1)
	count, remainder := max(frames-window+step, 0)/step, max(frames-window+step, 0)%step
	if count == 0 || float64(remainder+window-step)/window >= .8 {
		count++
	}
	target := window + step*(count-1)
	x := mlx.Transpose(mels, 1, 0)
	if target > frames {
		x = mlx.PadConstant(x, []int{0}, []int{0}, []int{target - frames})
	}
	indices := make([]int32, count*window)
	for i := range count {
		for j := range window {
			indices[i*window+j] = int32(i*step + j)
		}
	}
	return mlx.Reshape(mlx.Take(x, mlx.FromValues(indices, len(indices)), 0), int32(count), window, 40)
}

func (v *voiceEncoder) forward(partials *mlx.Array) *mlx.Array {
	x := partials
	for _, l := range v.layers {
		x = l.forward(x)
	}
	x = mlx.Squeeze(x.Slice(mlx.Slice(), mlx.Slice(x.Dim(1)-1, x.Dim(1)), mlx.Slice()), 1)
	return v.normalize(mlx.ReLU(v.project.Forward(x)))
}

func (v *voiceEncoder) embedding(mels *mlx.Array) *mlx.Array {
	return v.normalize(mlx.Mean(v.forward(v.partials(mels)), 0, true))
}

func (v *voiceEncoder) normalize(x *mlx.Array) *mlx.Array {
	norm := mlx.Sum(x.Power(mlx.FromValue(2)), 1, true).Sqrt()
	if v.epsilon != 0 {
		norm = mlx.Add(norm, mlx.FromValue(v.epsilon))
	}
	return mlx.Div(x, norm)
}
