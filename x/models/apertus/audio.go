package apertus

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

const (
	maxApertusAudioBytes      = 32 << 20
	maxApertusAudioChannels   = 8
	minApertusAudioSampleRate = 8_000
	maxApertusAudioSampleRate = 192_000
	maxApertusAudioSamples    = 24_000 * 600
)

type AudioTokenizerConfig struct {
	AudioChannels      int32   `json:"audio_channels"`
	CodebookDim        int32   `json:"codebook_dim"`
	CodebookSize       int32   `json:"codebook_size"`
	Compress           int32   `json:"compress"`
	DilationGrowthRate int32   `json:"dilation_growth_rate"`
	HiddenSize         int32   `json:"hidden_size"`
	KernelSize         int32   `json:"kernel_size"`
	LastKernelSize     int32   `json:"last_kernel_size"`
	NormType           string  `json:"norm_type"`
	NumFilters         int32   `json:"num_filters"`
	NumLSTMLayers      int32   `json:"num_lstm_layers"`
	NumResidualLayers  int32   `json:"num_residual_layers"`
	PadMode            string  `json:"pad_mode"`
	ResidualKernelSize int32   `json:"residual_kernel_size"`
	SamplingRate       int32   `json:"sampling_rate"`
	UpsamplingRatios   []int32 `json:"upsampling_ratios"`
	UseCausalConv      bool    `json:"use_causal_conv"`
	UseConvShortcut    bool    `json:"use_conv_shortcut"`
}

func (c AudioTokenizerConfig) validate() error {
	if c.AudioChannels != 1 || c.CodebookDim != 512 || c.CodebookSize != 4096 || c.Compress != 2 ||
		c.DilationGrowthRate != 2 || c.HiddenSize != 512 || c.KernelSize != 7 || c.LastKernelSize != 7 ||
		c.NormType != "weight_norm" || c.NumFilters != 32 || c.NumLSTMLayers != 2 || c.NumResidualLayers != 1 ||
		c.PadMode != "reflect" || c.ResidualKernelSize != 3 || c.SamplingRate != 24000 ||
		!slices.Equal(c.UpsamplingRatios, []int32{6, 5, 5, 4}) || c.UseCausalConv || !c.UseConvShortcut {
		return errors.New("unsupported Apertus 1.5 audio tokenizer configuration")
	}
	return nil
}

func (c AudioTokenizerConfig) hopLength() int {
	hop := 1
	for _, ratio := range c.UpsamplingRatios {
		hop *= int(ratio)
	}
	return hop
}

type apertureAudioConv struct {
	conv                     *nn.Conv1d
	kernel, stride, dilation int32
}

func loadApertureAudioConv(tensors map[string]*mlx.Array, path string, stride, dilation int32) (*apertureAudioConv, error) {
	g := tensors[path+".conv.parametrizations.weight.original0"]
	v := tensors[path+".conv.parametrizations.weight.original1"]
	b := tensors[path+".conv.bias"]
	if g == nil || v == nil || b == nil || g.NumDims() != 3 || v.NumDims() != 3 || b.NumDims() != 1 || g.Dim(0) != v.Dim(0) || b.Dim(0) != v.Dim(0) {
		return nil, fmt.Errorf("missing or invalid Apertus audio convolution %s", path)
	}
	norm := mlx.Sum(mlx.Mul(v, v), 2, true)
	norm = mlx.Sum(norm, 1, true).Sqrt()
	weight := mlx.Mul(v, mlx.Div(g, norm))
	weight = mlx.Transpose(weight, 0, 2, 1)
	return &apertureAudioConv{conv: nn.NewConv1d(weight, b, stride, 0, dilation, 1), kernel: int32(v.Dim(2)), stride: stride, dilation: dilation}, nil
}

func (c *apertureAudioConv) forward(x *mlx.Array) *mlx.Array {
	length := int32(x.Dim(1))
	effective := (c.kernel-1)*c.dilation + 1
	paddingTotal := effective - c.stride
	right := paddingTotal / 2
	left := paddingTotal - right
	extra := ((length+c.stride-1)/c.stride)*c.stride - length
	maxPad := max(left, right+extra)
	extraZero := int32(0)
	reflectLength := length
	if length <= maxPad {
		extraZero = maxPad - length + 1
		x = mlx.PadConstant(x, []int{1}, []int{0}, []int{int(extraZero)})
		reflectLength += extraZero
	}
	paddedLength := reflectLength + left + right + extra
	indices := make([]int32, paddedLength)
	period := max(int32(1), 2*(reflectLength-1))
	for i := range indices {
		position := int32(i) - left
		position %= period
		if position < 0 {
			position += period
		}
		if position >= reflectLength {
			position = period - position
		}
		indices[i] = position
	}
	x = mlx.Take(x, mlx.FromValues(indices, len(indices)), 1)
	if extraZero > 0 {
		x = x.Slice(mlx.Slice(), mlx.Slice(0, int(length+left+right+extra)), mlx.Slice())
	}
	return c.conv.Forward(x)
}

func apertureELU(x *mlx.Array) *mlx.Array {
	zero := mlx.NewScalarArray(float32(0))
	return mlx.Where(x.Greater(zero), x, mlx.Sub(mlx.Exp(x), mlx.NewScalarArray(float32(1))))
}

type apertureAudioResBlock struct{ first, second, shortcut *apertureAudioConv }

func loadApertureAudioResBlock(tensors map[string]*mlx.Array, path string) (*apertureAudioResBlock, error) {
	a, err := loadApertureAudioConv(tensors, path+".block.1", 1, 1)
	if err != nil {
		return nil, err
	}
	b, err := loadApertureAudioConv(tensors, path+".block.3", 1, 1)
	if err != nil {
		return nil, err
	}
	s, err := loadApertureAudioConv(tensors, path+".shortcut", 1, 1)
	if err != nil {
		return nil, err
	}
	return &apertureAudioResBlock{first: a, second: b, shortcut: s}, nil
}

func (b *apertureAudioResBlock) forward(x *mlx.Array) *mlx.Array {
	h := b.first.forward(apertureELU(x))
	h = b.second.forward(apertureELU(h))
	return mlx.Add(b.shortcut.forward(x), h)
}

type apertureLSTM struct {
	inputWeights, hiddenWeights []*mlx.Array
	inputBiases, hiddenBiases   []*mlx.Array
	hidden                      int32
}

func loadApertureLSTM(tensors map[string]*mlx.Array, path string, layers int, hidden int32) (*apertureLSTM, error) {
	l := &apertureLSTM{hidden: hidden}
	for i := range layers {
		s := fmt.Sprintf("%s.lstm", path)
		inputWeight, hiddenWeight := tensors[fmt.Sprintf("%s.weight_ih_l%d", s, i)], tensors[fmt.Sprintf("%s.weight_hh_l%d", s, i)]
		inputBias, hiddenBias := tensors[fmt.Sprintf("%s.bias_ih_l%d", s, i)], tensors[fmt.Sprintf("%s.bias_hh_l%d", s, i)]
		if inputWeight == nil || hiddenWeight == nil || inputBias == nil || hiddenBias == nil {
			return nil, fmt.Errorf("missing Apertus audio LSTM layer %d", i)
		}
		l.inputWeights = append(l.inputWeights, inputWeight)
		l.hiddenWeights = append(l.hiddenWeights, hiddenWeight)
		l.inputBiases = append(l.inputBiases, inputBias)
		l.hiddenBiases = append(l.hiddenBiases, hiddenBias)
	}
	return l, nil
}

func (l *apertureLSTM) forward(input *mlx.Array) (*mlx.Array, error) {
	residual := input
	x := input
	for layer := range l.inputWeights {
		if x == nil || x.Dim(1) == 0 {
			return nil, fmt.Errorf("Apertus audio LSTM layer %d received an empty sequence", layer)
		}
		h := mlx.Zeros(mlx.DTypeFloat32, 1, int(l.hidden))
		c := mlx.Zeros(mlx.DTypeFloat32, 1, int(l.hidden))
		steps := x.Dim(1)
		outputs := make([]*mlx.Array, 0, steps)
		for t := range steps {
			xt := x.Slice(mlx.Slice(), mlx.Slice(t, t+1), mlx.Slice()).Squeeze(1)
			gates := mlx.Add(mlx.Add(mlx.Matmul(xt, mlx.Transpose(l.inputWeights[layer], 1, 0)), l.inputBiases[layer]), mlx.Add(mlx.Matmul(h, mlx.Transpose(l.hiddenWeights[layer], 1, 0)), l.hiddenBiases[layer]))
			i := mlx.Sigmoid(gates.Slice(mlx.Slice(), mlx.Slice(0, int(l.hidden))))
			f := mlx.Sigmoid(gates.Slice(mlx.Slice(), mlx.Slice(int(l.hidden), int(2*l.hidden))))
			g := gates.Slice(mlx.Slice(), mlx.Slice(int(2*l.hidden), int(3*l.hidden))).Tanh()
			o := mlx.Sigmoid(gates.Slice(mlx.Slice(), mlx.Slice(int(3*l.hidden), int(4*l.hidden))))
			c = mlx.Add(mlx.Mul(f, c), mlx.Mul(i, g))
			h = mlx.Mul(o, c.Tanh())
			outputs = append(outputs, h.ExpandDims(1))
		}
		x = mlx.Concatenate(outputs, 1)
	}
	return mlx.Add(x, residual), nil
}

type AudioTokenizer struct {
	State      []*mlx.Array
	config     AudioTokenizerConfig
	initial    *apertureAudioConv
	residuals  []*apertureAudioResBlock
	downsample []*apertureAudioConv
	lstm       *apertureLSTM
	final      *apertureAudioConv
	codebook   *mlx.Array
}

func loadAudioTokenizer(tensors map[string]*mlx.Array, cfg AudioTokenizerConfig) (*AudioTokenizer, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	const p = "model.audio_tokenizer.encoder.layers."
	a := &AudioTokenizer{config: cfg}
	var err error
	a.initial, err = loadApertureAudioConv(tensors, p+"0", 1, 1)
	if err != nil {
		return nil, err
	}
	resIndices := []int{1, 4, 7, 10}
	downIndices := []int{3, 6, 9, 12}
	ratios := []int32{4, 5, 5, 6}
	for i := range resIndices {
		r, err := loadApertureAudioResBlock(tensors, fmt.Sprintf("%s%d", p, resIndices[i]))
		if err != nil {
			return nil, err
		}
		a.residuals = append(a.residuals, r)
		d, err := loadApertureAudioConv(tensors, fmt.Sprintf("%s%d", p, downIndices[i]), ratios[i], 1)
		if err != nil {
			return nil, err
		}
		a.downsample = append(a.downsample, d)
	}
	a.lstm, err = loadApertureLSTM(tensors, p+"13", int(cfg.NumLSTMLayers), cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	a.final, err = loadApertureAudioConv(tensors, p+"15", 1, 1)
	if err != nil {
		return nil, err
	}
	a.codebook = tensors["model.audio_tokenizer.quantizer.codebook.embed"]
	if a.codebook == nil || !slices.Equal(a.codebook.Dims(), []int{int(cfg.CodebookSize), int(cfg.CodebookDim)}) {
		return nil, errors.New("missing or invalid Apertus audio codebook")
	}
	for name, tensor := range tensors {
		if (strings.HasPrefix(name, p) || name == "model.audio_tokenizer.quantizer.codebook.embed") && tensor != nil {
			a.State = append(a.State, tensor)
		}
	}
	addConv := func(conv *apertureAudioConv) {
		if conv != nil {
			a.State = append(a.State, conv.conv.Weight)
		}
	}
	addConv(a.initial)
	for i := range a.residuals {
		addConv(a.residuals[i].first)
		addConv(a.residuals[i].second)
		addConv(a.residuals[i].shortcut)
		addConv(a.downsample[i])
	}
	addConv(a.final)
	return a, nil
}

func (a *AudioTokenizer) encode(data *mlx.Array, sampleCount int) (*mlx.Array, error) {
	if sampleCount == 0 {
		return nil, errors.New("empty Apertus audio")
	}
	x := mlx.Reshape(data, 1, int32(sampleCount), 1)
	h := a.initial.forward(x)
	if h.Dim(1) == 0 {
		return nil, fmt.Errorf("Apertus audio initial convolution produced an empty sequence from %d samples", sampleCount)
	}
	for i := range a.residuals {
		h = a.residuals[i].forward(h)
		h = a.downsample[i].forward(apertureELU(h))
		if h.Dim(1) == 0 {
			return nil, fmt.Errorf("Apertus audio downsample stage %d produced an empty sequence", i)
		}
	}
	var err error
	h, err = a.lstm.forward(h)
	if err != nil {
		return nil, err
	}
	h = a.final.forward(apertureELU(h))
	flat := mlx.Reshape(h, int32(h.Dim(1)), a.config.CodebookDim)
	dot := mlx.Mul(mlx.Matmul(flat, mlx.Transpose(a.codebook, 1, 0)), mlx.NewScalarArray(float32(2)))
	codeNorm := mlx.Sum(mlx.Mul(a.codebook, a.codebook), 1, false)
	return mlx.Reshape(mlx.Sub(dot, codeNorm).Argmax(1, false).AsType(mlx.DTypeInt32), 1, int32(h.Dim(1))), nil
}

type apertusAudioInput struct {
	samples []float32
	codes   int
}

func preprocessApertusAudio(ctx context.Context, data []byte, cfg AudioTokenizerConfig) (*apertusAudioInput, error) {
	samples, err := decodeApertusWAV(ctx, data, int(cfg.SamplingRate))
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, errors.New("Apertus audio is empty")
	}
	var peak float32
	for _, v := range samples {
		if p := float32(math.Abs(float64(v))); p > peak {
			peak = p
		}
	}
	if peak > 0 {
		scale := float32(math.Pow(10, -3.0/20.0)) / peak
		for i := range samples {
			samples[i] *= scale
		}
	}
	return &apertusAudioInput{samples: samples, codes: (len(samples) + cfg.hopLength() - 1) / cfg.hopLength()}, nil
}

func decodeApertusWAV(ctx context.Context, data []byte, targetRate int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) > maxApertusAudioBytes {
		return nil, fmt.Errorf("Apertus audio is %d bytes, limit %d", len(data), maxApertusAudioBytes)
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, errors.New("Apertus audio must be a RIFF/WAVE file")
	}
	var format, channels, bits, align uint16
	var rate uint32
	var pcm []byte
	for off := uint64(12); off+8 <= uint64(len(data)); {
		start := int(off)
		size := uint64(binary.LittleEndian.Uint32(data[start+4 : start+8]))
		begin, end := off+8, off+8+size
		if end < begin || end > uint64(len(data)) {
			return nil, errors.New("truncated Apertus WAV chunk")
		}
		chunk := data[int(begin):int(end)]
		switch string(data[start : start+4]) {
		case "fmt ":
			if len(chunk) < 16 {
				return nil, errors.New("Apertus WAV fmt chunk is too short")
			}
			format = binary.LittleEndian.Uint16(chunk)
			channels = binary.LittleEndian.Uint16(chunk[2:])
			rate = binary.LittleEndian.Uint32(chunk[4:])
			align = binary.LittleEndian.Uint16(chunk[12:])
			bits = binary.LittleEndian.Uint16(chunk[14:])
			if format == 0xfffe {
				if len(chunk) < 40 || binary.LittleEndian.Uint16(chunk[16:]) < 22 {
					return nil, errors.New("Apertus extensible WAV fmt chunk is too short")
				}
				guid := chunk[24:40]
				standardTail := []byte{0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}
				if !slices.Equal(guid[4:], standardTail) {
					return nil, errors.New("unsupported Apertus extensible WAV subformat")
				}
				format = uint16(binary.LittleEndian.Uint32(guid[:4]))
			}
		case "data":
			if pcm == nil {
				pcm = chunk
			}
		}
		off = end + size%2
	}
	if format == 0 || pcm == nil {
		return nil, errors.New("Apertus WAV is missing fmt or data")
	}
	if channels == 0 || channels > maxApertusAudioChannels {
		return nil, fmt.Errorf("unsupported Apertus WAV channels %d", channels)
	}
	if rate < minApertusAudioSampleRate || rate > maxApertusAudioSampleRate {
		return nil, fmt.Errorf("unsupported Apertus WAV sample rate %d", rate)
	}
	valid := format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) || format == 3 && bits == 32
	if !valid {
		return nil, fmt.Errorf("unsupported Apertus WAV encoding format=%d bits=%d", format, bits)
	}
	bps := int(bits / 8)
	frameBytes := int(channels) * bps
	if int(align) != frameBytes || len(pcm)%frameBytes != 0 {
		return nil, errors.New("invalid Apertus WAV block alignment")
	}
	frames := len(pcm) / frameBytes
	if int64(frames)*int64(targetRate) > int64(maxApertusAudioSamples)*int64(rate) {
		return nil, fmt.Errorf("Apertus audio exceeds %d output samples", maxApertusAudioSamples)
	}
	source := make([]float32, frames)
	for i := range frames {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		var sum float64
		for ch := range int(channels) {
			o := (i*int(channels) + ch) * bps
			switch {
			case format == 1 && bits == 8:
				sum += (float64(pcm[o]) - 128) / 128
			case format == 1 && bits == 16:
				sum += float64(int16(binary.LittleEndian.Uint16(pcm[o:o+2]))) / 32768
			case format == 1 && bits == 24:
				v := int32(pcm[o]) | int32(pcm[o+1])<<8 | int32(pcm[o+2])<<16
				if v&0x800000 != 0 {
					v |= ^int32(0xffffff)
				}
				sum += float64(v) / 8388608
			case format == 1 && bits == 32:
				sum += float64(int32(binary.LittleEndian.Uint32(pcm[o:o+4]))) / 2147483648
			case format == 3 && bits == 32:
				v := math.Float32frombits(binary.LittleEndian.Uint32(pcm[o : o+4]))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return nil, errors.New("Apertus WAV contains non-finite samples")
				}
				sum += float64(v)
			}
		}
		source[i] = float32(sum / float64(channels))
	}
	if int(rate) == targetRate {
		return source, nil
	}
	outLen := (len(source)*targetRate + int(rate) - 1) / int(rate)
	out := make([]float32, outLen)
	for i := range out {
		pos := float64(i) * float64(rate) / float64(targetRate)
		lo := int(pos)
		if lo >= len(source)-1 {
			out[i] = source[len(source)-1]
			continue
		}
		frac := float32(pos - float64(lo))
		out[i] = source[lo]*(1-frac) + source[lo+1]*frac
	}
	return out, nil
}
