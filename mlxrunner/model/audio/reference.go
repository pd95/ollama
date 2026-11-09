package audio

import (
	"encoding/binary"
	"errors"
	"math"
)

// ValidateReferenceWAV admits the bounded mono PCM16 format used for voice
// cloning. This is stricter than Decode, which also supports general media.
// Unknown-length streamed RIFF/data sizes are accepted when the actual bytes
// form a complete clip. Ordinary declared chunk sizes must match the input.
func ValidateReferenceWAV(data []byte) error {
	if len(data) < 44 || len(data) > 4<<20 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return errors.New("reference_audio must contain a mono PCM16 WAV")
	}
	riffSize := binary.LittleEndian.Uint32(data[4:])
	if riffSize != math.MaxUint32 && uint64(riffSize)+8 != uint64(len(data)) {
		return errors.New("reference_audio has an incomplete RIFF container")
	}
	var rate, samples int
	foundFormat, foundData := false, false
	for offset := 12; offset < len(data); {
		if len(data)-offset < 8 {
			return errors.New("reference_audio has an incomplete WAV chunk")
		}
		id := string(data[offset : offset+4])
		size := uint64(binary.LittleEndian.Uint32(data[offset+4:]))
		offset += 8
		if id == "data" && size == math.MaxUint32 {
			size = uint64(len(data) - offset)
		}
		if size > uint64(len(data)-offset) {
			return errors.New("reference_audio has an incomplete WAV chunk")
		}
		body := data[offset : offset+int(size)]
		switch id {
		case "fmt ":
			if foundFormat || len(body) < 16 || binary.LittleEndian.Uint16(body) != 1 || binary.LittleEndian.Uint16(body[2:]) != 1 || binary.LittleEndian.Uint16(body[14:]) != 16 {
				return errors.New("reference_audio requires one mono PCM16 format chunk")
			}
			rate = int(binary.LittleEndian.Uint32(body[4:]))
			if rate < 16000 || rate > 48000 || binary.LittleEndian.Uint16(body[12:]) != 2 || binary.LittleEndian.Uint32(body[8:]) != uint32(rate*2) {
				return errors.New("reference_audio requires a sample rate from 16000 to 48000 Hz with consistent PCM16 alignment")
			}
			foundFormat = true
		case "data":
			if foundData || len(body) == 0 || len(body)%2 != 0 {
				return errors.New("reference_audio requires one nonempty complete PCM16 data chunk")
			}
			samples, foundData = len(body)/2, true
		}
		offset += int(size) + int(size%2)
		if offset > len(data) {
			return errors.New("reference_audio is missing WAV chunk padding")
		}
	}
	if !foundFormat || !foundData || samples > rate*15 {
		return errors.New("reference_audio requires mono PCM16 WAV at 16000–48000 Hz, at most 15 seconds")
	}
	return nil
}

// DecodeReferenceWAV validates before allocating float samples. Models resample
// the result to their encoder's rate using Resample.
func DecodeReferenceWAV(data []byte) ([]float32, int, error) {
	if err := ValidateReferenceWAV(data); err != nil {
		return nil, 0, err
	}
	return DecodeLimit(data, 15)
}
