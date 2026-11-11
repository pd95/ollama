package qwen3asr

// EstimatedLoadMemory accounts for original BF16 tensors and promoted F32
// weights coexisting during loading. The validated schema bounds all dimensions.
// This is tensor memory, not a request activation budget or a hard RSS limit.
func EstimatedLoadMemory(config []byte) (uint64, error) {
	shapes, err := TensorShapes(config)
	if err != nil {
		return 0, err
	}
	var total uint64
	for _, shape := range shapes {
		elements := uint64(1)
		for _, d := range shape {
			elements *= uint64(d)
		}
		total += elements * (2 + 4)
	}
	return total, nil
}
