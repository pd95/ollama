These small fixtures contain fixed synthetic values, not trained weights.
They were generated with `torch.save` (PyTorch 2.15.0), using pickle protocols
2 and 4. `synthetic.safetensors` independently stores the same logical tensors
through `safetensors.torch.save_file`.

They cover F32, F16, BF16, I64, scalar/empty tensors, an offset storage view and
an OrderedDict carrying PyTorch state-dictionary metadata. Tests additionally
exercise transposed and partial reads, malformed metadata, archive structure,
storage integrity and nonfinite values. Original-provider archive comparisons
use separately installed model assets; those weights are not included here.
