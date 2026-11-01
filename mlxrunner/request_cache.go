package mlxrunner

// finishRequestCache lets independent complete-audio models release generation
// history on the serialized MLX worker. Other models retain their current cache
// policy. This does not change cache behavior during prefill or token decoding.
func (r *Runner) finishRequestCache() bool {
	if policy, ok := r.Model.(interface{ RetainRequestCache() bool }); ok && !policy.RetainRequestCache() {
		r.cache.close()
		return true
	}
	return false
}
