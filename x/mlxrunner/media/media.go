// Package media provides shared media preprocessing helpers for MLX models.
package media

import (
	"errors"
	"fmt"
	"math"
)

// AddTokenBudget adds a media token count without overflowing int and rejects
// non-positive counts.
func AddTokenBudget(total, count int) (int, error) {
	if count <= 0 {
		return 0, fmt.Errorf("invalid media token count %d", count)
	}
	if total < 0 || count > math.MaxInt-total {
		return 0, errors.New("media token budget overflows int")
	}
	return total + count, nil
}
