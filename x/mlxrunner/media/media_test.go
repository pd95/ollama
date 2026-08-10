package media

import (
	"math"
	"testing"
)

func TestAddTokenBudget(t *testing.T) {
	if got, err := AddTokenBudget(2, 3); err != nil || got != 5 {
		t.Fatalf("AddTokenBudget(2, 3) = %d, %v; want 5, nil", got, err)
	}
	if _, err := AddTokenBudget(math.MaxInt, 1); err == nil {
		t.Fatal("AddTokenBudget() accepted overflow")
	}
	if _, err := AddTokenBudget(0, 0); err == nil {
		t.Fatal("AddTokenBudget() accepted zero tokens")
	}
}
