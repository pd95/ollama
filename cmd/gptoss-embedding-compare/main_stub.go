//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("gptoss-embedding-compare is only supported on darwin")
}
