//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("gptoss-dense-blob-compare is only supported on darwin")
}
