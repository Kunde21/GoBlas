package main

import "fmt"

func main() {
	m, n, k := 1, 1, 1
	A, B := make([]float64, m*k), make([]float64, k*n)
	A1, B1 := make([]float64, m*k), make([]float64, k*n)

	for i := range A {
		A[i] = float64(i + 1)
	}
	for i := range B {
		B[i] = -float64(i + 1)
	}

	// Both same copy

	/****************************************************************
	//	This block will return column-major result
	//
	// C := make([]float64, m*n)
	//
	// gemmNcopy_2(k, m, A, k, A1)
	// gemmTcopy_2(k, n, B, n, B1)
	// gemmKernel_2x2(m, n, k, 1, A1, B1, C, m)
	// *************************************************************/

	C := make([]float64, m*n)

	dgemm_ncopy_2(k, m, A, k, A1)
	dgemm_tcopy_2(n, k, B, n, B1)
	gemmKernel_2x2(n, m, k, 1, B1, A1, C, n)
	fmt.Println("Result (%v x %v) matrix:", m, n, C)
}
