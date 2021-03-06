package main

func dgemm_ncopy_2(m, n int, a []float64, lda int, b []float64)

func dgemm_ncopy_4(m, n int, a []float64, lda int, b []float64)

func dgemm_ncopy_8(m, n int, a []float64, lda int, b []float64)

func dgemm_tcopy_2(m, n int, a []float64, lda int, b []float64)

func dgemm_tcopy_4(m, n int, a []float64, lda int, b []float64)

func dgemm_tcopy_8(m, n int, a []float64, lda int, b []float64)

func gemm_kernel_2x2(m, n, k int, alpha float64, a, b, c []float64, ldc int) int

func gemm_kernel_4x4(m, n, k int, alpha float64, A, B, C []float64, ldc int) int