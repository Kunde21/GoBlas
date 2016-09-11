package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/gonum/blas"
	"github.com/gonum/blas/native"
	"github.com/gonum/matrix/mat64"
)

func TestMM(t *testing.T) {
	lim := 25

	i := 0
	for m := 1; m < lim; m++ {
		for n := 1; n < lim; n++ {
			for k := 1; k < lim; k++ {
				t.Run(fmt.Sprint(i, m, n, k), testmm(i, m, n, k))
				i++
				if i%1000 == 0 {
					fmt.Println(i, m, n, k)
				}
			}
		}
	}
}

func testmm(tnum, m, n, k int) func(t *testing.T) {
	return func(t *testing.T) {
		// t.Parallel()
		// m,n,k := m,n,k
		A, B := make([]float64, m*k), make([]float64, k*n)
		A1, B1 := make([]float64, m*k), make([]float64, k*n)
		A2, B2 := make([]float64, m*k), make([]float64, k*n)

		r := rand.New(rand.NewSource(time.Now().UnixNano()))

		for i := range A {
			A[i] = r.Float64()
		}
		for i := range B {
			B[i] = r.Float64()
		}

		C := make([]float64, m*n)

		w := new(sync.WaitGroup)
		w.Add(2)
		go func() {
			gemmNcopy_2(k, m, A, k, A1)
			// fmt.Println(m,n,k,A,A2)
			dgemm_ncopy_2(k, m, A, k, A2)
			for i := range A1 {
				if A1[i] != A2[i] {
					t.Errorf("A Copy mismatch %v:\nBase:%v\nGo: %v\nAsm: %v\n\n", tnum, A, A1, A2)
					break
				}
			}
			runtime.GC()
			w.Done()
		}()

		go func() {
			gemmTcopy_2(k, n, B, n, B1)
			dgemm_tcopy_2(n, k, B, n, B2)
			for i := range B1 {
				if B1[i] != B2[i] {
					t.Errorf("B Copy mismatch %v:\nBase:%v\nGo: %v\nAsm: %v\n\n", tnum, B, B1, B2)
					break
				}
			}
			runtime.GC()
			w.Done()
		}()

		w.Wait()
		gemmKernel_2x2(n, m, k, 1, B2, A2, C, n)
		// fmt.Println("Both N copy:", C)

		Am := mat64.NewDense(m, k, A)
		Bm := mat64.NewDense(k, n, B)
		Cm := mat64.NewDense(m, n, nil)
		Cm.Mul(Am, Bm)

		Cdat := Cm.RawMatrix().Data
		for i, v := range Cdat {
			if v != C[i] {
				t.Errorf("Result mismatch %v:\nGoNum: %v\nAsm: %v\n\n", tnum, Cdat, C)
			}
		}
		// fmt.Println(Cdat)
		// fmt.Println(C)
	}
}

var m, n, k = 1024, 1024, 256

func BenchmarkMat64(t *testing.B) {
	A, B := make([]float64, m*k), make([]float64, k*n)
	for i := range A {
		A[i] = float64(i + 1)
	}
	for i := range B {
		B[i] = -float64(i + 1)
	}
	t.ResetTimer()
	t.ReportAllocs()
	for i := 0; i < t.N; i++ {
		Am := mat64.NewDense(m, k, A)
		Bm := mat64.NewDense(k, n, B)
		Cm := mat64.NewDense(m, n, nil)
		Cm.Mul(Am, Bm)
	}
}

func BenchmarkPGoBlas(t *testing.B) {
	A, B := make([]float64, m*k), make([]float64, k*n)
	C := make([]float64, m*n)
	for i := range A {
		A[i] = float64(i + 1)
	}
	for i := range B {
		B[i] = -float64(i + 1)
	}
	A1, B1 := make([]float64, m*k), make([]float64, k*n)
	w := new(sync.WaitGroup)
	w.Add(1)
	go func() {
		dgemm_ncopy_2(k, m, A, k, A1)
		w.Done()
	}()
	dgemm_tcopy_2(n, k, B, n, B1)
	w.Wait()

	t.ResetTimer()
	t.ReportAllocs()
	for i := 0; i < t.N; i++ {
		gemmKernel_2x2(n, m, k, 1, B1, A1, C, n)
	}
}

func BenchmarkGoBlas(t *testing.B) {
	A, B := make([]float64, m*k), make([]float64, k*n)
	C := make([]float64, m*n)
	for i := range A {
		A[i] = float64(i + 1)
	}
	for i := range B {
		B[i] = -float64(i + 1)
	}
	A1, B1 := make([]float64, m*k), make([]float64, k*n)
	dgemm_ncopy_2(k, m, A, k, A1)
	dgemm_tcopy_2(n, k, B, n, B1)

	t.ResetTimer()
	t.ReportAllocs()
	for i := 0; i < t.N; i++ {
		gemmKernel_2x2(n, m, k, 1, B1, A1, C, n)
	}
}

func BenchmarkNCpy(t *testing.B) {
	A := make([]float64, m*k)
	t.ResetTimer()
	t.ReportAllocs()
	for i := 0; i < t.N; i++ {
		A1 := make([]float64, m*k)
		dgemm_ncopy_2(k, m, A, k, A1)
	}
}

func BenchmarkTCpy(t *testing.B) {
	B := make([]float64, k*n)
	t.ResetTimer()
	t.ReportAllocs()
	for i := 0; i < t.N; i++ {
		B1 := make([]float64, k*n)
		dgemm_tcopy_2(n, k, B, n, B1)
	}
}

func BenchmarkGonumBlas(t *testing.B) {
	A, B := make([]float64, m*k), make([]float64, k*n)
	C := make([]float64, m*n)
	for i := range A {
		A[i] = float64(i + 1)
	}
	for i := range B {
		B[i] = -float64(i + 1)
	}
	var imp native.Implementation
	t.ResetTimer()
	t.ReportAllocs()
	for i := 0; i < t.N; i++ {
		imp.Dgemm(blas.NoTrans, blas.NoTrans, m, n, k, 2, A, m, B, k, 2, C, m)
	}
}
