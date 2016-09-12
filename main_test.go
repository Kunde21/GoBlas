package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/Kunde21/internal/asm/f64"
	"github.com/Kunde21/numgo"
	"github.com/gonum/blas"
	"github.com/gonum/blas/native"
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
		A, B := make([]float64, m*k), make([]float64, k*n)
		A1, B1 := make([]float64, m*k), make([]float64, k*n)

		r := rand.New(rand.NewSource(time.Now().UnixNano()))

		for i := range A {
			A[i] = r.Float64() // float64(i + 1) //
		}
		for i := range B {
			B[i] = -r.Float64() // -float64(i + 1) //
		}

		C := make([]float64, m*n)
		C1 := make([]float64, m*n)

		dgemm_ncopy_2(k, m, A, k, A1)
		dgemm_tcopy_2(n, k, B, n, B1)
		f64.ScalUnitary(2, C)

		gemmKernel_2x2(n, m, k, 2, B1, A1, C, n)
		var imp native.Implementation
		imp.Dgemm(blas.NoTrans, blas.NoTrans, m, n, k, 2, A, k, B, n, 2, C1, n)

		for i, v := range C1 {
			if v != C[i] {
				t.Errorf("Result mismatch %v:\nGoNum: %v\nAsm: %v\n\n", tnum, C, C1)
				break
			}
			C[i] = 0
		}

		dgemm_full(m, n, k, 2, A, k, B, n, 2, C, n)
		for i, v := range C1 {
			if math.Abs(v-C[i]) > 1e-10 {
				t.Errorf("Result mismatch %v:\nGoNum:\n%v\nGN_Asm:\n%v\n\n", tnum, numgo.NewArray64(C1, m, n), numgo.NewArray64(C, m, n))
				break
			}
		}
	}
}

func dgemm_full(m, n, k int, Alpha float64,
	A []float64, lda int, B []float64, ldb int, Beta float64, C []float64, ldc int) {
	A1, B1 := make([]float64, m*k), make([]float64, k*n)
	wg := new(sync.WaitGroup)

	wg.Add(1)
	go func() {
		dgemm_ncopy_2(k, m, A, k, A1)
		f64.ScalUnitary(Beta, C)
		wg.Done()
	}()
	dgemm_tcopy_2(n, k, B, n, B1)
	wg.Wait()

	ParallelDGemm(m, n, k, Alpha, A1, B1, C, ldc)
}

func ParallelDGemm(m, n, k int, Alpha float64, A, B, C []float64, ldc int) {
	wrk := make(chan int)
	wg := new(sync.WaitGroup)

	wg.Add(NumCPU)
	for j := 0; j < NumCPU; j++ {
		go func(win <-chan int) {
			for w := range win {
				gemm_kernel_2x2(2, n, k, Alpha, A[w*2*k:], B, C[w*2*n:], n)
			}
			wg.Done()
		}(wrk)
	}
	for j := 0; j < m>>1; j++ {
		wrk <- j
	}
	if m&1 > 0 {
		gemm_kernel_2x2(1, n, k, Alpha, A[(m-1)*k:], B, C[(m-1)*n:], n)
	}
	close(wrk)
	wg.Wait()
}
