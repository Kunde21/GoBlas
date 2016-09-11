// ******************************************************************
// Copyright ©2016 Chad Kunde. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The origianal code comes from OpenBLAS library,
// which can be found at www.openblas.net 
// and is distributed under the following terms:
//
// *******************************************************************

// *******************************************************************
// Copyright 2009, 2010 The University of Texas at Austin.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or
// without modification, are permitted provided that the following
// conditions are met:
//
// 1. Redistributions of source code must retain the above
// copyright notice, this list of conditions and the following
// disclaimer.
//
// 2. Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following
// disclaimer in the documentation and/or other materials
// provided with the distribution.
//
// THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT
// AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,
// INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF
// MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE
// DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT
// AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE
// GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF
// LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT
// OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE
// POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and
// documentation are those of the authors and should not be
// interpreted as representing official policies, either expressed
// or implied, of The University of Texas at Austin.
// *******************************************************************

#include "textflag.h"

#define SIZE 8

#define PREFETCH	PREFETCHT0
#define PREFETCHW	PREFETCHT0

#define N	SI
#define M	DI
#define A	DX
#define LDA	CX
#define B	R8

#define AO1	R9
#define AO2	R10
#define LDA3	R11
#define M8	R12

#define I	AX
#define B0	BP
#define B3	R13

// func dgemm_tcopy_2(m, n int, a []float64, lda int, b []float64)
TEXT ·dgemm_tcopy_2(SB), NOSPLIT, $0

	MOVQ m+0(FP), M
	MOVQ n+8(FP), N
	MOVQ a_base+16(FP), A
	MOVQ lda+40(FP), LDA
	MOVQ b_base+48(FP), B

	SUBQ $-16 * SIZE, B

	MOVQ  M, B3
	ANDQ  $-2, B3
	IMULQ N, B3

	LEAQ (B)(B3*SIZE), B3

	LEAQ (LDA*SIZE), LDA
	LEAQ (LDA)(LDA*2), LDA3

	LEAQ (N*SIZE), M8

	CMPQ N, $2
	JL   L40

L31:
	SUBQ $2, N

	MOVQ A, AO1
	LEAQ (A)(LDA*1), AO2
	LEAQ (A)(LDA*2), A

	MOVQ B, B0
	ADDQ $4 * SIZE, B

	MOVQ M, I
	SARQ $3, I
	JLE  L34

L33:

	PREFETCH 16 * 2 * SIZE(AO1)

	MOVUPS (AO1), X0
	MOVUPS 2 * SIZE(AO1), X1
	MOVUPS (AO2), X2
	MOVUPS 2 * SIZE(AO2), X3

	PREFETCHW 16 * 4 * SIZE(B)

	MOVAPS X0, -16 * SIZE(B0)
	MOVAPS X2, -14 * SIZE(B0)
	MOVAPS X1, -16 * SIZE(B0)(M8*2)
	MOVAPS X3, -14 * SIZE(B0)(M8*2)

	LEAQ (B0)(M8*4), B0

	PREFETCH 16 * 2 * SIZE(AO2)

	MOVUPS 4 * SIZE(AO1), X0
	MOVUPS 6 * SIZE(AO1), X1
	MOVUPS 4 * SIZE(AO2), X2
	MOVUPS 6 * SIZE(AO2), X3

	MOVAPS X0, -16 * SIZE(B0)
	MOVAPS X2, -14 * SIZE(B0)
	MOVAPS X1, -16 * SIZE(B0)(M8*2)
	MOVAPS X3, -14 * SIZE(B0)(M8*2)

	LEAQ (B0)(M8*4), B0

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2

	DECQ I
	JG   L33

L34:
	TESTQ $4, M
	JLE   L36

	MOVUPS (AO1), X0
	MOVUPS 2 * SIZE(AO1), X1
	MOVUPS (AO2), X2
	MOVUPS 2 * SIZE(AO2), X3

	MOVAPS X0, -16 * SIZE(B0)
	MOVAPS X2, -14 * SIZE(B0)
	MOVAPS X1, -16 * SIZE(B0)(M8*2)
	MOVAPS X3, -14 * SIZE(B0)(M8*2)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	LEAQ (B0)(M8*4), B0

L36:
	TESTQ $2, M
	JLE   L38

	MOVUPS (AO1), X0
	MOVUPS (AO2), X1

	MOVAPS X0, -16 * SIZE(B0)
	MOVAPS X1, -14 * SIZE(B0)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	LEAQ (B0)(M8*2), B0

L38:
	TESTQ $1, M
	JLE   L39

	MOVSD  0 * SIZE(AO1), X0
	MOVHPD 0 * SIZE(AO2), X0

	MOVAPS X0, -16 * SIZE(B3)
	SUBQ   $-2 * SIZE, B3

L39:
	CMPQ N, $2
	JGE  L31

L40:
	CMPQ N, $1
	JL   L999

	MOVQ A, AO1
	MOVQ B, B0

	MOVQ M, I
	SARQ $3, I
	JLE  L44

L43:

	PREFETCH 16 * 4 * SIZE(AO1)

	MOVUPS (AO1), X0
	MOVUPS 2 * SIZE(AO1), X1
	MOVUPS 4 * SIZE(AO1), X2
	MOVUPS 6 * SIZE(AO1), X3

	PREFETCHW 16 * 4 * SIZE(B)

	ADDQ $8 * SIZE, AO1

	MOVAPS X0, -16 * SIZE(B0)
	MOVAPS X1, -16 * SIZE(B0)(M8*2)
	LEAQ   (B0)(M8*4), B0
	MOVAPS X2, -16 * SIZE(B0)
	MOVAPS X3, -16 * SIZE(B0)(M8*2)
	LEAQ   (B0)(M8*4), B0

	DECQ I
	JG   L43

L44:
	TESTQ $4, M
	JLE   L45

	MOVUPS (AO1), X0
	MOVUPS 2 * SIZE(AO1), X1

	ADDQ $4 * SIZE, AO1

	MOVAPS X0, -16 * SIZE(B0)
	MOVAPS X1, -16 * SIZE(B0)(M8*2)
	LEAQ   (B0)(M8*4), B0

L45:
	TESTQ $2, M
	JLE   L46

	MOVUPS (AO1), X0

	MOVAPS X0, -16 * SIZE(B0)

	ADDQ $2 * SIZE, AO1

L46:
	TESTQ $1, M
	JLE   L999

	MOVSD 0 * SIZE(AO1), X0

	MOVLPD X0, -16 * SIZE(B3)

L999:
	RET
