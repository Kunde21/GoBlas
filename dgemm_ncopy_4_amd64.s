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

#define RPREFETCHSIZE	16
#define WPREFETCHSIZE (RPREFETCHSIZE * 4)
#define PREFETCH      PREFETCHT0
#define PREFETCHW     PREFETCHT0

#define M	DI
#define N	SI
#define A	DX
#define LDA	CX
#define B	R8

#define I	R9

#define J	R10
#define AO1	R11
#define AO2	R12
#define AO3	R13
#define AO4	AX

// func dgemm_ncopy_4(m, n int, a []float64, lda int, b []float64)
TEXT ·dgemm_ncopy_4(SB), NOSPLIT, $0

	MOVQ m+0(FP), M
	MOVQ n+8(FP), N
	MOVQ a_base+16(FP), A
	MOVQ lda+40(FP), LDA
	MOVQ b_base+48(FP), B

	LEAQ (LDA*SIZE), LDA // Scaling

	MOVQ N, J
	SARQ $2, J
	JLE  L20

L12:
	MOVQ A, AO1
	LEAQ (A)(LDA*1), AO2
	LEAQ (A)(LDA*2), AO3
	LEAQ (AO2)(LDA*2), AO4
	LEAQ (A)(LDA*4), A

	MOVQ M, I
	SARQ $2, I
	JLE  L14

L13:

	PREFETCH RPREFETCHSIZE * SIZE(AO1)
	MOVSD    0 * SIZE(AO1), X0
	MOVHPD   0 * SIZE(AO2), X0
	MOVSD    1 * SIZE(AO1), X2
	MOVHPD   1 * SIZE(AO2), X2
	PREFETCH RPREFETCHSIZE * SIZE(AO2)
	MOVSD    2 * SIZE(AO1), X4
	MOVHPD   2 * SIZE(AO2), X4
	MOVSD    3 * SIZE(AO1), X6
	MOVHPD   3 * SIZE(AO2), X6

	PREFETCH RPREFETCHSIZE * SIZE(AO3)
	MOVSD    0 * SIZE(AO3), X1
	MOVHPD   0 * SIZE(AO4), X1
	MOVSD    1 * SIZE(AO3), X3
	MOVHPD   1 * SIZE(AO4), X3
	PREFETCH RPREFETCHSIZE * SIZE(AO4)
	MOVSD    2 * SIZE(AO3), X5
	MOVHPD   2 * SIZE(AO4), X5
	MOVSD    3 * SIZE(AO3), X7
	MOVHPD   3 * SIZE(AO4), X7

	PREFETCHW WPREFETCHSIZE * SIZE(B)
	MOVAPD    X0, 0 * SIZE(B)
	MOVAPD    X1, 2 * SIZE(B)
	MOVAPD    X2, 4 * SIZE(B)
	MOVAPD    X3, 6 * SIZE(B)
	MOVAPD    X4, 8 * SIZE(B)
	MOVAPD    X5, 10 * SIZE(B)
	MOVAPD    X6, 12 * SIZE(B)
	MOVAPD    X7, 14 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	ADDQ $4 * SIZE, AO3
	ADDQ $4 * SIZE, AO4

	SUBQ $-16 * SIZE, B
	DECQ I
	JG   L13

L14:
	MOVQ M, I
	ANDQ $3, I
	JLE  L16

L15:
	MOVSD  0 * SIZE(AO1), X0
	MOVHPD 0 * SIZE(AO2), X0
	MOVSD  0 * SIZE(AO3), X1
	MOVHPD 0 * SIZE(AO4), X1

	MOVAPD X0, 0 * SIZE(B)
	MOVAPD X1, 2 * SIZE(B)

	ADDQ $SIZE, AO1
	ADDQ $SIZE, AO2
	ADDQ $SIZE, AO3
	ADDQ $SIZE, AO4
	ADDQ $4 * SIZE, B
	DECQ I
	JG   L15

L16:
	DECQ J
	JG   L12

L20:
	TESTQ $2, N
	JLE   L30

	MOVQ A, AO1
	LEAQ (A)(LDA*1), AO2
	LEAQ (A)(LDA*2), A

	MOVQ M, I
	SARQ $2, I
	JLE  L24

L23:
	MOVSD  0 * SIZE(AO1), X0
	MOVHPD 0 * SIZE(AO2), X0
	MOVSD  1 * SIZE(AO1), X1
	MOVHPD 1 * SIZE(AO2), X1

	MOVSD  2 * SIZE(AO1), X2
	MOVHPD 2 * SIZE(AO2), X2
	MOVSD  3 * SIZE(AO1), X3
	MOVHPD 3 * SIZE(AO2), X3

	MOVAPD X0, 0 * SIZE(B)
	MOVAPD X1, 2 * SIZE(B)
	MOVAPD X2, 4 * SIZE(B)
	MOVAPD X3, 6 * SIZE(B)

	PREFETCH RPREFETCHSIZE * SIZE(AO1)
	PREFETCH RPREFETCHSIZE * SIZE(AO2)

	PREFETCHW WPREFETCHSIZE * SIZE(B)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	SUBQ $-8 * SIZE, B
	DECQ I
	JG   L23

L24:
	MOVQ M, I
	ANDQ $3, I
	JLE  L30

L25:
	MOVSD  0 * SIZE(AO1), X0
	MOVHPD 0 * SIZE(AO2), X0

	MOVAPD X0, 0 * SIZE(B)

	ADDQ $SIZE, AO1
	ADDQ $SIZE, AO2
	ADDQ $2 * SIZE, B
	DECQ I
	JG   L25

L30:
	TESTQ $1, N
	JLE   L999

	MOVQ A, AO1

	MOVQ M, I
	SARQ $2, I
	JLE  L34

L33:
	MOVSD  0 * SIZE(AO1), X0
	MOVHPD 1 * SIZE(AO1), X0

	MOVSD  2 * SIZE(AO1), X1
	MOVHPD 3 * SIZE(AO1), X1

	MOVAPD X0, 0 * SIZE(B)
	MOVAPD X1, 2 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	SUBQ $-4 * SIZE, B
	DECQ I
	JG   L33

L34:
	MOVQ M, I
	ANDQ $3, I
	JLE  L999

L35:
	MOVSD 0 * SIZE(AO1), X0
	MOVSD X0, 0 * SIZE(B)

	ADDQ $SIZE, AO1
	ADDQ $1 * SIZE, B
	DECQ I
	JG   L35

L999:
	RET
