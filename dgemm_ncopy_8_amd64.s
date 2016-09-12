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

#define PREFETCHSIZE	12
#define PREFETCH	PREFETCHT0
#define PREFETCHW	PREFETCHT0

#define M	DI
#define N	SI
#define A	DX
#define LDA	CX
#define B	R8

#define AO1	R9
#define AO2	R10
#define LDA3	R11
#define J	R12
#define MM	R13

#define I	AX

// func dgemm_ncopy_8(m, n int, a []float64, lda int, b []float64)
TEXT ·dgemm_ncopy_8(SB), NOSPLIT, $0

	MOVQ m+0(FP), M
	MOVQ n+8(FP), N
	MOVQ a_base+16(FP), A
	MOVQ lda+40(FP), LDA
	MOVQ b_base+48(FP), B

	LEAQ (LDA*SIZE), LDA
	LEAQ (LDA)(LDA*2), LDA3
	SUBQ $-16 * SIZE, B

	MOVQ    M, MM
	LEAQ    -1(M), AX
	TESTQ   $SIZE, A
	CMOVQNE AX, MM

	TESTQ $SIZE, LDA
	JNE   L50

	MOVQ N, J
	SARQ $3, J
	JLE  L20

L11:
	MOVQ A, AO1
	LEAQ (A)(LDA*4), AO2
	LEAQ (A)(LDA*8), A

	TESTQ $SIZE, A
	JE    L12

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO1)(LDA*2), X2
	MOVSD 0 * SIZE(AO1)(LDA3*1), X3

	MOVSD 0 * SIZE(AO2), X4
	MOVSD 0 * SIZE(AO2)(LDA*1), X5
	MOVSD 0 * SIZE(AO2)(LDA*2), X6
	MOVSD 0 * SIZE(AO2)(LDA3*1), X7

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2
	UNPCKLPD X5, X4
	UNPCKLPD X7, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-8 * SIZE, B

L12:
	MOVQ MM, I
	SARQ $3, I
	JLE  L14

L13:
	PREFETCH PREFETCHSIZE * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO1)(LDA*2), X2
	MOVAPS 0 * SIZE(AO1)(LDA3*1), X3

	MOVAPS   X0, X8
	UNPCKLPD X1, X0
	MOVAPS   X2, X9
	UNPCKLPD X3, X2

	PREFETCH PREFETCHSIZE * SIZE(AO1)(LDA*1)

	MOVAPS 0 * SIZE(AO2), X4
	MOVAPS 0 * SIZE(AO2)(LDA*1), X5
	MOVAPS 0 * SIZE(AO2)(LDA*2), X6
	MOVAPS 0 * SIZE(AO2)(LDA3*1), X7

	MOVAPS   X4, X10
	UNPCKLPD X5, X4
	MOVAPS   X6, X11
	UNPCKLPD X7, X6

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	UNPCKHPD X1, X8
	UNPCKHPD X3, X9
	UNPCKHPD X5, X10
	UNPCKHPD X7, X11

	PREFETCHW (PREFETCHSIZE * 8 +  8) * SIZE(B)

	MOVAPS X8, -8 * SIZE(B)
	MOVAPS X9, -6 * SIZE(B)
	MOVAPS X10, -4 * SIZE(B)
	MOVAPS X11, -2 * SIZE(B)

	PREFETCH PREFETCHSIZE * SIZE(AO1)(LDA*2)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1)(LDA*1), X1
	MOVAPS 2 * SIZE(AO1)(LDA*2), X2
	MOVAPS 2 * SIZE(AO1)(LDA3*1), X3

	MOVAPS   X0, X8
	UNPCKLPD X1, X0
	MOVAPS   X2, X9
	UNPCKLPD X3, X2

	PREFETCH PREFETCHSIZE * SIZE(AO1)(LDA3*1)

	MOVAPS 2 * SIZE(AO2), X4
	MOVAPS 2 * SIZE(AO2)(LDA*1), X5
	MOVAPS 2 * SIZE(AO2)(LDA*2), X6
	MOVAPS 2 * SIZE(AO2)(LDA3*1), X7

	MOVAPS   X4, X10
	UNPCKLPD X5, X4
	MOVAPS   X6, X11
	UNPCKLPD X7, X6

	PREFETCHW (PREFETCHSIZE * 8 + 16) * SIZE(B)

	MOVAPS X0, 0 * SIZE(B)
	MOVAPS X2, 2 * SIZE(B)
	MOVAPS X4, 4 * SIZE(B)
	MOVAPS X6, 6 * SIZE(B)

	UNPCKHPD X1, X8
	UNPCKHPD X3, X9
	UNPCKHPD X5, X10
	UNPCKHPD X7, X11

	PREFETCHW (PREFETCHSIZE * 8 + 24) * SIZE(B)

	MOVAPS X8, 8 * SIZE(B)
	MOVAPS X9, 10 * SIZE(B)
	MOVAPS X10, 12 * SIZE(B)
	MOVAPS X11, 14 * SIZE(B)

	PREFETCH PREFETCHSIZE * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 4 * SIZE(AO1)(LDA*1), X1
	MOVAPS 4 * SIZE(AO1)(LDA*2), X2
	MOVAPS 4 * SIZE(AO1)(LDA3*1), X3

	MOVAPS   X0, X8
	UNPCKLPD X1, X0
	MOVAPS   X2, X9
	UNPCKLPD X3, X2

	PREFETCH PREFETCHSIZE * SIZE(AO2)(LDA*1)

	MOVAPS 4 * SIZE(AO2), X4
	MOVAPS 4 * SIZE(AO2)(LDA*1), X5
	MOVAPS 4 * SIZE(AO2)(LDA*2), X6
	MOVAPS 4 * SIZE(AO2)(LDA3*1), X7

	MOVAPS   X4, X10
	UNPCKLPD X5, X4
	MOVAPS   X6, X11
	UNPCKLPD X7, X6

	PREFETCHW (PREFETCHSIZE * 8 + 32) * SIZE(B)

	MOVAPS X0, 16 * SIZE(B)
	MOVAPS X2, 18 * SIZE(B)
	MOVAPS X4, 20 * SIZE(B)
	MOVAPS X6, 22 * SIZE(B)

	UNPCKHPD X1, X8
	UNPCKHPD X3, X9
	UNPCKHPD X5, X10
	UNPCKHPD X7, X11

	PREFETCHW (PREFETCHSIZE * 8 + 40) * SIZE(B)

	MOVAPS X8, 24 * SIZE(B)
	MOVAPS X9, 26 * SIZE(B)
	MOVAPS X10, 28 * SIZE(B)
	MOVAPS X11, 30 * SIZE(B)

	PREFETCH PREFETCHSIZE * SIZE(AO2)(LDA*2)

	MOVAPS 6 * SIZE(AO1), X0
	MOVAPS 6 * SIZE(AO1)(LDA*1), X1
	MOVAPS 6 * SIZE(AO1)(LDA*2), X2
	MOVAPS 6 * SIZE(AO1)(LDA3*1), X3

	MOVAPS   X0, X8
	UNPCKLPD X1, X0
	MOVAPS   X2, X9
	UNPCKLPD X3, X2

	PREFETCH PREFETCHSIZE * SIZE(AO2)(LDA3*1)

	MOVAPS 6 * SIZE(AO2), X4
	MOVAPS 6 * SIZE(AO2)(LDA*1), X5
	MOVAPS 6 * SIZE(AO2)(LDA*2), X6
	MOVAPS 6 * SIZE(AO2)(LDA3*1), X7

	MOVAPS   X4, X10
	UNPCKLPD X5, X4
	MOVAPS   X6, X11
	UNPCKLPD X7, X6

	PREFETCHW (PREFETCHSIZE * 8 + 48) * SIZE(B)

	MOVAPS X0, 32 * SIZE(B)
	MOVAPS X2, 34 * SIZE(B)
	MOVAPS X4, 36 * SIZE(B)
	MOVAPS X6, 38 * SIZE(B)

	UNPCKHPD X1, X8
	UNPCKHPD X3, X9
	UNPCKHPD X5, X10
	UNPCKHPD X7, X11

	PREFETCHW (PREFETCHSIZE * 8 + 56) * SIZE(B)

	MOVAPS X8, 40 * SIZE(B)
	MOVAPS X9, 42 * SIZE(B)
	MOVAPS X10, 44 * SIZE(B)
	MOVAPS X11, 46 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-64 * SIZE, B

	DECQ I
	JG   L13

L14:
	TESTQ $4, MM
	JLE   L16

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO1)(LDA*2), X2
	MOVAPS 0 * SIZE(AO1)(LDA3*1), X3

	MOVAPS 0 * SIZE(AO2), X4
	MOVAPS 0 * SIZE(AO2)(LDA*1), X5
	MOVAPS 0 * SIZE(AO2)(LDA*2), X6
	MOVAPS 0 * SIZE(AO2)(LDA3*1), X7

	MOVAPS   X0, X8
	UNPCKLPD X1, X0
	MOVAPS   X2, X9
	UNPCKLPD X3, X2

	MOVAPS   X4, X10
	UNPCKLPD X5, X4
	MOVAPS   X6, X11
	UNPCKLPD X7, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	UNPCKHPD X1, X8
	UNPCKHPD X3, X9
	UNPCKHPD X5, X10
	UNPCKHPD X7, X11

	MOVAPS X8, -8 * SIZE(B)
	MOVAPS X9, -6 * SIZE(B)
	MOVAPS X10, -4 * SIZE(B)
	MOVAPS X11, -2 * SIZE(B)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1)(LDA*1), X1
	MOVAPS 2 * SIZE(AO1)(LDA*2), X2
	MOVAPS 2 * SIZE(AO1)(LDA3*1), X3

	MOVAPS 2 * SIZE(AO2), X4
	MOVAPS 2 * SIZE(AO2)(LDA*1), X5
	MOVAPS 2 * SIZE(AO2)(LDA*2), X6
	MOVAPS 2 * SIZE(AO2)(LDA3*1), X7

	MOVAPS   X0, X8
	UNPCKLPD X1, X0
	MOVAPS   X2, X9
	UNPCKLPD X3, X2

	MOVAPS   X4, X10
	UNPCKLPD X5, X4
	MOVAPS   X6, X11
	UNPCKLPD X7, X6

	MOVAPS X0, 0 * SIZE(B)
	MOVAPS X2, 2 * SIZE(B)
	MOVAPS X4, 4 * SIZE(B)
	MOVAPS X6, 6 * SIZE(B)

	UNPCKHPD X1, X8
	UNPCKHPD X3, X9
	UNPCKHPD X5, X10
	UNPCKHPD X7, X11

	MOVAPS X8, 8 * SIZE(B)
	MOVAPS X9, 10 * SIZE(B)
	MOVAPS X10, 12 * SIZE(B)
	MOVAPS X11, 14 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	SUBQ $-32 * SIZE, B

L16:
	TESTQ $2, MM
	JLE   L18

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO1)(LDA*2), X2
	MOVAPS 0 * SIZE(AO1)(LDA3*1), X3

	MOVAPS 0 * SIZE(AO2), X4
	MOVAPS 0 * SIZE(AO2)(LDA*1), X5
	MOVAPS 0 * SIZE(AO2)(LDA*2), X6
	MOVAPS 0 * SIZE(AO2)(LDA3*1), X7

	MOVAPS   X0, X8
	UNPCKLPD X1, X0
	MOVAPS   X2, X9
	UNPCKLPD X3, X2

	MOVAPS   X4, X10
	UNPCKLPD X5, X4
	MOVAPS   X6, X11
	UNPCKLPD X7, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	UNPCKHPD X1, X8
	UNPCKHPD X3, X9
	UNPCKHPD X5, X10
	UNPCKHPD X7, X11

	MOVAPS X8, -8 * SIZE(B)
	MOVAPS X9, -6 * SIZE(B)
	MOVAPS X10, -4 * SIZE(B)
	MOVAPS X11, -2 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	SUBQ $-16 * SIZE, B

L18:
	TESTQ $1, MM
	JLE   L19

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO1)(LDA*2), X2
	MOVSD 0 * SIZE(AO1)(LDA3*1), X3

	MOVSD 0 * SIZE(AO2), X4
	MOVSD 0 * SIZE(AO2)(LDA*1), X5
	MOVSD 0 * SIZE(AO2)(LDA*2), X6
	MOVSD 0 * SIZE(AO2)(LDA3*1), X7

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2
	UNPCKLPD X5, X4
	UNPCKLPD X7, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	SUBQ $-8 * SIZE, B

L19:
	DECQ J
	JG   L11

L20:
	TESTQ $4, N
	JLE   L30

	MOVQ A, AO1
	LEAQ (A)(LDA*2), AO2
	LEAQ (A)(LDA*4), A

	TESTQ $SIZE, A
	JE    L22

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO2), X2
	MOVSD 0 * SIZE(AO2)(LDA*1), X3

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-4 * SIZE, B

L22:
	MOVQ MM, I
	SARQ $3, I
	JLE  L24

L23:
	PREFETCH PREFETCHSIZE * 2 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO2), X2
	MOVAPS 0 * SIZE(AO2)(LDA*1), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO1)(LDA*1)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1)(LDA*1), X1
	MOVAPS 2 * SIZE(AO2), X2
	MOVAPS 2 * SIZE(AO2)(LDA*1), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	PREFETCHW (PREFETCHSIZE * 8 +  8) * SIZE(B)

	MOVAPS X0, -8 * SIZE(B)
	MOVAPS X2, -6 * SIZE(B)
	MOVAPS X4, -4 * SIZE(B)
	MOVAPS X6, -2 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 4 * SIZE(AO1)(LDA*1), X1
	MOVAPS 4 * SIZE(AO2), X2
	MOVAPS 4 * SIZE(AO2)(LDA*1), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	PREFETCHW (PREFETCHSIZE * 8 + 16) * SIZE(B)

	MOVAPS X0, 0 * SIZE(B)
	MOVAPS X2, 2 * SIZE(B)
	MOVAPS X4, 4 * SIZE(B)
	MOVAPS X6, 6 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO2)(LDA*1)

	MOVAPS 6 * SIZE(AO1), X0
	MOVAPS 6 * SIZE(AO1)(LDA*1), X1
	MOVAPS 6 * SIZE(AO2), X2
	MOVAPS 6 * SIZE(AO2)(LDA*1), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	PREFETCHW (PREFETCHSIZE * 8 + 24) * SIZE(B)

	MOVAPS X0, 8 * SIZE(B)
	MOVAPS X2, 10 * SIZE(B)
	MOVAPS X4, 12 * SIZE(B)
	MOVAPS X6, 14 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-32 * SIZE, B

	DECQ I
	JG   L23

L24:
	TESTQ $4, MM
	JLE   L26

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO2), X2
	MOVAPS 0 * SIZE(AO2)(LDA*1), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1)(LDA*1), X1
	MOVAPS 2 * SIZE(AO2), X2
	MOVAPS 2 * SIZE(AO2)(LDA*1), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	MOVAPS X0, -8 * SIZE(B)
	MOVAPS X2, -6 * SIZE(B)
	MOVAPS X4, -4 * SIZE(B)
	MOVAPS X6, -2 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	SUBQ $-16 * SIZE, B

L26:
	TESTQ $2, MM
	JLE   L28

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO2), X2
	MOVAPS 0 * SIZE(AO2)(LDA*1), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	SUBQ $-8 * SIZE, B

L28:
	TESTQ $1, MM
	JLE   L30

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO2), X2
	MOVSD 0 * SIZE(AO2)(LDA*1), X3

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	SUBQ   $-4 * SIZE, B

L30:
	TESTQ $2, N
	JLE   L40

	MOVQ A, AO1
	LEAQ (A)(LDA*1), AO2
	LEAQ (A)(LDA*2), A

	TESTQ $SIZE, A
	JE    L32

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-2 * SIZE, B

L32:
	MOVQ MM, I
	SARQ $3, I
	JLE  L34

L33:
	PREFETCH PREFETCHSIZE * 4 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO2), X1
	MOVAPS 2 * SIZE(AO1), X2
	MOVAPS 2 * SIZE(AO2), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X4, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	PREFETCH PREFETCHSIZE * 4 * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 4 * SIZE(AO2), X1
	MOVAPS 6 * SIZE(AO1), X2
	MOVAPS 6 * SIZE(AO2), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	MOVAPS   X2, X6
	UNPCKLPD X3, X2

	UNPCKHPD X1, X4
	UNPCKHPD X3, X6

	PREFETCHW (PREFETCHSIZE * 8 +  8) * SIZE(B)

	MOVAPS X0, -8 * SIZE(B)
	MOVAPS X4, -6 * SIZE(B)
	MOVAPS X2, -4 * SIZE(B)
	MOVAPS X6, -2 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-16 * SIZE, B

	DECQ I
	JG   L33

L34:
	TESTQ $4, MM
	JLE   L36

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO2), X1
	MOVAPS 2 * SIZE(AO1), X2
	MOVAPS 2 * SIZE(AO2), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	UNPCKHPD X1, X4

	MOVAPS   X2, X6
	UNPCKLPD X3, X2
	UNPCKHPD X3, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X4, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	SUBQ $-8 * SIZE, B

L36:
	TESTQ $2, MM
	JLE   L38

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 0 * SIZE(AO2), X1

	MOVAPS   X0, X2
	UNPCKLPD X1, X0
	UNPCKHPD X1, X2

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	SUBQ $-4 * SIZE, B

L38:
	TESTQ $1, MM
	JLE   L40

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)
	SUBQ   $-2 * SIZE, B

L40:
	TESTQ $1, N
	JLE   L999

	MOVQ A, AO1

	TESTQ $SIZE, A
	JNE   L45

	MOVQ MM, I
	SARQ $3, I
	JLE  L42

L41:
	PREFETCH PREFETCHSIZE * 8 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1), X1
	MOVAPS 4 * SIZE(AO1), X2
	MOVAPS 6 * SIZE(AO1), X3

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X3, -10 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	SUBQ $-8 * SIZE, B

	DECQ I
	JG   L41

L42:
	TESTQ $4, MM
	JLE   L43

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1), X1

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	SUBQ $-4 * SIZE, B

L43:
	TESTQ $2, MM
	JLE   L44

	MOVAPS 0 * SIZE(AO1), X0

	MOVAPS X0, -16 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	SUBQ $-2 * SIZE, B

L44:
	TESTQ $1, MM
	JLE   L999

	MOVSD 0 * SIZE(AO1), X0

	MOVLPD X0, -16 * SIZE(B)
	JMP    L999

L45:
	MOVAPS -1 * SIZE(AO1), X0

	MOVQ M, I
	SARQ $3, I
	JLE  L46

L46:
	PREFETCH PREFETCHSIZE * 8 * SIZE(AO1)

	MOVAPS 1 * SIZE(AO1), X1
	MOVAPS 3 * SIZE(AO1), X2
	MOVAPS 5 * SIZE(AO1), X3
	MOVAPS 7 * SIZE(AO1), X4

	SHUFPD $1, X1, X0
	SHUFPD $1, X2, X1
	SHUFPD $1, X3, X2
	SHUFPD $1, X4, X3

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X3, -10 * SIZE(B)

	MOVAPS X4, X0

	ADDQ $8 * SIZE, AO1
	SUBQ $-8 * SIZE, B

	DECQ I
	JG   L46

L47:
	TESTQ $4, M
	JLE   L48

	MOVAPS 1 * SIZE(AO1), X1
	MOVAPS 3 * SIZE(AO1), X2

	SHUFPD $1, X1, X0
	SHUFPD $1, X2, X1

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)

	MOVAPS X2, X0

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, B

L48:
	TESTQ $2, M
	JLE   L49

	MOVAPS 1 * SIZE(AO1), X1

	SHUFPD $1, X1, X0

	MOVAPS X0, -16 * SIZE(B)

	MOVAPS X1, X0

	ADDQ $2 * SIZE, AO1
	SUBQ $-2 * SIZE, B

L49:
	TESTQ $1, M
	JLE   L999

	SHUFPD $1, X0, X0

	MOVLPD X0, -16 * SIZE(B)
	JMP    L999

L50:
	MOVQ N, J
	SARQ $3, J
	JLE  L60

L51:
	MOVQ A, AO1
	LEAQ (A)(LDA*4), AO2
	LEAQ (A)(LDA*8), A

	TESTQ $SIZE, A
	JE    L52

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO1)(LDA*2), X2
	MOVSD 0 * SIZE(AO1)(LDA3*1), X3
	MOVSD 0 * SIZE(AO2), X4
	MOVSD 0 * SIZE(AO2)(LDA*1), X5
	MOVSD 0 * SIZE(AO2)(LDA*2), X6
	MOVSD 0 * SIZE(AO2)(LDA3*1), X7

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2
	UNPCKLPD X5, X4
	UNPCKLPD X7, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-8 * SIZE, B

L52:
	MOVAPS -1 * SIZE(AO1)(LDA*1), X9
	MOVAPS -1 * SIZE(AO1)(LDA3*1), X10
	MOVAPS -1 * SIZE(AO2)(LDA*1), X11
	MOVAPS -1 * SIZE(AO2)(LDA3*1), X12

	MOVQ MM, I
	SARQ $3, I
	JLE  L54

L53:
	PREFETCH PREFETCHSIZE * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO1)(LDA*2), X2
	MOVAPS 1 * SIZE(AO1)(LDA3*1), X3

	PREFETCH PREFETCHSIZE * SIZE(AO1)(LDA*1)

	MOVAPS 0 * SIZE(AO2), X4
	MOVAPS 1 * SIZE(AO2)(LDA*1), X5
	MOVAPS 0 * SIZE(AO2)(LDA*2), X6
	MOVAPS 1 * SIZE(AO2)(LDA3*1), X7

	MOVSD X0, X9
	MOVSD X2, X10
	MOVSD X4, X11
	MOVSD X6, X12

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X9, -16 * SIZE(B)
	MOVAPS X10, -14 * SIZE(B)
	MOVAPS X11, -12 * SIZE(B)
	MOVAPS X12, -10 * SIZE(B)

	SHUFPD $1, X1, X0
	SHUFPD $1, X3, X2
	SHUFPD $1, X5, X4
	SHUFPD $1, X7, X6

	PREFETCHW (PREFETCHSIZE * 8 +  8) * SIZE(B)

	MOVAPS X0, -8 * SIZE(B)
	MOVAPS X2, -6 * SIZE(B)
	MOVAPS X4, -4 * SIZE(B)
	MOVAPS X6, -2 * SIZE(B)

	PREFETCH PREFETCHSIZE * SIZE(AO1)(LDA*2)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 3 * SIZE(AO1)(LDA*1), X9
	MOVAPS 2 * SIZE(AO1)(LDA*2), X2
	MOVAPS 3 * SIZE(AO1)(LDA3*1), X10

	PREFETCH PREFETCHSIZE * SIZE(AO1)(LDA3*1)

	MOVAPS 2 * SIZE(AO2), X4
	MOVAPS 3 * SIZE(AO2)(LDA*1), X11
	MOVAPS 2 * SIZE(AO2)(LDA*2), X6
	MOVAPS 3 * SIZE(AO2)(LDA3*1), X12

	MOVSD X0, X1
	MOVSD X2, X3
	MOVSD X4, X5
	MOVSD X6, X7

	PREFETCHW (PREFETCHSIZE * 8 + 16) * SIZE(B)

	MOVAPS X1, 0 * SIZE(B)
	MOVAPS X3, 2 * SIZE(B)
	MOVAPS X5, 4 * SIZE(B)
	MOVAPS X7, 6 * SIZE(B)

	SHUFPD $1, X9, X0
	SHUFPD $1, X10, X2
	SHUFPD $1, X11, X4
	SHUFPD $1, X12, X6

	PREFETCHW (PREFETCHSIZE * 8 + 24) * SIZE(B)

	MOVAPS X0, 8 * SIZE(B)
	MOVAPS X2, 10 * SIZE(B)
	MOVAPS X4, 12 * SIZE(B)
	MOVAPS X6, 14 * SIZE(B)

	PREFETCH PREFETCHSIZE * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 5 * SIZE(AO1)(LDA*1), X1
	MOVAPS 4 * SIZE(AO1)(LDA*2), X2
	MOVAPS 5 * SIZE(AO1)(LDA3*1), X3

	PREFETCH PREFETCHSIZE * SIZE(AO2)(LDA*1)

	MOVAPS 4 * SIZE(AO2), X4
	MOVAPS 5 * SIZE(AO2)(LDA*1), X5
	MOVAPS 4 * SIZE(AO2)(LDA*2), X6
	MOVAPS 5 * SIZE(AO2)(LDA3*1), X7

	MOVSD X0, X9
	MOVSD X2, X10
	MOVSD X4, X11
	MOVSD X6, X12

	PREFETCHW (PREFETCHSIZE * 8 + 32) * SIZE(B)

	MOVAPS X9, 16 * SIZE(B)
	MOVAPS X10, 18 * SIZE(B)
	MOVAPS X11, 20 * SIZE(B)
	MOVAPS X12, 22 * SIZE(B)

	SHUFPD $1, X1, X0
	SHUFPD $1, X3, X2
	SHUFPD $1, X5, X4
	SHUFPD $1, X7, X6

	PREFETCHW (PREFETCHSIZE * 4 +  8) * SIZE(B)

	MOVAPS X0, 24 * SIZE(B)
	MOVAPS X2, 26 * SIZE(B)
	MOVAPS X4, 28 * SIZE(B)
	MOVAPS X6, 30 * SIZE(B)

	PREFETCH PREFETCHSIZE * SIZE(AO2)(LDA*2)

	MOVAPS 6 * SIZE(AO1), X0
	MOVAPS 7 * SIZE(AO1)(LDA*1), X9
	MOVAPS 6 * SIZE(AO1)(LDA*2), X2
	MOVAPS 7 * SIZE(AO1)(LDA3*1), X10

	PREFETCH PREFETCHSIZE * SIZE(AO2)(LDA3*1)

	MOVAPS 6 * SIZE(AO2), X4
	MOVAPS 7 * SIZE(AO2)(LDA*1), X11
	MOVAPS 6 * SIZE(AO2)(LDA*2), X6
	MOVAPS 7 * SIZE(AO2)(LDA3*1), X12

	MOVSD X0, X1
	MOVSD X2, X3
	MOVSD X4, X5
	MOVSD X6, X7

	PREFETCHW (PREFETCHSIZE * 8 + 40) * SIZE(B)

	MOVAPS X1, 32 * SIZE(B)
	MOVAPS X3, 34 * SIZE(B)
	MOVAPS X5, 36 * SIZE(B)
	MOVAPS X7, 38 * SIZE(B)

	SHUFPD $1, X9, X0
	SHUFPD $1, X10, X2
	SHUFPD $1, X11, X4
	SHUFPD $1, X12, X6

	PREFETCHW (PREFETCHSIZE * 8 + 48) * SIZE(B)
	MOVAPS    X0, 40 * SIZE(B)
	MOVAPS    X2, 42 * SIZE(B)
	MOVAPS    X4, 44 * SIZE(B)
	MOVAPS    X6, 46 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-64 * SIZE, B

	DECQ I
	JG   L53

L54:
	TESTQ $4, MM
	JLE   L56

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO1)(LDA*2), X2
	MOVAPS 1 * SIZE(AO1)(LDA3*1), X3
	MOVAPS 0 * SIZE(AO2), X4
	MOVAPS 1 * SIZE(AO2)(LDA*1), X5
	MOVAPS 0 * SIZE(AO2)(LDA*2), X6
	MOVAPS 1 * SIZE(AO2)(LDA3*1), X7

	MOVSD X0, X9
	MOVSD X2, X10
	MOVSD X4, X11
	MOVSD X6, X12

	MOVAPS X9, -16 * SIZE(B)
	MOVAPS X10, -14 * SIZE(B)
	MOVAPS X11, -12 * SIZE(B)
	MOVAPS X12, -10 * SIZE(B)

	SHUFPD $1, X1, X0
	SHUFPD $1, X3, X2
	SHUFPD $1, X5, X4
	SHUFPD $1, X7, X6

	MOVAPS X0, -8 * SIZE(B)
	MOVAPS X2, -6 * SIZE(B)
	MOVAPS X4, -4 * SIZE(B)
	MOVAPS X6, -2 * SIZE(B)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 3 * SIZE(AO1)(LDA*1), X9
	MOVAPS 2 * SIZE(AO1)(LDA*2), X2
	MOVAPS 3 * SIZE(AO1)(LDA3*1), X10
	MOVAPS 2 * SIZE(AO2), X4
	MOVAPS 3 * SIZE(AO2)(LDA*1), X11
	MOVAPS 2 * SIZE(AO2)(LDA*2), X6
	MOVAPS 3 * SIZE(AO2)(LDA3*1), X12

	MOVSD X0, X1
	MOVSD X2, X3
	MOVSD X4, X5
	MOVSD X6, X7

	MOVAPS X1, 0 * SIZE(B)
	MOVAPS X3, 2 * SIZE(B)
	MOVAPS X5, 4 * SIZE(B)
	MOVAPS X7, 6 * SIZE(B)

	SHUFPD $1, X9, X0
	SHUFPD $1, X10, X2
	SHUFPD $1, X11, X4
	SHUFPD $1, X12, X6

	MOVAPS X0, 8 * SIZE(B)
	MOVAPS X2, 10 * SIZE(B)
	MOVAPS X4, 12 * SIZE(B)
	MOVAPS X6, 14 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	SUBQ $-32 * SIZE, B

L56:
	TESTQ $2, MM
	JLE   L58

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO1)(LDA*2), X2
	MOVAPS 1 * SIZE(AO1)(LDA3*1), X3
	MOVAPS 0 * SIZE(AO2), X4
	MOVAPS 1 * SIZE(AO2)(LDA*1), X5
	MOVAPS 0 * SIZE(AO2)(LDA*2), X6
	MOVAPS 1 * SIZE(AO2)(LDA3*1), X7

	MOVSD X0, X9
	MOVSD X2, X10
	MOVSD X4, X11
	MOVSD X6, X12

	MOVAPS X9, -16 * SIZE(B)
	MOVAPS X10, -14 * SIZE(B)
	MOVAPS X11, -12 * SIZE(B)
	MOVAPS X12, -10 * SIZE(B)

	SHUFPD $1, X1, X0
	SHUFPD $1, X3, X2
	SHUFPD $1, X5, X4
	SHUFPD $1, X7, X6

	MOVAPS X0, -8 * SIZE(B)
	MOVAPS X2, -6 * SIZE(B)
	MOVAPS X4, -4 * SIZE(B)
	MOVAPS X6, -2 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	SUBQ $-16 * SIZE, B

L58:
	TESTQ $1, MM
	JLE   L59

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO1)(LDA*2), X2
	MOVSD 0 * SIZE(AO1)(LDA3*1), X3
	MOVSD 0 * SIZE(AO2), X4
	MOVSD 0 * SIZE(AO2)(LDA*1), X5
	MOVSD 0 * SIZE(AO2)(LDA*2), X6
	MOVSD 0 * SIZE(AO2)(LDA3*1), X7

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2
	UNPCKLPD X5, X4
	UNPCKLPD X7, X6

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	SUBQ $-8 * SIZE, B

L59:
	DECQ J
	JG   L51

L60:
	TESTQ $4, N
	JLE   L70

	MOVQ A, AO1
	LEAQ (A)(LDA*2), AO2
	LEAQ (A)(LDA*4), A

	TESTQ $SIZE, A
	JE    L62

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO2), X2
	MOVSD 0 * SIZE(AO2)(LDA*1), X3

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-4 * SIZE, B

L62:
	MOVAPS -1 * SIZE(AO1)(LDA*1), X5
	MOVAPS -1 * SIZE(AO2)(LDA*1), X7

	MOVQ MM, I
	SARQ $3, I
	JLE  L64

L63:
	PREFETCH PREFETCHSIZE * 2 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO2), X2
	MOVAPS 1 * SIZE(AO2)(LDA*1), X3

	MOVSD  X0, X5
	MOVSD  X2, X7
	SHUFPD $1, X1, X0
	SHUFPD $1, X3, X2

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X7, -14 * SIZE(B)
	MOVAPS X0, -12 * SIZE(B)
	MOVAPS X2, -10 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO1)(LDA*1)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 3 * SIZE(AO1)(LDA*1), X5
	MOVAPS 2 * SIZE(AO2), X2
	MOVAPS 3 * SIZE(AO2)(LDA*1), X7

	MOVSD  X0, X1
	MOVSD  X2, X3
	SHUFPD $1, X5, X0
	SHUFPD $1, X7, X2

	PREFETCHW (PREFETCHSIZE * 8 +  8) * SIZE(B)

	MOVAPS X1, -8 * SIZE(B)
	MOVAPS X3, -6 * SIZE(B)
	MOVAPS X0, -4 * SIZE(B)
	MOVAPS X2, -2 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 5 * SIZE(AO1)(LDA*1), X1
	MOVAPS 4 * SIZE(AO2), X2
	MOVAPS 5 * SIZE(AO2)(LDA*1), X3

	MOVSD  X0, X5
	MOVSD  X2, X7
	SHUFPD $1, X1, X0
	SHUFPD $1, X3, X2

	PREFETCHW (PREFETCHSIZE * 8 + 16) * SIZE(B)

	MOVAPS X5, 0 * SIZE(B)
	MOVAPS X7, 2 * SIZE(B)
	MOVAPS X0, 4 * SIZE(B)
	MOVAPS X2, 6 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO2)(LDA*1)

	MOVAPS 6 * SIZE(AO1), X0
	MOVAPS 7 * SIZE(AO1)(LDA*1), X5
	MOVAPS 6 * SIZE(AO2), X2
	MOVAPS 7 * SIZE(AO2)(LDA*1), X7

	MOVSD  X0, X1
	MOVSD  X2, X3
	SHUFPD $1, X5, X0
	SHUFPD $1, X7, X2

	PREFETCHW (PREFETCHSIZE * 8 + 24) * SIZE(B)

	MOVAPS X1, 8 * SIZE(B)
	MOVAPS X3, 10 * SIZE(B)
	MOVAPS X0, 12 * SIZE(B)
	MOVAPS X2, 14 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-32 * SIZE, B

	DECQ I
	JG   L63

L64:
	TESTQ $4, MM
	JLE   L66

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO2), X2
	MOVAPS 1 * SIZE(AO2)(LDA*1), X3

	MOVSD  X0, X5
	SHUFPD $1, X1, X0
	MOVSD  X2, X7
	SHUFPD $1, X3, X2

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X7, -14 * SIZE(B)
	MOVAPS X0, -12 * SIZE(B)
	MOVAPS X2, -10 * SIZE(B)

	MOVAPS 2 * SIZE(AO1), X0
	MOVAPS 3 * SIZE(AO1)(LDA*1), X5
	MOVAPS 2 * SIZE(AO2), X2
	MOVAPS 3 * SIZE(AO2)(LDA*1), X7

	MOVSD  X0, X1
	SHUFPD $1, X5, X0
	MOVSD  X2, X3
	SHUFPD $1, X7, X2

	MOVAPS X1, -8 * SIZE(B)
	MOVAPS X3, -6 * SIZE(B)
	MOVAPS X0, -4 * SIZE(B)
	MOVAPS X2, -2 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	SUBQ $-16 * SIZE, B

L66:
	TESTQ $2, MM
	JLE   L68

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO1)(LDA*1), X1
	MOVAPS 0 * SIZE(AO2), X2
	MOVAPS 1 * SIZE(AO2)(LDA*1), X3

	MOVSD  X0, X5
	MOVSD  X2, X7
	SHUFPD $1, X1, X0
	SHUFPD $1, X3, X2

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X7, -14 * SIZE(B)
	MOVAPS X0, -12 * SIZE(B)
	MOVAPS X2, -10 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	SUBQ $-8 * SIZE, B

L68:
	TESTQ $1, MM
	JLE   L70

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO1)(LDA*1), X1
	MOVSD 0 * SIZE(AO2), X2
	MOVSD 0 * SIZE(AO2)(LDA*1), X3

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	SUBQ   $-4 * SIZE, B

L70:
	TESTQ $2, N
	JLE   L80

	MOVQ A, AO1
	LEAQ (A)(LDA*1), AO2
	LEAQ (A)(LDA*2), A

	TESTQ $SIZE, A
	JE    L72

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-2 * SIZE, B

L72:
	MOVAPS -1 * SIZE(AO2), X5

	MOVQ MM, I
	SARQ $3, I
	JLE  L74

L73:
	PREFETCH PREFETCHSIZE * 4 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO2), X1
	MOVAPS 2 * SIZE(AO1), X2
	MOVAPS 3 * SIZE(AO2), X3

	MOVSD  X0, X5
	SHUFPD $1, X1, X0
	MOVSD  X2, X1
	SHUFPD $1, X3, X2

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X0, -14 * SIZE(B)
	MOVAPS X1, -12 * SIZE(B)
	MOVAPS X2, -10 * SIZE(B)

	PREFETCH PREFETCHSIZE * 4 * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 5 * SIZE(AO2), X1
	MOVAPS 6 * SIZE(AO1), X2
	MOVAPS 7 * SIZE(AO2), X5

	MOVSD  X0, X3
	SHUFPD $1, X1, X0
	MOVSD  X2, X1
	SHUFPD $1, X5, X2

	PREFETCHW (PREFETCHSIZE * 8 +  8) * SIZE(B)

	MOVAPS X3, -8 * SIZE(B)
	MOVAPS X0, -6 * SIZE(B)
	MOVAPS X1, -4 * SIZE(B)
	MOVAPS X2, -2 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-16 * SIZE, B

	DECQ I
	JG   L73

L74:
	TESTQ $4, MM
	JLE   L76

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO2), X1
	MOVAPS 2 * SIZE(AO1), X2
	MOVAPS 3 * SIZE(AO2), X3

	MOVSD  X0, X5
	SHUFPD $1, X1, X0
	MOVSD  X2, X1
	SHUFPD $1, X3, X2

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X0, -14 * SIZE(B)
	MOVAPS X1, -12 * SIZE(B)
	MOVAPS X2, -10 * SIZE(B)

	MOVAPS X3, X5

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, AO2
	SUBQ $-8 * SIZE, B

L76:
	TESTQ $2, MM
	JLE   L78

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO2), X1

	MOVSD  X0, X5
	SHUFPD $1, X1, X0

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X0, -14 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	SUBQ $-4 * SIZE, B

L78:
	TESTQ $1, MM
	JLE   L80

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)
	SUBQ   $-2 * SIZE, B

L80:
	TESTQ $1, N
	JLE   L999

	MOVQ A, AO1

	TESTQ $SIZE, A
	JNE   L85

	MOVQ MM, I
	SARQ $3, I
	JLE  L82

L81:
	PREFETCH PREFETCHSIZE * 8 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1), X2
	MOVAPS 4 * SIZE(AO1), X4
	MOVAPS 6 * SIZE(AO1), X6

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)
	MOVAPS X4, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	SUBQ $-8 * SIZE, B

	DECQ I
	JG   L81

L82:
	TESTQ $4, MM
	JLE   L83

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1), X2

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X2, -14 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	SUBQ $-4 * SIZE, B

L83:
	TESTQ $2, MM
	JLE   L84

	MOVAPS 0 * SIZE(AO1), X0

	MOVAPS X0, -16 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	SUBQ $-2 * SIZE, B

L84:
	TESTQ $1, MM
	JLE   L999

	MOVSD 0 * SIZE(AO1), X0

	MOVLPD X0, -16 * SIZE(B)
	JMP    L999

L85:
	MOVAPS -1 * SIZE(AO1), X0

	MOVQ M, I
	SARQ $3, I
	JLE  L86

L86:
	PREFETCH PREFETCHSIZE * 8 * SIZE(AO1)

	MOVAPS 1 * SIZE(AO1), X1
	MOVAPS 3 * SIZE(AO1), X2
	MOVAPS 5 * SIZE(AO1), X3
	MOVAPS 7 * SIZE(AO1), X4

	SHUFPD $1, X1, X0
	SHUFPD $1, X2, X1
	SHUFPD $1, X3, X2
	SHUFPD $1, X4, X3

	PREFETCHW (PREFETCHSIZE * 8 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X3, -10 * SIZE(B)

	MOVAPS X4, X0

	ADDQ $8 * SIZE, AO1
	SUBQ $-8 * SIZE, B

	DECQ I
	JG   L86

L87:
	TESTQ $4, M
	JLE   L88

	MOVAPS 1 * SIZE(AO1), X1
	MOVAPS 3 * SIZE(AO1), X2

	SHUFPD $1, X1, X0
	SHUFPD $1, X2, X1

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)

	MOVAPS X2, X0

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, B

L88:
	TESTQ $2, M
	JLE   L89

	MOVAPS 1 * SIZE(AO1), X1

	SHUFPD $1, X1, X0

	MOVAPS X0, -16 * SIZE(B)

	MOVAPS X1, X0

	ADDQ $2 * SIZE, AO1
	SUBQ $-2 * SIZE, B

L89:
	TESTQ $1, M
	JLE   L999

	SHUFPD $1, X0, X0

	MOVLPD X0, -16 * SIZE(B)

L999:
	RET