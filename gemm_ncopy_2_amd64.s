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

#define PREFETCHSIZE	16
#define PREFETCH	PREFETCHT0
#define PREFETCHW	PREFETCHT0

#define M	DI
#define N	SI
#define A	DX
#define LDA	CX
#define B	R8

#define I	R9

#define J	R10
#define AO1	R11
#define AO2	R12
#define MM	R13

// func dgemm_ncopy_2(m, n int, a []float64, lda int, b []float64)
TEXT ·dgemm_ncopy_2(SB), NOSPLIT, $0

	MOVQ m+0(FP), M
	MOVQ n+8(FP), N
	MOVQ a_base+16(FP), A
	MOVQ lda+40(FP), LDA
	MOVQ b_base+48(FP), B

	IMULQ $SIZE, LDA
	SUBQ  $-16 * SIZE, B

	MOVQ    M, MM
	LEAQ    -1(M), AX
	TESTQ   $SIZE, A
	CMOVQNE AX, MM

	TESTQ $SIZE, LDA
	JNE   L50

	MOVQ N, J
	SARQ $1, J
	JLE  L30

L21:
	MOVQ A, AO1
	LEAQ (A)(LDA*1), AO2
	LEAQ (A)(LDA*2), A

	TESTQ $SIZE, A
	JE    L22

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-2 * SIZE, B

L22:
	MOVQ MM, I
	SARQ $3, I
	JLE  L24

L23:
	PREFETCH PREFETCHSIZE * 2 * SIZE(AO1)

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

	PREFETCHW (PREFETCHSIZE * 4 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X4, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X6, -10 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 4 * SIZE(AO2), X1
	MOVAPS 6 * SIZE(AO1), X2
	MOVAPS 6 * SIZE(AO2), X3

	MOVAPS   X0, X4
	UNPCKLPD X1, X0
	UNPCKHPD X1, X4

	MOVAPS   X2, X6
	UNPCKLPD X3, X2
	UNPCKHPD X3, X6

	PREFETCHW (PREFETCHSIZE * 4 +  8) * SIZE(B)

	MOVAPS X0, -8 * SIZE(B)
	MOVAPS X4, -6 * SIZE(B)
	MOVAPS X2, -4 * SIZE(B)
	MOVAPS X6, -2 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-16 * SIZE, B

	DECQ I
	JG   L23

L24:
	TESTQ $4, MM
	JLE   L26

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

L26:
	TESTQ $2, MM
	JLE   L28

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

L28:
	TESTQ $1, MM
	JLE   L29

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)
	SUBQ   $-2 * SIZE, B

L29:
	DECQ J
	JG   L21

L30:
	TESTQ $1, N
	JLE   L999

L30x:
	MOVQ A, AO1

	TESTQ $SIZE, A
	JNE   L35

	MOVQ M, I
	SARQ $3, I
	JLE  L32

L31:

	PREFETCH PREFETCHSIZE * 4 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1), X1
	MOVAPS 4 * SIZE(AO1), X2
	MOVAPS 6 * SIZE(AO1), X3

	PREFETCHW (PREFETCHSIZE * 4 +  0) * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X3, -10 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, B

	DECQ I
	JG   L31

L32:
	TESTQ $4, M
	JLE   L33

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 2 * SIZE(AO1), X1

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)

	ADDQ $4 * SIZE, AO1
	SUBQ $-4 * SIZE, B

L33:
	TESTQ $2, M
	JLE   L34

	MOVAPS 0 * SIZE(AO1), X0

	MOVAPS X0, -16 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	SUBQ $-2 * SIZE, B

L34:
	TESTQ $1, M
	JLE   L999

	MOVSD  0 * SIZE(AO1), X0
	MOVLPD X0, -16 * SIZE(B)
	JMP    L999

L35:
	MOVAPS -1 * SIZE(AO1), X0

	MOVQ M, I
	SARQ $3, I
	JLE  L37

L36:

	PREFETCH PREFETCHSIZE * 4 * SIZE(AO1)

	MOVAPS 1 * SIZE(AO1), X1
	MOVAPS 3 * SIZE(AO1), X2
	MOVAPS 5 * SIZE(AO1), X3
	MOVAPS 7 * SIZE(AO1), X4

	SHUFPD $1, X1, X0
	SHUFPD $1, X2, X1
	SHUFPD $1, X3, X2
	SHUFPD $1, X4, X3

	PREFETCHW PREFETCHSIZE * 4 * SIZE(B)

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)
	MOVAPS X2, -12 * SIZE(B)
	MOVAPS X3, -10 * SIZE(B)

	MOVAPS X4, X0

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, B

	DECQ I
	JG   L36

L37:
	TESTQ $4, M
	JLE   L38

	MOVAPS 1 * SIZE(AO1), X1
	MOVAPS 3 * SIZE(AO1), X2

	SHUFPD $1, X1, X0
	SHUFPD $1, X2, X1

	MOVAPS X0, -16 * SIZE(B)
	MOVAPS X1, -14 * SIZE(B)

	MOVAPS X2, X0

	ADDQ $4 * SIZE, AO1
	ADDQ $4 * SIZE, B

L38:
	TESTQ $2, M
	JLE   L39

	MOVAPS 1 * SIZE(AO1), X1

	SHUFPD $1, X1, X0

	MOVAPS X0, -16 * SIZE(B)

	MOVAPS X1, X0

	ADDQ $2 * SIZE, AO1
	SUBQ $-2 * SIZE, B

L39:
	TESTQ $1, M
	JLE   L999

	MOVHPD X0, -16 * SIZE(B)
	JMP    L999

L50:
	MOVQ N, J
	SARQ $1, J
	JLE  L30

L61:
	MOVQ A, AO1
	LEAQ (A)(LDA*1), AO2
	LEAQ (A)(LDA*2), A

	TESTQ $SIZE, A
	JE    L62

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)

	ADDQ $1 * SIZE, AO1
	ADDQ $1 * SIZE, AO2
	SUBQ $-2 * SIZE, B

L62:
	MOVAPS -1 * SIZE(AO2), X5

	MOVQ MM, I
	SARQ $3, I
	JLE  L64

L63:

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO1)

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO2), X1
	MOVAPS 2 * SIZE(AO1), X2
	MOVAPS 3 * SIZE(AO2), X3

	MOVSD  X0, X5
	SHUFPD $1, X1, X0
	MOVSD  X2, X1
	SHUFPD $1, X3, X2

	PREFETCHW (PREFETCHSIZE * 4 +  0) * SIZE(B)

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X0, -14 * SIZE(B)
	MOVAPS X1, -12 * SIZE(B)
	MOVAPS X2, -10 * SIZE(B)

	PREFETCH PREFETCHSIZE * 2 * SIZE(AO2)

	MOVAPS 4 * SIZE(AO1), X0
	MOVAPS 5 * SIZE(AO2), X1
	MOVAPS 6 * SIZE(AO1), X2
	MOVAPS 7 * SIZE(AO2), X5

	MOVSD  X0, X3
	SHUFPD $1, X1, X0
	MOVSD  X2, X1
	SHUFPD $1, X5, X2

	PREFETCHW (PREFETCHSIZE * 4 +  0) * SIZE(B)

	MOVAPS X3, -8 * SIZE(B)
	MOVAPS X0, -6 * SIZE(B)
	MOVAPS X1, -4 * SIZE(B)
	MOVAPS X2, -2 * SIZE(B)

	ADDQ $8 * SIZE, AO1
	ADDQ $8 * SIZE, AO2
	SUBQ $-16 * SIZE, B

	DECQ I
	JG   L63

L64:
	TESTQ $4, MM
	JLE   L66

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

L66:
	TESTQ $2, MM
	JLE   L68

	MOVAPS 0 * SIZE(AO1), X0
	MOVAPS 1 * SIZE(AO2), X1

	MOVSD  X0, X5
	SHUFPD $1, X1, X0

	MOVAPS X5, -16 * SIZE(B)
	MOVAPS X0, -14 * SIZE(B)

	ADDQ $2 * SIZE, AO1
	ADDQ $2 * SIZE, AO2
	SUBQ $-4 * SIZE, B

L68:
	TESTQ $1, MM
	JLE   L69

	MOVSD 0 * SIZE(AO1), X0
	MOVSD 0 * SIZE(AO2), X1

	UNPCKLPD X1, X0

	MOVAPS X0, -16 * SIZE(B)
	SUBQ   $-2 * SIZE, B

L69:
	DECQ J
	JG   L61

	TESTQ $1, N
	JNE   L30

L999:
	RET
