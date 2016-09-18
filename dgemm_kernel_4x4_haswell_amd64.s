/* ********************************************************************************
COPYRIGHT (c) 2013, The OpenBLAS Project
ALL rights reserved.
REDISTRIBUTION and use in source and binary forms, with or without
MODIFICATION, are permitted provided that the following conditions are
MET:
1. Redistributions of source code must retain the above copyright
NOTICE, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
NOTICE, this list of conditions and the following disclaimer in
THE documentation and/or other materials provided with the
DISTRIBUTION.
3. Neither the name of the OpenBLAS project nor the names of
ITS contributors may be used to endorse or promote products
DERIVED from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************/
/* ********************************************************************
 * 2013/10/28 Saar
 *        BLASTEST               : OK
 *        CTEST                  : OK
 *        TEST                   : OK

 *
 *
 * 2013/10/27 Saar
 * Parameter:
 *       DGEMM_DEFAULT_UNROLL_N  4
 *       DGEMM_DEFAULT_UNROLL_M  4
 *       DGEMM_DEFAULT_P         512
 *       DGEMM_DEFAULT_Q         256
 *	A_PR1			512
 *	B_PR1			512
 *
 *
 * Performance at 9216x9216x9216:
 *       1 thread:       53.3 GFLOPS (MKL:  54)
 *       2 threads:     100.0 GFLOPS (MKL:  97)
 *       3 threads:     147.0 GFLOPS (MKL: 133)
 *       4 threads:     184.0 GFLOPS (MKL: 170)
******************************************************************** */


#define VMOVAPS VMOVDQU
#define VMOVUPS VMOVDQU
#define VMOVSD 	MOVSD
#define VXORPD 	VPXOR

#define DEBGR(reg) \
 	MOVQ reg, ret+112(FP); \
 	RET

// #define DEBG(reg) \
//  	MOVQ reg, (C);\
//  	MOVQ  C_base+80(FP), C ;\
//  	RET
// #define DEBG2(reg1,reg2) \
//  	MOVQ reg1, ret+112(FP); \
//  	MOVQ  C_base+80(FP), C ;\
//  	VMOVUPS reg2, (C);\
//  	RET


#define PREFETCHSIZE	16
#define PREFETCH	PREFETCHT0
#define PREFETCHW	PREFETCHT0

#define SIZE    8
#define BASE_SHIFT 3
#define ZBASE_SHIFT 4

#define OLD_N	DI
#define OLD_M	m+0(FP)
#define N	R13
#define J	R14
#define OLD_K	DX

#define A	R8
#define B	B_base+56(FP)
#define C	R9
#define LDC	R10

#define I	R11
#define AO	SI
#define BO	DI
#define CO1	R15
#define K	R12

#define AO1	DI
#define AO2	R15
#define AO3	BP

#define L_BUFFER_SIZE 512*8*12+4096

#define Mdiv12	 24(BX)
#define Mmod12	 32(BX)
#define M	 m+0(FP)
#define ALPHA	 48(BX)
#define BUFFER1	128(BX)

#define A_PR1	512
#define B_PR1	512


#include "dgemm_kernel_4x4_haswell_macros.h"

// func gemm_kernel_4x4(m, n, k int, alpha float64, a,b,c []float64, ldc int)
TEXT ·gemm_kernel_4x4(SB), 0, $54272-120
	MOVQ n+8(FP), OLD_N
	MOVQ k+16(FP), OLD_K

	MOVQ $0, ret+112(FP)

	CMPQ OLD_N, $0
	JE   L999

	CMPQ OLD_M, $0
	JE   L999

	CMPQ OLD_K, $0
	JE   L999

	VZEROUPPER

	MOVQ  A_base+32(FP), A
	MOVQ  C_base+80(FP), C
	MOVQ  ldc+104(FP), LDC

	LEAQ base-0(SP), BX
	SUBQ $128 + L_BUFFER_SIZE, BX
	MOVQ BX, AX
	ANDQ $-4096, BX               // align stack

	MOVQ OLD_N, N
	MOVQ OLD_K, K

	MOVQ alpha+24(FP), X0
	MOVQ X0, ALPHA

	SALQ $BASE_SHIFT, LDC

	MOVQ OLD_M, AX
	XORQ DX, DX
	MOVQ $12, DI
	DIVQ DI         // M / 12
	MOVQ AX, Mdiv12 // M / 12
	MOVQ DX, Mmod12 // M % 12

	MOVQ Mdiv12, J
	CMPQ J, $ 0
	JE   L4_0

L12_01:
	// copy to sub buffer
	MOVQ K, AX
	SALQ $2, AX              // K * 4 ; read 2 values
	MOVQ A, AO1
	LEAQ (A)(AX*SIZE), AO2   // next offset to AO2
	LEAQ (AO2)(AX*SIZE), AO3 // next offset to AO2

	LEAQ BUFFER1, AO // first buffer to AO
	MOVQ K, AX
	SARQ $1, AX      // K / 2
	JZ   L12_01a_2

L12_01a_1:

	PREFETCHT0 512(AO1)
	PREFETCHT0 512(AO2)
	PREFETCHT0 512(AO3)
	PREFETCHW  512(AO)

	VMOVUPS 0 * SIZE(AO1), Y1
	VMOVUPS 4 * SIZE(AO1), Y5
	VMOVUPS 0 * SIZE(AO2), Y2
	VMOVUPS 4 * SIZE(AO2), Y6
	VMOVUPS 0 * SIZE(AO3), Y3
	VMOVUPS 4 * SIZE(AO3), Y7

	VMOVUPS Y1, 0 * SIZE(AO)
	VMOVUPS Y2, 4 * SIZE(AO)
	VMOVUPS Y3, 8 * SIZE(AO)

	VMOVUPS Y5, 12 * SIZE(AO)
	VMOVUPS Y6, 16 * SIZE(AO)
	VMOVUPS Y7, 20 * SIZE(AO)

	ADDQ $ 8 * SIZE, AO1
	ADDQ $ 8 * SIZE, AO2
	ADDQ $ 8 * SIZE, AO3
	ADDQ $ 24 *SIZE, AO

	DECQ AX
	JNZ  L12_01a_1

L12_01a_2:

	MOVQ K, AX
	ANDQ $1, AX  // K % 2
	JZ   L12_03c

L12_02b:

	VMOVUPS 0 * SIZE(AO1), Y1
	VMOVUPS 0 * SIZE(AO2), Y2
	VMOVUPS 0 * SIZE(AO3), Y3
	VMOVUPS Y1, 0 * SIZE(AO)
	VMOVUPS Y2, 4 * SIZE(AO)
	VMOVUPS Y3, 8 * SIZE(AO)
	ADDQ    $ 4*SIZE, AO1
	ADDQ    $ 4*SIZE, AO2
	ADDQ    $ 4*SIZE, AO3
	ADDQ    $ 12*SIZE, AO
	DECQ    AX
	JNZ     L12_02b

L12_03c:

	MOVQ AO3, A // next offset of A

L12_10:
	MOVQ C, CO1
	LEAQ (C)(LDC*8), C
	LEAQ (C)(LDC*4), C // c += 12 * ldc

	MOVQ B, BO          // boffset = B
	ADDQ $16 * SIZE, BO

	MOVQ N, I
	SARQ $2, I  // I = N / 4
	JE   L12_20

L12_11:
	LEAQ BUFFER1, AO    // first buffer to AO
	ADDQ $12 * SIZE, AO

	MOVQ K, AX

	SARQ $3, AX // K / 8
	CMPQ AX, $2

	JL L12_13
	KERNEL4x12_I
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_M2

	KERNEL4x12_M1
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_M2

	SUBQ $2, AX
	JE   L12_12a

L12_12:

	KERNEL4x12_M1
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_M2

	KERNEL4x12_M1
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_M2

	DECQ AX
	JNE L12_12

L12_12a:

	KERNEL4x12_M1
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_M2

	KERNEL4x12_M1
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_E

	JMP L12_16

L12_13:

	TESTQ $1, AX
	JZ   L12_14

	KERNEL4x12_I
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_M2

	KERNEL4x12_M1
	KERNEL4x12_M2
	KERNEL4x12_M1
	KERNEL4x12_E

	JMP L12_16

L12_14:

	INIT4x12

L12_16:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L12_19

L12_17:

	KERNEL4x12_SUB

	DECQ AX
	JNE L12_17

L12_19:

	SAVE4x12
	DECQ I      // i --
	JNE  L12_11

/* *************************************************************************
 * Rest of N
************************************************************************** */
L12_20:
	// Test rest of N

	TESTQ $3, N
	JZ    L12_100 // to next 16 lines of M

L12_30:
	TESTQ $2, N
	JZ    L12_40

L12_31:
	LEAQ BUFFER1, AO    // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT2x12

	MOVQ K, AX

	SARQ $3, AX
	JE   L12_36

L12_32:

	KERNEL2x12_SUB
	KERNEL2x12_SUB
	KERNEL2x12_SUB
	KERNEL2x12_SUB

	KERNEL2x12_SUB
	KERNEL2x12_SUB
	KERNEL2x12_SUB
	KERNEL2x12_SUB

	DECQ AX
	JNE L12_32

L12_36:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L12_39

L12_37:

	KERNEL2x12_SUB

	DECQ AX
	JNE L12_37

L12_39:

	SAVE2x12

L12_40:
	TESTQ $1, N
	JZ    L12_100 // to next 3 lines of M

L12_41:
	LEAQ BUFFER1, AO    // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT1x12

	MOVQ K, AX

	SARQ $3, AX
	JE   L12_46

L12_42:

	KERNEL1x12_SUB
	KERNEL1x12_SUB
	KERNEL1x12_SUB
	KERNEL1x12_SUB

	KERNEL1x12_SUB
	KERNEL1x12_SUB
	KERNEL1x12_SUB
	KERNEL1x12_SUB

	DECQ AX
	JNE L12_42

L12_46:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L12_49

L12_47:

	KERNEL1x12_SUB

	DECQ AX
	JNE L12_47

L12_49:

	SAVE1x12

L12_100:
	
	DECQ J      // j --
	JG   L12_01

L4_0:

	CMPQ Mmod12, $ 0 // M % 12 == 0
	JE   L999


	MOVQ Mmod12, J
	SARQ $2, J     // j = j / 4
	JE   L2_0

L4_10:
	MOVQ C, CO1
	LEAQ (C)(LDC*4), C // c += 4 * ldc

	MOVQ B, BO          // aoffset = a
	ADDQ $16 * SIZE, BO


	MOVQ N, I
	SARQ $2, I // i = m / 4
	JE   L4_20

L4_11:
	MOVQ A, AO
	ADDQ $12 * SIZE, AO

	MOVQ K, AX

	SARQ $3, AX // K / 8
	CMPQ AX, $2
	JL   L4_13

	KERNEL4x4_I
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	SUBQ $2, AX
	JE   L4_12a

L4_12:

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	DECQ AX
	JNE L4_12

L4_12a:

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_E

	JMP L4_16

L4_13:

	TESTQ $1, AX
	JZ   L4_14

	KERNEL4x4_I
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_E

	JMP L4_16

L4_14:

	INIT4x4

L4_16:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L4_19

L4_17:

	KERNEL4x4_SUB

	DECQ AX
	JNE L4_17

L4_19:

	SAVE4x4

	DECQ I     // i --
	JG   L4_11

/* *************************************************************************
 * Rest of N
************************************************************************** */
L4_20:
	// Test rest of N

	TESTQ $3, N
	JZ    L4_100 // to next 16 lines of M

L4_30:
	TESTQ $2, N
	JZ    L4_40

L4_31:
	MOVQ A, AO          // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT2x4

	MOVQ K, AX

	SARQ $3, AX
	JE   L4_36

L4_32:

	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB

	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB
	KERNEL2x4_SUB

	DECQ AX
	JNE L4_32

L4_36:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L4_39

L4_37:

	KERNEL2x4_SUB

	DECQ AX
	JNE L4_37

L4_39:

	SAVE2x4

L4_40:
	TESTQ $1, N
	JZ    L4_100 // to next 3 lines of M

L4_41:
	MOVQ A, AO          // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT1x4

	MOVQ K, AX

	SARQ $3, AX
	JE   L4_46

L4_42:

	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB

	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB
	KERNEL1x4_SUB

	DECQ AX
	JNE L4_42

L4_46:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L4_49

L4_47:

	KERNEL1x4_SUB

	DECQ AX
	JNE L4_47

L4_49:

	SAVE1x4

L4_100:

	MOVQ K, AX
	SALQ $2, AX          // * 4
	LEAQ (A)(AX*SIZE), A
	DECQ J               // j --
	JG   L4_10

	// *************************************************************************************************************

L2_0:

	MOVQ  Mmod12, J
	TESTQ $2, J
	JE    L1_0

L2_10:
	MOVQ C, CO1
	LEAQ (C)(LDC*2), C // c += 2 * ldc

	MOVQ B, BO          // aoffset = a
	ADDQ $16 * SIZE, BO

	MOVQ N, I
	SARQ $2, I // i = m / 4
	JE   L2_20

L2_11:
	MOVQ A, AO
	ADDQ $12 * SIZE, AO

	INIT4x2

	MOVQ K, AX
	SARQ $3, AX // K / 8

	JE L2_16

L2_12:

	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB

	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB
	KERNEL4x2_SUB

	DECQ AX
	JNE L2_12

L2_16:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L2_19

L2_17:

	KERNEL4x2_SUB

	DECQ AX
	JNE L2_17

L2_19:

	SAVE4x2

	DECQ I     // i --
	JG   L2_11

/* *************************************************************************
 * Rest of N
************************************************************************** */
L2_20:
	// Test rest of N

	TESTQ $3, N
	JZ    L2_100 // to next 16 lines of M

L2_30:
	TESTQ $2, N
	JZ    L2_40

L2_31:
	MOVQ A, AO          // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT2x2

	MOVQ K, AX

	SARQ $3, AX
	JE   L2_36

L2_32:

	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB

	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB
	KERNEL2x2_SUB

	DECQ AX
	JNE L2_32

L2_36:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L2_39

L2_37:

	KERNEL2x2_SUB

	DECQ AX
	JNE L2_37

L2_39:

	SAVE2x2

L2_40:
	TESTQ $1, N
	JZ    L2_100 // to next 3 lines of M

L2_41:
	MOVQ A, AO          // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT1x2

	MOVQ K, AX

	SARQ $3, AX
	JE   L2_46

L2_42:

	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB

	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB
	KERNEL1x2_SUB

	DECQ AX
	JNE L2_42

L2_46:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L2_49

L2_47:

	KERNEL1x2_SUB


	DECQ AX
	JNE L2_47

L2_49:

	SAVE1x2

L2_100:

	MOVQ K, AX
	SALQ $1, AX          // * 2
	LEAQ (A)(AX*SIZE), A

	// *************************************************************************************************************

L1_0:

	MOVQ  Mmod12, J
	TESTQ $1, J
	JE    L999

L1_10:

	MOVQ C, CO1
	LEAQ (C)(LDC*1), C // c += 1 * ldc
	

	MOVQ B, BO          // aoffset = a
	ADDQ $16 * SIZE, BO

	MOVQ N, I
	SARQ $2, I // i = m / 4
	JE   L1_20

L1_11:
	MOVQ A, AO
	ADDQ $12 * SIZE, AO

	INIT4x1

	MOVQ K, AX

	SARQ $3, AX // K / 8
	JE   L1_16

L1_12:

	KERNEL4x1

	DECQ AX
	JNE L1_12

L1_16:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L1_19

L1_17:

	KERNEL4x1_SUB

	DECQ AX
	JNE L1_17

L1_19:

	SAVE4x1

	DECQ I     // i --
	JG   L1_11

/* *************************************************************************
 * Rest of N
************************************************************************** */
L1_20:
	// Test rest of N

	TESTQ $3, N
	JZ    L999

L1_30:
	TESTQ $2, N
	JZ    L1_40

L1_31:
	MOVQ A, AO          // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT2x1

	MOVQ K, AX

	SARQ $3, AX
	JE   L1_36

L1_32:

	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB

	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB
	KERNEL2x1_SUB

	DECQ AX
	JNE L1_32

L1_36:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 1)
	JE   L1_39

L1_37:

	KERNEL2x1_SUB

	DECQ AX
	JNE L1_37

L1_39:
	SAVE2x1

L1_40:
	TESTQ $1, N
	JZ    L999 // to next 3 lines of M

L1_41:
	MOVQ A, AO          // first buffer to AO
	ADDQ $12 * SIZE, AO

	INIT1x1

	MOVQ K, AX

	SARQ $3, AX
	JE   L1_46

L1_42:
	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB

	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB
	KERNEL1x1_SUB

	DECQ AX
	JNE L1_42

L1_46:
	MOVQ K, AX

	ANDQ $7, AX // if (k & 7)
	JE   L1_49

L1_47:

	KERNEL1x1_SUB

	DECQ AX
	JNE L1_47

L1_49:

	SAVE1x1

L1_100:
L999:
	VZEROUPPER
	RET

LRET:

	LEAQ BUFFER1, BO
	
LRET_LOOP:

	VMOVUPS (BO), Y1
	VMOVUPS Y1, (C)
	ADDQ $4*SIZE, BO
	ADDQ $4*SIZE, A
	ADDQ $4*SIZE, C
	CMPQ A, AO3
	JL LRET_LOOP
	RET
