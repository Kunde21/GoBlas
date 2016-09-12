#include "textflag.h"

#define SIZE 8

#define M			m+0(FP)
#define N			n+8(FP)
#define K_PAR		k+16(FP)
#define ALPHA_PAR 	alpha+24(FP)
#define A_PAR 		a_base+32(FP)
#define B_PAR		b_base+56(FP)
#define C_PAR		c_base+80(FP)
#define LDC_PAR		ldc+104(FP)

#define A   R12
#define B	R13
#define C   R14
#define LDC	R15

#define I	R9
#define J	R10
#define K	R11
#define KSZ R8

#define KCT	CX

#define AO1	SI
#define BO1	BX
#define CO1	DI
#define CO2	DX

#define ALPHA X0
#define RES01 X4
#define RES23 X5

#define SCRATCH1 X1
#define SCRATCH2 X2
#define SCRATCH3 X3

#define DEBUG(val) \
	MOVQ val, (C); \
	RET

#include "gemm_2x2_macros.h"

// func gemm_kernel_2x2(m, n, k int, alpha float64, a,b,c []float64, ldc int)
TEXT ·gemm_kernel_2x2(SB), NOSPLIT, $0
	MOVQ  K_PAR, K
	MOVDDUP_(ALPHA_PAR, ALPHA)
	MOVQ  A_PAR, A
	MOVQ  B_PAR, B
	MOVQ  C_PAR, C
	MOVQ  LDC_PAR, LDC
	IMULQ $SIZE, LDC
	MOVQ  K, KSZ
	IMULQ $SIZE, KSZ

	MOVQ M, J
	SARQ $1, J    // M / 2
	JZ   d1_setup // if M / 2 <= 1

d2:
	MOVQ C, CO1
	LEAQ (C)(LDC*1), CO2
	MOVQ B, BO1

	MOVQ N, I
	SARQ $1, I     // N / 2
	JZ   d21_setup

d22:
	MOVQ A, AO1
	KERNEL_2X2_INIT

	MOVQ K, KCT
	SARQ $2, KCT       // KCT = K / 4
	JZ   d22_2x2_setup

d22_8x8:
	KERNEL_2X2_(0)
	KERNEL_2X2_(2)
	KERNEL_2X2_(4)
	KERNEL_2X2_(6)

	ADDQ $8*SIZE, BO1
	ADDQ $8*SIZE, AO1

	DECQ KCT
	JG   d22_8x8

d22_2x2_setup:
	MOVQ K, KCT
	ANDQ $3, KCT
	JZ   d22_2x2_store

d22_2x2:
	KERNEL_2X2_(0)
	ADDQ $2*SIZE, BO1
	ADDQ $2*SIZE, AO1

	DECQ KCT
	JG   d22_2x2

d22_2x2_store:

	KERNEL_2X2_SAVE

	DECQ I
	JG   d22

d21_setup:
	MOVQ N, I
	ANDQ $1, I  // N % 2
	JZ   d2_end

d21:
	MOVQ A, AO1

	KERNEL_1X2_INIT

	MOVQ K, KCT

d21_1x2:

	KERNEL_1X2

	DECQ KCT
	JG   d21_1x2

d21_1x2_save:

	KERNEL_1X2_SAVE

d2_end:

	LEAQ (A)(KSZ*2), A
	LEAQ (C)(LDC*2), C

	DECQ J
	JG   d2

d1_setup:

	MOVQ M, J
	ANDQ $1, J
	JZ   end

d1:

	MOVQ C, CO1
	MOVQ B, BO1

	MOVQ N, I
	SARQ $1, I
	JZ   d11_setup

d12:

	MOVQ A, AO1

	KERNEL_2X1_INIT

	MOVQ K, KCT

d12_2x1:

	KERNEL_2X1

	DECQ KCT
	JG   d12_2x1

d12_2x1_save:

	KERNEL_2X1_SAVE

	DECQ I
	JG   d12

d11_setup:

	MOVQ N, I
	ANDQ $1, I
	JZ   end

d11:
	MOVQ A, AO1

	KERNEL_1X1_INIT

	MOVQ K, KCT

d11_1x1:

	KERNEL_1X1

	DECQ KCT
	JG   d11_1x1

d11_1x1_save:

	KERNEL_1X1_SAVE

d1_end:

end:
	RET
