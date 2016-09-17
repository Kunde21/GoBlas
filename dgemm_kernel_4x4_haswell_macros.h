#include "dgemm_kernel_4x4_haswell_byteInstr.h"

/* ******************************************************************************************
 * Macro definitions
****************************************************************************************** */
#define INIT4x12 \
	VXORPD Y4, Y4, Y4;    \
	VXORPD Y5, Y5, Y5;    \
	VXORPD Y6, Y6, Y6;    \
	VXORPD Y7, Y7, Y7;    \
	VXORPD Y8, Y8, Y8;    \
	VXORPD Y9, Y9, Y9;    \
	VXORPD Y10, Y10, Y10; \
	VXORPD Y11, Y11, Y11; \
	VXORPD Y12, Y12, Y12; \
	VXORPD Y13, Y13, Y13; \
	VXORPD Y14, Y14, Y14; \
	VXORPD Y15, Y15, Y15

#define KERNEL4x12_I \
	PREFETCHT0 A_PR1(BO);          \
	VMOVUPS    -12 * SIZE(AO), Y1; \
	PREFETCHT0 B_PR1(AO);          \
	VMOVUPS    -16 * SIZE(BO), Y0; \
	PREFETCHT0 B_PR1+64(AO);       \
	VMOVUPS    -8 * SIZE(AO), Y2;  \
	PREFETCHT0 B_PR1+128(AO);      \
	VMOVUPS    -4 * SIZE(AO), Y3;  \
	VMULPD__Y0_Y1_Y4;         \
	PREFETCHT0 B_PR1+192(AO);      \
	VMULPD__Y0_Y2_Y8;         \
	VMULPD__Y0_Y3_Y12;        \
	PREFETCHT0 B_PR1+256(AO);      \
	VPERMPD__0xb1_Y0_Y0;     \
	VMULPD__Y0_Y1_Y5;         \
	VMULPD__Y0_Y2_Y9;         \
	VMULPD__Y0_Y3_Y13;        \
	VPERMPD__0x1b_Y0_Y0;     \
	VMULPD__Y0_Y1_Y6;         \
	VMULPD__Y0_Y2_Y10;        \
\
	ADDQ    $ 12*SIZE, AO;      \
	VMULPD__Y0_Y3_Y14;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VMULPD__Y0_Y1_Y7;         \
	VMOVUPS -12 * SIZE(AO), Y1; \
	VMULPD__Y0_Y2_Y11;        \
	VMOVUPS -8 * SIZE(AO), Y2;  \
	VMULPD__Y0_Y3_Y15;        \
	VMOVUPS -4 * SIZE(AO), Y3

#define KERNEL4x12_M1 \
	PREFETCHT0  A_PR1(BO);          \
	VMOVUPS     -16 * SIZE(BO), Y0; \
	PREFETCHT0  B_PR1(AO);          \
	VFMADD231PD__Y0_Y1_Y4;         \
	PREFETCHT0  B_PR1+64(AO);       \
	VFMADD231PD__Y0_Y2_Y8;         \
	PREFETCHT0  B_PR1+128(AO);      \
	VFMADD231PD__Y0_Y3_Y12;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	VFMADD231PD__Y0_Y2_Y9;         \
	VFMADD231PD__Y0_Y3_Y13;        \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
	VFMADD231PD__Y0_Y2_Y10;        \
\
	VFMADD231PD__Y0_Y3_Y14;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y7;         \
	VMOVUPS     -12 * SIZE(AO), Y1; \
	VFMADD231PD__Y0_Y2_Y11;        \
	VMOVUPS     -8 * SIZE(AO), Y2;  \
	VFMADD231PD__Y0_Y3_Y15;        \
	VMOVUPS     -4 * SIZE(AO), Y3

#define KERNEL4x12_M2 \
	VMOVUPS     -12 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y1_Y4;         \
	VFMADD231PD__Y0_Y2_Y8;         \
	VFMADD231PD__Y0_Y3_Y12;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	VFMADD231PD__Y0_Y2_Y9;         \
	VFMADD231PD__Y0_Y3_Y13;        \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
	VFMADD231PD__Y0_Y2_Y10;        \
\
	ADDQ        $ 8*SIZE, BO;     \
	VFMADD231PD__Y0_Y3_Y14;      \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y7;       \
	VMOVUPS     0 * SIZE(AO), Y1; \
	VFMADD231PD__Y0_Y2_Y11;      \
	VMOVUPS     4 * SIZE(AO), Y2; \
	VFMADD231PD__Y0_Y3_Y15;      \
	VMOVUPS     8 * SIZE(AO), Y3; \
	ADDQ        $ 24*SIZE, AO

#define KERNEL4x12_E \
	VMOVUPS     -12 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y1_Y4;         \
	VFMADD231PD__Y0_Y2_Y8;         \
	VFMADD231PD__Y0_Y3_Y12;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	VFMADD231PD__Y0_Y2_Y9;         \
	VFMADD231PD__Y0_Y3_Y13;        \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
	VFMADD231PD__Y0_Y2_Y10;        \
\
	ADDQ        $ 8*SIZE, BO;   \
	VFMADD231PD__Y0_Y3_Y14;    \
	VPERMPD__0xb1_Y0_Y0; \
	VFMADD231PD__Y0_Y1_Y7;     \
	VFMADD231PD__Y0_Y2_Y11;    \
	VFMADD231PD__Y0_Y3_Y15;    \
	ADDQ        $ 12*SIZE, AO

#define KERNEL4x12_SUB \
	VMOVUPS     -12 * SIZE(AO), Y1; \
	VMOVUPS     -16 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y1_Y4;         \
	VMOVUPS     -8 * SIZE(AO), Y2;  \
	VFMADD231PD__Y0_Y2_Y8;         \
	VMOVUPS     -4 * SIZE(AO), Y3;  \
	VFMADD231PD__Y0_Y3_Y12;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	VFMADD231PD__Y0_Y2_Y9;         \
	ADDQ        $ 12*SIZE, AO;      \
	VFMADD231PD__Y0_Y3_Y13;        \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
	VFMADD231PD__Y0_Y2_Y10;        \
	ADDQ        $ 4*SIZE, BO;       \
	VFMADD231PD__Y0_Y3_Y14;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y7;         \
	VFMADD231PD__Y0_Y2_Y11;        \
	VFMADD231PD__Y0_Y3_Y15

#define SAVE4x12 \
	VBROADCASTSD__ALPHA_Y0; \
\
	VMULPD__Y0_Y4_Y4; \
	VMULPD__Y0_Y5_Y5; \
	VMULPD__Y0_Y6_Y6; \
	VMULPD__Y0_Y7_Y7; \
\
	VMULPD__Y0_Y8_Y8;   \
	VMULPD__Y0_Y9_Y9;   \
	VMULPD__Y0_Y10_Y10; \
	VMULPD__Y0_Y11_Y11; \
\
	VMULPD__Y0_Y12_Y12; \
	VMULPD__Y0_Y13_Y13; \
	VMULPD__Y0_Y14_Y14; \
	VMULPD__Y0_Y15_Y15; \
\
	VPERMPD__0xb1_Y5_Y5; \
	VPERMPD__0xb1_Y7_Y7; \
\
	VBLENDPD__0x0a_Y5_Y4_Y0; \
	VBLENDPD__0x05_Y5_Y4_Y1; \
	VBLENDPD__0x0a_Y7_Y6_Y2; \
	VBLENDPD__0x05_Y7_Y6_Y3; \
\
	VPERMPD__0x1b_Y2_Y2; \
	VPERMPD__0x1b_Y3_Y3; \
	VPERMPD__0xb1_Y2_Y2; \
	VPERMPD__0xb1_Y3_Y3; \
\
	VBLENDPD__0x03_Y0_Y2_Y4; \
	VBLENDPD__0x03_Y1_Y3_Y5; \
	VBLENDPD__0x03_Y2_Y0_Y6; \
	VBLENDPD__0x03_Y3_Y1_Y7; \
\
	LEAQ (CO1)(LDC*2), AX; \
\
	VADDPD__CO1_Y4_Y4;       \
	VADDPD__CO1_LDC__Y5_Y5;  \
	VADDPD__AX_Y6_Y6;      \
	VADDPD__AX_LDC__Y7_Y7; \
\
	VMOVUPS Y4, (CO1);       \
	VMOVUPS Y5, (CO1)(LDC*1);  \
	VMOVUPS Y6, (AX);      \
	VMOVUPS Y7, (AX)(LDC*1); \
\
	PREFETCHT0 32(CO1);       \
	PREFETCHT0 32(CO1)(LDC*1);  \
	PREFETCHT0 32(AX);      \
	PREFETCHT0 32(AX)(LDC*1); \
\
	VPERMPD__0xb1_Y9_Y9;   \
	VPERMPD__0xb1_Y11_Y11; \
\
	VBLENDPD__0x0a_Y9_Y8_Y0;   \
	VBLENDPD__0x05_Y9_Y8_Y1;   \
	VBLENDPD__0x0a_Y11_Y10_Y2; \
	VBLENDPD__0x05_Y11_Y10_Y3; \
\
	VPERMPD__0x1b_Y2_Y2; \
	VPERMPD__0x1b_Y3_Y3; \
	VPERMPD__0xb1_Y2_Y2; \
	VPERMPD__0xb1_Y3_Y3; \
\
	VBLENDPD__0x03_Y0_Y2_Y4; \
	VBLENDPD__0x03_Y1_Y3_Y5; \
	VBLENDPD__0x03_Y2_Y0_Y6; \
	VBLENDPD__0x03_Y3_Y1_Y7; \
\
	LEAQ (AX)(LDC*2), AX; \
	LEAQ (AX)(LDC*2), BP; \
\
	VADDPD__AX_Y4_Y4;      \
	VADDPD__AX_LDC__Y5_Y5; \
	VADDPD__BP_Y6_Y6;      \
	VADDPD__BP_LDC__Y7_Y7; \
\
	VMOVUPS Y7, (AX);      \
	VMOVUPS Y6, (AX)(LDC*1); \
	VMOVUPS Y5, (BP);      \
	VMOVUPS Y4, (BP)(LDC*1); \
\
	PREFETCHT0 32(AX);      \
	PREFETCHT0 32(AX)(LDC*1); \
	PREFETCHT0 32(BP);      \
	PREFETCHT0 32(BP)(LDC*1); \
\
	VPERMPD__0xb1_Y13_Y13; \
	VPERMPD__0xb1_Y15_Y15; \
\
	VBLENDPD__0x0a_Y13_Y12_Y0; \
	VBLENDPD__0x05_Y13_Y12_Y1; \
	VBLENDPD__0x0a_Y15_Y14_Y2; \
	VBLENDPD__0x05_Y15_Y14_Y3; \
\
	VPERMPD__0x1b_Y2_Y2; \
	VPERMPD__0x1b_Y3_Y3; \
	VPERMPD__0xb1_Y2_Y2; \
	VPERMPD__0xb1_Y3_Y3; \
\
	VBLENDPD__0x03_Y0_Y2_Y4; \
	VBLENDPD__0x03_Y1_Y3_Y5; \
	VBLENDPD__0x03_Y2_Y0_Y6; \
	VBLENDPD__0x03_Y3_Y1_Y7; \
\
	LEAQ (AX)(LDC*4), AX; \
	LEAQ (BP)(LDC*4), BP; \
\
	VADDPD__AX_Y4_Y4;      \
	VADDPD__AX_LDC__Y5_Y5; \
	VADDPD__BP_Y6_Y6;      \
	VADDPD__BP_LDC__Y7_Y7; \
\
	VMOVUPS Y4, (AX);      \
	VMOVUPS Y5, (AX)(LDC*1); \
	VMOVUPS Y6, (BP);      \
	VMOVUPS Y7, (BP)(LDC*1); \
\
	PREFETCHT0 32(AX);      \
	PREFETCHT0 32(AX)(LDC*1); \
	PREFETCHT0 32(BP);      \
	PREFETCHT0 32(BP)(LDC*1); \
\
	ADDQ $ 4*SIZE, CO1

// ****************************************************************************************

#define INIT2x12 \
	VXORPD X4, X4, X4;    \
	VXORPD X5, X5, X5;    \
	VXORPD X6, X6, X6;    \
	VXORPD X7, X7, X7;    \
	VXORPD X8, X8, X8;    \
	VXORPD X9, X9, X9;    \
	VXORPD X10, X10, X10; \
	VXORPD X11, X11, X11; \
	VXORPD X12, X12, X12; \
	VXORPD X13, X13, X13; \
	VXORPD X14, X14, X14; \
	VXORPD X15, X15, X15

#define KERNEL2x12_SUB \
	VMOVUPS     -16 * SIZE(BO), X0; \
	VMOVDDUP__n12_BO__X1; \
	VMOVDDUP__n11_BO__X2; \
	VMOVDDUP__n10_BO__X3; \
	VFMADD231PD__X0_X1_X4;         \
	VMOVDDUP__n9_BO__X1;  \
	VFMADD231PD__X0_X2_X5;         \
	VMOVDDUP__n8_BO__X2;  \
	VFMADD231PD__X0_X3_X6;         \
	VMOVDDUP__n7_BO__X3;  \
	VFMADD231PD__X0_X1_X7;         \
	VMOVDDUP__n6_BO__X1;  \
	VFMADD231PD__X0_X2_X8;         \
	VMOVDDUP__n5_BO__X2;  \
	VFMADD231PD__X0_X3_X9;         \
	VMOVDDUP__n4_BO__X3;  \
	VFMADD231PD__X0_X1_X10;        \
	VMOVDDUP__n3_BO__X1;  \
	VFMADD231PD__X0_X2_X11;        \
	VMOVDDUP__n2_BO__X2;  \
	VFMADD231PD__X0_X3_X12;        \
	VMOVDDUP__n1_BO__X3;  \
	VFMADD231PD__X0_X1_X13;        \
	ADDQ        $ 12*SIZE, AO;      \
	VFMADD231PD__X0_X2_X14;        \
	ADDQ        $ 2*SIZE, BO;       \
	VFMADD231PD__X0_X3_X15

#define SAVE2x12 \
	VMOVDDUP__ALPHA_X0; \
\
	VMULPD__X0_X4_X4; \
	VMULPD__X0_X5_X5; \
	VMULPD__X0_X6_X6; \
	VMULPD__X0_X7_X7; \
\
	VMULPD__X0_X8_X8;   \
	VMULPD__X0_X9_X9;   \
	VMULPD__X0_X10_X10; \
	VMULPD__X0_X11_X11; \
\
	VMULPD__X0_X12_X12; \
	VMULPD__X0_X13_X13; \
	VMULPD__X0_X14_X14; \
	VMULPD__X0_X15_X15; \
\
	LEAQ (CO1)(LDC*2), AX; \
\
	VADDPD__CO1_X4_X4;       \
	VADDPD__CO1_LDC__X5_X5;  \
	VADDPD__AX_X6_X6;      \
	VADDPD__AX_LDC__X7_X7; \
\
	VMOVUPS X4, (CO1);       \
	VMOVUPS X5, (CO1)(LDC*1);  \
	VMOVUPS X6, (AX);      \
	VMOVUPS X7, (AX)(LDC*1); \
\
	LEAQ (AX)(LDC*2), AX; \
	LEAQ (AX)(LDC*2), BP; \
\
	VADDPD__AX_X8_X4;       \
	VADDPD__AX_LDC__X9_X5;  \
	VADDPD__BP_X10_X6;      \
	VADDPD__BP_LDC__X11_X7; \
\
	VMOVUPS X4, (AX);      \
	VMOVUPS X5, (AX)(LDC*1); \
	VMOVUPS X6, (BP);      \
	VMOVUPS X7, (BP)(LDC*1); \
\
	LEAQ (AX)(LDC*4), AX; \
	LEAQ (BP)(LDC*4), BP; \
\
	VADDPD__AX_X12_X4;      \
	VADDPD__AX_LDC__X13_X5; \
	VADDPD__BP_X14_X6;      \
	VADDPD__BP_LDC__X15_X7; \
\
	VMOVUPS X4, (AX);      \
	VMOVUPS X5, (AX)(LDC*1); \
	VMOVUPS X6, (BP);      \
	VMOVUPS X7, (BP)(LDC*1); \
\
	ADDQ $ 2*SIZE, CO1

// ****************************************************************************************

#define INIT1x12 \
	VXORPD X4, X4, X4;    \
	VXORPD X5, X5, X5;    \
	VXORPD X6, X6, X6;    \
	VXORPD X7, X7, X7;    \
	VXORPD X8, X8, X8;    \
	VXORPD X9, X9, X9;    \
	VXORPD X10, X10, X10; \
	VXORPD X11, X11, X11; \
	VXORPD X12, X12, X12; \
	VXORPD X13, X13, X13; \
	VXORPD X14, X14, X14; \
	VXORPD X15, X15, X15

#define KERNEL1x12_SUB \
	VMOVSD      -16 * SIZE(BO), X0; \
	VMOVSD      -12 * SIZE(AO), X1; \
	VMOVSD      -11 * SIZE(AO), X2; \
	VMOVSD      -10 * SIZE(AO), X3; \
	VFMADD231SD__X0_X1_X4;         \
	VMOVSD      -9 * SIZE(AO), X1;  \
	VFMADD231SD__X0_X2_X5;         \
	VMOVSD      -8 * SIZE(AO), X2;  \
	VFMADD231SD__X0_X3_X6;         \
	VMOVSD      -7 * SIZE(AO), X3;  \
	VFMADD231SD__X0_X1_X7;         \
	VMOVSD      -6 * SIZE(AO), X1;  \
	VFMADD231SD__X0_X2_X8;         \
	VMOVSD      -5 * SIZE(AO), X2;  \
	VFMADD231SD__X0_X3_X9;         \
	VMOVSD      -4 * SIZE(AO), X3;  \
	VFMADD231SD__X0_X1_X10;        \
	VMOVSD      -3 * SIZE(AO), X1;  \
	VFMADD231SD__X0_X2_X11;        \
	VMOVSD      -2 * SIZE(AO), X2;  \
	VFMADD231SD__X0_X3_X12;        \
	VMOVSD      -1 * SIZE(AO), X3;  \
	VFMADD231SD__X0_X1_X13;        \
	ADDQ        $ 12*SIZE, AO;      \
	VFMADD231SD__X0_X2_X14;        \
	ADDQ        $ 1*SIZE, BO;       \
	VFMADD231SD__X0_X3_X15

#define SAVE1x12 \
	VMOVSD ALPHA, X0; \
\
	VMULSD__X0_X4_X4; \
	VMULSD__X0_X5_X5; \
	VMULSD__X0_X6_X6; \
	VMULSD__X0_X7_X7; \
\
	VMULSD__X0_X8_X8;   \
	VMULSD__X0_X9_X9;   \
	VMULSD__X0_X10_X10; \
	VMULSD__X0_X11_X11; \
\
	VMULSD__X0_X12_X12; \
	VMULSD__X0_X13_X13; \
	VMULSD__X0_X14_X14; \
	VMULSD__X0_X15_X15; \
\
	LEAQ (CO1)(LDC*2), AX; \
\
	VADDSD__CO1_X4_X4;       \
	VADDSD__CO1_LDC__X5_X5;  \
	VADDSD__AX_X6_X6;      \
	VADDSD__AX_LDC__X7_X7; \
\
	VMOVSD X4, (CO1);       \
	VMOVSD X5, (CO1)(LDC*1);  \
	VMOVSD X6, (AX);      \
	VMOVSD X7, (AX)(LDC*1); \
\
	LEAQ (AX)(LDC*2), AX; \
	LEAQ (AX)(LDC*2), BP; \
\
	VADDSD__AX_X8_X4;       \
	VADDSD__AX_LDC__X9_X5;  \
	VADDSD__BP_X10_X6;      \
	VADDSD__BP_LDC__X11_X7; \
\
	VMOVSD X4, (AX);      \
	VMOVSD X5, (AX)(LDC*1); \
	VMOVSD X6, (BP);      \
	VMOVSD X7, (BP)(LDC*1); \
\
	LEAQ (AX)(LDC*4), AX; \
	LEAQ (BP)(LDC*4), BP; \
\
	VADDSD__AX_X12_X4;      \
	VADDSD__AX_LDC__X13_X5; \
	VADDSD__BP_X14_X6;      \
	VADDSD__BP_LDC__X15_X7; \
\
	VMOVSD X4, (AX);      \
	VMOVSD X5, (AX)(LDC*1); \
	VMOVSD X6, (BP);      \
	VMOVSD X7, (BP)(LDC*1); \
\
	ADDQ $ 1*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT4x4 \
	VXORPD Y4, Y4, Y4; \
	VXORPD Y5, Y5, Y5; \
	VXORPD Y6, Y6, Y6; \
	VXORPD Y7, Y7, Y7

#define KERNEL4x4_I \
	PREFETCHT0 A_PR1(BO);          \
	VMOVUPS    -12 * SIZE(AO), Y1; \
	VMOVUPS    -16 * SIZE(BO), Y0; \
	VMULPD__Y0_Y1_Y4;         \
	VPERMPD__0xb1_Y0_Y0;     \
	VMULPD__Y0_Y1_Y5;         \
	VPERMPD__0x1b_Y0_Y0;     \
	VMULPD__Y0_Y1_Y6;         \
\
	ADDQ    $ 4*SIZE, AO;       \
	VPERMPD__0xb1_Y0_Y0;     \
	VMULPD__Y0_Y1_Y7;         \
	VMOVUPS -12 * SIZE(AO), Y1

#define KERNEL4x4_M1 \
	PREFETCHT0  A_PR1(BO);          \
	VMOVUPS     -16 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y1_Y4;         \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
\
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y7;         \
	VMOVUPS     -12 * SIZE(AO), Y1

#define KERNEL4x4_M2 \
	VMOVUPS     -12 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y1_Y4;         \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
\
	ADDQ        $ 8*SIZE, BO;      \
	VPERMPD__0xb1_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y7;        \
	VMOVUPS     -8 * SIZE(AO), Y1; \
	ADDQ        $ 8*SIZE, AO

#define KERNEL4x4_E \
	VMOVUPS     -12 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y1_Y4;         \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
\
	ADDQ        $ 8*SIZE, BO;   \
	VPERMPD__0xb1_Y0_Y0; \
	VFMADD231PD__Y0_Y1_Y7;     \
	ADDQ        $ 4*SIZE, AO

#define KERNEL4x4_SUB \
	VMOVUPS     -12 * SIZE(AO), Y1; \
	VMOVUPS     -16 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y1_Y4;         \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y5;         \
	ADDQ        $ 4*SIZE, AO;       \
	VPERMPD__0x1b_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y6;         \
	ADDQ        $ 4*SIZE, BO;       \
	VPERMPD__0xb1_Y0_Y0;     \
	VFMADD231PD__Y0_Y1_Y7

#define SAVE4x4 \
	VBROADCASTSD__ALPHA_Y0; \
\
	VMULPD__Y0_Y4_Y4; \
	VMULPD__Y0_Y7_Y7; \
	VMULPD__Y0_Y5_Y5; \
	VMULPD__Y0_Y6_Y6; \
\
	VPERMPD__0xb1_Y5_Y5; \
	VPERMPD__0xb1_Y7_Y7; \
\
	VBLENDPD__0x0a_Y5_Y4_Y0; \
	VBLENDPD__0x05_Y5_Y4_Y1; \
	VBLENDPD__0x0a_Y7_Y6_Y2; \
	VBLENDPD__0x05_Y7_Y6_Y3; \
\
	VPERMPD__0x1b_Y2_Y2; \
	VPERMPD__0x1b_Y3_Y3; \
	VPERMPD__0xb1_Y2_Y2; \
	VPERMPD__0xb1_Y3_Y3; \
\
	VBLENDPD__0x03_Y0_Y2_Y4; \
	VBLENDPD__0x03_Y1_Y3_Y5; \
	VBLENDPD__0x03_Y2_Y0_Y6; \
	VBLENDPD__0x03_Y3_Y1_Y7; \
\
	LEAQ (CO1)(LDC*2), AX; \
\
	VADDPD__CO1_Y4_Y4;       \
	VADDPD__CO1_LDC__Y5_Y5;  \
	VADDPD__AX_Y6_Y6;      \
	VADDPD__AX_LDC__Y7_Y7; \
\
	VMOVUPS Y4, (CO1);       \
	VMOVUPS Y5, (CO1)(LDC*1);  \
	VMOVUPS Y6, (AX);      \
	VMOVUPS Y7, (AX)(LDC*1); \
\
	ADDQ $ 4*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT2x4 \
	VXORPD X4, X4, X4; \
	VXORPD X5, X5, X5; \
	VXORPD X6, X6, X6; \
	VXORPD X7, X7, X7; \

#define KERNEL2x4_SUB \
	VMOVDDUP__n12_BO__X1; \
	VMOVUPS     -16 * SIZE(BO), X0; \
	VMOVDDUP__n11_BO__X2; \
	VFMADD231PD__X0_X1_X4;         \
	VMOVDDUP__n10_BO__X3; \
	VFMADD231PD__X0_X2_X5;         \
	VMOVDDUP__n9_BO__X8;  \
	VFMADD231PD__X0_X3_X6;         \
	ADDQ        $ 4*SIZE, AO;       \
	VFMADD231PD__X0_X8_X7;         \
	ADDQ        $ 2*SIZE, BO

#define SAVE2x4 \
	VMOVDDUP__ALPHA_X0; \
\
	VMULPD__X0_X4_X4; \
	VMULPD__X0_X5_X5; \
	VMULPD__X0_X6_X6; \
	VMULPD__X0_X7_X7; \
\
	LEAQ (CO1)(LDC*2), AX; \
\
	VADDPD__CO1_X4_X4;       \
	VADDPD__CO1_LDC__X5_X5;  \
	VADDPD__AX_X6_X6;      \
	VADDPD__AX_LDC__X7_X7; \
\
	VMOVUPS X4, (CO1);       \
	VMOVUPS X5, (CO1)(LDC*1);  \
	VMOVUPS X6, (AX);      \
	VMOVUPS X7, (AX)(LDC*1); \
\
	ADDQ $ 2*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT1x4 \
	VXORPD X4, X4, X4; \
	VXORPD X5, X5, X5; \
	VXORPD X6, X6, X6; \
	VXORPD X7, X7, X7

#define KERNEL1x4_SUB \
	VMOVSD      -12 * SIZE(AO), X1; \
	VMOVSD      -16 * SIZE(BO), X0; \
	VMOVSD      -11 * SIZE(AO), X2; \
	VFMADD231SD__X0_X1_X4;         \
	VMOVSD      -10 * SIZE(AO), X3; \
	VFMADD231SD__X0_X2_X5;         \
	VMOVSD      -9 * SIZE(AO), X8;  \
	VFMADD231SD__X0_X3_X6;         \
	ADDQ        $ 4*SIZE, AO;       \
	VFMADD231SD__X0_X8_X7;         \
	ADDQ        $ 1*SIZE, BO

#define SAVE1x4 \
	VMOVSD ALPHA, X0; \
\
	VMULSD__X0_X4_X4; \
	VMULSD__X0_X5_X5; \
	VMULSD__X0_X6_X6; \
	VMULSD__X0_X7_X7; \
\
	LEAQ (CO1)(LDC*2), AX; \
\
	VADDSD__CO1_X4_X4;       \
	VADDSD__CO1_LDC__X5_X5;  \
	VADDSD__AX_X6_X6;      \
	VADDSD__AX_LDC__X7_X7; \
\
	VMOVSD X4, (CO1);       \
	VMOVSD X5, (CO1)(LDC*1);  \
	VMOVSD X6, (AX);      \
	VMOVSD X7, (AX)(LDC*1); \
\
	ADDQ $ 1*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT4x2 \
	VXORPD X4, X4, X4; \
	VXORPD X5, X5, X5; \
	VXORPD X6, X6, X6; \
	VXORPD X7, X7, X7

#define KERNEL4x2_SUB \
	VMOVDDUP__n12_BO__X2; \
	VMOVUPS     -16 * SIZE(BO), X0; \
	VMOVUPS     -14 * SIZE(BO), X1; \
	VMOVDDUP__n11_BO__X3; \
	VFMADD231PD__X0_X2_X4;         \
	VFMADD231PD__X1_X2_X5;         \
	VFMADD231PD__X0_X3_X6;         \
	VFMADD231PD__X1_X3_X7;         \
	ADDQ        $ 2*SIZE, AO;       \
	ADDQ        $ 4*SIZE, BO

#define SAVE4x2 \
	VMOVDDUP__ALPHA_X0; \
\
	VMULPD__X0_X4_X4; \
	VMULPD__X0_X5_X5; \
	VMULPD__X0_X6_X6; \
	VMULPD__X0_X7_X7; \
\
	VADDPD__CO1_X4_X4;              \
	VADDPD__2_CO1__X5_X5;      \
	VADDPD__CO1_LDC__X6_X6;         \
	VADDPD__2_CO1_LDC__X7_X7; \
\
	VMOVUPS X4, (CO1);              \
	VMOVUPS X5, 2 * SIZE(CO1);      \
	VMOVUPS X6, (CO1)(LDC*1);         \
	VMOVUPS X7, 2 * SIZE(CO1)(LDC*1); \
\
	ADDQ $ 4*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT2x2 \
	VXORPD X4, X4, X4; \
	VXORPD X6, X6, X6

#define KERNEL2x2_SUB \
	VMOVDDUP__n12_BO__X2; \
	VMOVUPS     -16 * SIZE(BO), X0; \
	VMOVDDUP__n11_BO__X3; \
	VFMADD231PD__X0_X2_X4;         \
	VFMADD231PD__X0_X3_X6;         \
	ADDQ        $ 2*SIZE, AO;       \
	ADDQ        $ 2*SIZE, BO

#define SAVE2x2 \
	VMOVDDUP__ALPHA_X0; \
\
	VMULPD__X0_X4_X4; \
	VMULPD__X0_X6_X6; \
\
	VADDPD__CO1_X4_X4;      \
	VADDPD__CO1_LDC__X6_X6; \
\
	VMOVUPS X4, (CO1);      \
	VMOVUPS X6, (CO1)(LDC*1); \
\
	ADDQ $ 2*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT1x2 \
	VXORPD X4, X4, X4; \
	VXORPD X5, X5, X5

#define KERNEL1x2_SUB \
	VMOVSD      -12 * SIZE(AO), X1; \
	VMOVSD      -16 * SIZE(BO), X0; \
	VMOVSD      -11 * SIZE(AO), X2; \
	VFMADD231SD__X0_X1_X4;         \
	VFMADD231SD__X0_X2_X5;         \
	ADDQ        $ 2*SIZE, AO;       \
	ADDQ        $ 1*SIZE, BO

#define SAVE1x2 \
	VMOVSD ALPHA, X0; \
\
	VMULSD__X0_X4_X4; \
	VMULSD__X0_X5_X5; \
\
	VADDSD__CO1_X4_X4;      \
	VADDSD__CO1_LDC__X5_X5; \
\
	VMOVSD X4, (CO1);      \
	VMOVSD X5, (CO1)(LDC*1); \
\
	ADDQ $ 1*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT4x1 \
	VXORPD Y4, Y4, Y4; \
	VXORPD Y5, Y5, Y5; \
	VXORPD Y6, Y6, Y6; \
	VXORPD Y7, Y7, Y7

#define KERNEL4x1 \
	VBROADCASTSD__n12_BO__Y0; \
	VBROADCASTSD__n11_BO__Y1; \
	VBROADCASTSD__n10_BO__Y2; \
	VBROADCASTSD__n9_BO__Y3; \
\
	VFMADD231PD__n16_AO__Y0_Y4; \
	VFMADD231PD__n12_AO__Y1_Y5; \
\
	VBROADCASTSD__n8_BO__Y0; \
	VBROADCASTSD__n7_BO__Y1; \
\
	VFMADD231PD__n8_AO__Y2_Y6; \
	VFMADD231PD__n4_AO__Y3_Y7; \
\
	VBROADCASTSD__n6_BO__Y2; \
	VBROADCASTSD__n5_BO__Y3; \
\
	VFMADD231PD__0_AO__Y0_Y4; \
	VFMADD231PD__4_AO__Y1_Y5; \
	VFMADD231PD__8_AO__Y2_Y6; \
	VFMADD231PD__12_AO__Y3_Y7; \
\
	ADDQ $ 8 *SIZE, AO; \
	ADDQ $ 32*SIZE, BO

#define KERNEL4x1_SUB \
	VBROADCASTSD__n12_BO__Y2; \
	VMOVUPS      -16 * SIZE(BO), Y0; \
	VFMADD231PD__Y0_Y2_Y4;         \
	ADDQ         $ 1*SIZE, AO;       \
	ADDQ         $ 4*SIZE, BO

#define SAVE4x1 \
	VBROADCASTSD__ALPHA_Y0; \
\
	VADDPD__Y4_Y5_Y4; \
	VADDPD__Y6_Y7_Y6; \
	VADDPD__Y4_Y6_Y4; \
\
	VMULPD__Y0_Y4_Y4; \
\
	VADDPD__CO1_Y4_Y4; \
\
	VMOVUPS Y4, (CO1); \
\
	ADDQ $ 4*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT2x1 \
	VXORPD X4, X4, X4

#define KERNEL2x1_SUB \
	VMOVDDUP__n12_BO__X2; \
	VMOVUPS     -16 * SIZE(BO), X0; \
	VFMADD231PD__X0_X2_X4;         \
	ADDQ        $ 1*SIZE, AO;       \
	ADDQ        $ 2*SIZE, BO

#define SAVE2x1 \
	VMOVDDUP__ALPHA_X0; \
\
	VMULPD__X0_X4_X4; \
\
	VADDPD__CO1_X4_X4; \
\
	VMOVUPS X4, (CO1); \
\
	ADDQ $ 2*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT1x1 \
	VXORPD X4, X4, X4

#define KERNEL1x1_SUB \
	VMOVSD      -12 * SIZE(AO), X1; \
	VMOVSD      -16 * SIZE(BO), X0; \
	VFMADD231SD__X0_X1_X4;         \
	ADDQ        $ 1*SIZE, AO;       \
	ADDQ        $ 1*SIZE, BO

#define SAVE1x1 \
	VMOVSD ALPHA, X0; \
\
	VMULSD__X0_X4_X4; \
\
	VADDSD__CO1_X4_X4; \
\
	VMOVSD X4, (CO1); \
\
	ADDQ $ 1*SIZE, CO1

// *****************************************************************************************
