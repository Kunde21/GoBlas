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
	PREFETCHT0 A_PR1(BO);     \
	VMOVUPS__n12_AO__Y1;      \
	PREFETCHT0 B_PR1(AO);     \
	VMOVUPS__n16_BO__Y0;      \
	PREFETCHT0 B_PR1+64(AO);  \
	VMOVUPS__n8_AO__Y2;       \
	PREFETCHT0 B_PR1+128(AO); \
	VMOVUPS__n4_AO__Y3;       \
	VMULPD__Y0_Y1_Y4;         \
	PREFETCHT0 B_PR1+192(AO); \
	VMULPD__Y0_Y2_Y8;         \
	VMULPD__Y0_Y3_Y12;        \
	PREFETCHT0 B_PR1+256(AO); \
	VPERMPD__0xb1_Y0_Y0;      \
	VMULPD__Y0_Y1_Y5;         \
	VMULPD__Y0_Y2_Y9;         \
	VMULPD__Y0_Y3_Y13;        \
	VPERMPD__0x1b_Y0_Y0;      \
	VMULPD__Y0_Y1_Y6;         \
	VMULPD__Y0_Y2_Y10;        \
	                          \
	ADDQ       $ 12*SIZE, AO; \
	VMULPD__Y0_Y3_Y14;        \
	VPERMPD__0xb1_Y0_Y0;      \
	VMULPD__Y0_Y1_Y7;         \
	VMOVUPS__n12_AO__Y1;      \
	VMULPD__Y0_Y2_Y11;        \
	VMOVUPS__n8_AO__Y2;       \
	VMULPD__Y0_Y3_Y15;        \
	VMOVUPS__n4_AO__Y3

#define KERNEL4x12_M1 \
	PREFETCHT0 A_PR1(BO);     \
	VMOVUPS__n16_BO__Y0;      \
	PREFETCHT0 B_PR1(AO);     \
	VFMADD231PD__Y0_Y1_Y4;    \
	PREFETCHT0 B_PR1+64(AO);  \
	VFMADD231PD__Y0_Y2_Y8;    \
	PREFETCHT0 B_PR1+128(AO); \
	VFMADD231PD__Y0_Y3_Y12;   \
	VPERMPD__0xb1_Y0_Y0;      \
	VFMADD231PD__Y0_Y1_Y5;    \
	VFMADD231PD__Y0_Y2_Y9;    \
	VFMADD231PD__Y0_Y3_Y13;   \
	VPERMPD__0x1b_Y0_Y0;      \
	VFMADD231PD__Y0_Y1_Y6;    \
	VFMADD231PD__Y0_Y2_Y10;   \
	                          \
	VFMADD231PD__Y0_Y3_Y14;   \
	VPERMPD__0xb1_Y0_Y0;      \
	VFMADD231PD__Y0_Y1_Y7;    \
	VMOVUPS__n12_AO__Y1;      \
	VFMADD231PD__Y0_Y2_Y11;   \
	VMOVUPS__n8_AO__Y2;       \
	VFMADD231PD__Y0_Y3_Y15;   \
	VMOVUPS__n4_AO__Y3

#define KERNEL4x12_M2 \
	VMOVUPS__n12_BO__Y0;    \
	VFMADD231PD__Y0_Y1_Y4;  \
	VFMADD231PD__Y0_Y2_Y8;  \
	VFMADD231PD__Y0_Y3_Y12; \
	VPERMPD__0xb1_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y5;  \
	VFMADD231PD__Y0_Y2_Y9;  \
	VFMADD231PD__Y0_Y3_Y13; \
	VPERMPD__0x1b_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y6;  \
	VFMADD231PD__Y0_Y2_Y10; \
	                        \
	ADDQ $ 8*SIZE, BO;      \
	VFMADD231PD__Y0_Y3_Y14; \
	VPERMPD__0xb1_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y7;  \
	VMOVUPS__0_AO__Y1;      \
	VFMADD231PD__Y0_Y2_Y11; \
	VMOVUPS__4_AO__Y2;      \
	VFMADD231PD__Y0_Y3_Y15; \
	VMOVUPS__8_AO__Y3;      \
	ADDQ $ 24*SIZE, AO

#define KERNEL4x12_E \
	VMOVUPS__n12_BO__Y0;    \
	VFMADD231PD__Y0_Y1_Y4;  \
	VFMADD231PD__Y0_Y2_Y8;  \
	VFMADD231PD__Y0_Y3_Y12; \
	VPERMPD__0xb1_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y5;  \
	VFMADD231PD__Y0_Y2_Y9;  \
	VFMADD231PD__Y0_Y3_Y13; \
	VPERMPD__0x1b_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y6;  \
	VFMADD231PD__Y0_Y2_Y10; \
	                        \
	ADDQ $ 8*SIZE, BO;      \
	VFMADD231PD__Y0_Y3_Y14; \
	VPERMPD__0xb1_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y7;  \
	VFMADD231PD__Y0_Y2_Y11; \
	VFMADD231PD__Y0_Y3_Y15; \
	ADDQ $ 12*SIZE, AO

#define KERNEL4x12_SUB \
	VMOVUPS__n12_AO__Y1;    \
	VMOVUPS__n16_BO__Y0;    \
	VFMADD231PD__Y0_Y1_Y4;  \
	VMOVUPS__n8_AO__Y2;     \
	VFMADD231PD__Y0_Y2_Y8;  \
	VMOVUPS__n4_AO__Y3;     \
	VFMADD231PD__Y0_Y3_Y12; \
	VPERMPD__0xb1_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y5;  \
	VFMADD231PD__Y0_Y2_Y9;  \
	ADDQ $ 12*SIZE, AO;     \
	VFMADD231PD__Y0_Y3_Y13; \
	VPERMPD__0x1b_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y6;  \
	VFMADD231PD__Y0_Y2_Y10; \
	ADDQ $ 4*SIZE, BO;      \
	VFMADD231PD__Y0_Y3_Y14; \
	VPERMPD__0xb1_Y0_Y0;    \
	VFMADD231PD__Y0_Y1_Y7;  \
	VFMADD231PD__Y0_Y2_Y11; \
	VFMADD231PD__Y0_Y3_Y15

#define SAVE4x12 \
	VBROADCASTSD__ALPHA_Y0;      \
	                             \
	VUNPCKLPD_Y4_Y5_Y2;          \
	VUNPCKHPD_Y5_Y4_Y3;          \
	VUNPCKLPD_Y6_Y7_Y4;          \
	VUNPCKHPD_Y7_Y6_Y5;          \
	                             \
	VPERM2F128_0x31_Y4_Y2_Y6;    \
	VPERM2F128_0x31_Y5_Y3_Y7;    \
	VPERM2F128_0x20_Y2_Y4_Y4;    \
	VPERM2F128_0x20_Y3_Y5_Y5;    \
	                             \
	LEAQ       (CO1)(LDC*2), AX; \
	                             \
	VFMADD213PD_CO1__Y0_Y4;      \
	VFMADD213PD_CO1_LDC__Y0_Y5;  \
	VFMADD213PD_AX__Y0_Y6;       \
	VFMADD213PD_AX_LDC__Y0_Y7;   \
	                             \
	VMOVUPS__Y4_CO1;             \
	VMOVUPS__Y5_CO1_LDC;         \
	VMOVUPS__Y6_AX;              \
	VMOVUPS__Y7_AX_LDC;          \
	                             \
	PREFETCHT0 32(CO1);          \
	                             \
	VUNPCKLPD_Y8_Y9_Y2;          \
	VUNPCKHPD_Y9_Y8_Y3;          \
	VUNPCKLPD_Y10_Y11_Y4;        \
	VUNPCKHPD_Y11_Y10_Y5;        \
	                             \
	VPERM2F128_0x31_Y4_Y2_Y6;    \
	VPERM2F128_0x31_Y5_Y3_Y7;    \
	VPERM2F128_0x20_Y2_Y4_Y4;    \
	VPERM2F128_0x20_Y3_Y5_Y5;    \
	                             \
	PREFETCHT0 32(AX)(LDC*1);    \
	                             \
	LEAQ       (AX)(LDC*2), AX;  \
	LEAQ       (AX)(LDC*2), BP;  \
	                             \
	VFMADD213PD_AX__Y0_Y4;       \
	VFMADD213PD_AX_LDC__Y0_Y5;   \
	VFMADD213PD_BP__Y0_Y6;       \
	VFMADD213PD_BP_LDC__Y0_Y7;   \
	                             \
	VMOVUPS__Y4_AX;              \
	VMOVUPS__Y5_AX_LDC;          \
	VMOVUPS__Y6_BP;              \
	VMOVUPS__Y7_BP_LDC;          \
	                             \
	PREFETCHT0 32(AX);           \
	                             \
	VUNPCKLPD_Y12_Y13_Y2;        \
	VUNPCKHPD_Y13_Y12_Y3;        \
	VUNPCKLPD_Y14_Y15_Y4;        \
	VUNPCKHPD_Y15_Y14_Y5;        \
	                             \
	VPERM2F128_0x31_Y4_Y2_Y6;    \
	VPERM2F128_0x31_Y5_Y3_Y7;    \
	VPERM2F128_0x20_Y2_Y4_Y4;    \
	VPERM2F128_0x20_Y3_Y5_Y5;    \
	                             \
	PREFETCHT0 32(BP)(LDC*1);    \
	                             \
	LEAQ       (AX)(LDC*4), AX;  \
	LEAQ       (BP)(LDC*4), BP;  \
	                             \
	VFMADD213PD_AX__Y0_Y4;       \
	VFMADD213PD_AX_LDC__Y0_Y5;   \
	VFMADD213PD_BP__Y0_Y6;       \
	VFMADD213PD_BP_LDC__Y0_Y7;   \
	                             \
	VMOVUPS__Y4_AX;              \
	VMOVUPS__Y5_AX_LDC;          \
	VMOVUPS__Y6_BP;              \
	VMOVUPS__Y7_BP_LDC;          \
	                             \
	PREFETCHT0 32(AX);           \
	PREFETCHT0 32(AX)(LDC*1);    \
	PREFETCHT0 32(BP);           \
	PREFETCHT0 32(BP)(LDC*1);    \
	                             \
	ADDQ       $ 4*SIZE, CO1

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
	VMOVUPS__n16_BO__X0;    \
	VMOVDDUP__n12_BO__X1;   \
	VMOVDDUP__n11_BO__X2;   \
	VMOVDDUP__n10_BO__X3;   \
	VFMADD231PD__X0_X1_X4;  \
	VMOVDDUP__n9_BO__X1;    \
	VFMADD231PD__X0_X2_X5;  \
	VMOVDDUP__n8_BO__X2;    \
	VFMADD231PD__X0_X3_X6;  \
	VMOVDDUP__n7_BO__X3;    \
	VFMADD231PD__X0_X1_X7;  \
	VMOVDDUP__n6_BO__X1;    \
	VFMADD231PD__X0_X2_X8;  \
	VMOVDDUP__n5_BO__X2;    \
	VFMADD231PD__X0_X3_X9;  \
	VMOVDDUP__n4_BO__X3;    \
	VFMADD231PD__X0_X1_X10; \
	VMOVDDUP__n3_BO__X1;    \
	VFMADD231PD__X0_X2_X11; \
	VMOVDDUP__n2_BO__X2;    \
	VFMADD231PD__X0_X3_X12; \
	VMOVDDUP__n1_BO__X3;    \
	VFMADD231PD__X0_X1_X13; \
	ADDQ $ 12*SIZE, AO;     \
	VFMADD231PD__X0_X2_X14; \
	ADDQ $ 2*SIZE, BO;      \
	VFMADD231PD__X0_X3_X15

#define SAVE2x12 \
	VMOVDDUP__ALPHA_X0;         \
	                            \
	LEAQ (CO1)(LDC*2), AX;      \
	                            \
	VFMADD213PD_CO1__X0_X4;     \
	VFMADD213PD_CO1_LDC__X0_X5; \
	VFMADD213PD_AX__X0_X6;      \
	VFMADD213PD_AX_LDC__X0_X7;  \
	                            \
	VMOVUPS__X4_CO1;            \
	VMOVUPS__X5_CO1_LDC;        \
	VMOVUPS__X6_AX;             \
	VMOVUPS__X7_AX_LDC;         \
	                            \
	LEAQ (AX)(LDC*2), AX;       \
	LEAQ (AX)(LDC*2), BP;       \
	                            \
	VFMADD213PD_AX__X0_X8;      \
	VFMADD213PD_AX_LDC__X0_X9;  \
	VFMADD213PD_BP__X0_X10;     \
	VFMADD213PD_BP_LDC__X0_X11; \
	                            \
	VMOVUPS__X8_AX;             \
	VMOVUPS__X9_AX_LDC;         \
	VMOVUPS__X10_BP;            \
	VMOVUPS__X11_BP_LDC;        \
	                            \
	LEAQ (AX)(LDC*4), AX;       \
	LEAQ (BP)(LDC*4), BP;       \
	                            \
	VFMADD213PD_AX__X0_X12;     \
	VFMADD213PD_AX_LDC__X0_X13; \
	VFMADD213PD_BP__X0_X14;     \
	VFMADD213PD_BP_LDC__X0_X15; \
	                            \
	VMOVUPS__X12_AX;            \
	VMOVUPS__X13_AX_LDC;        \
	VMOVUPS__X14_BP;            \
	VMOVUPS__X15_BP_LDC;        \
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
	VMOVSD -16 * SIZE(BO), X0; \
	VMOVSD -12 * SIZE(AO), X1; \
	VMOVSD -11 * SIZE(AO), X2; \
	VMOVSD -10 * SIZE(AO), X3; \
	VFMADD231SD__X0_X1_X4;     \
	VMOVSD -9 * SIZE(AO), X1;  \
	VFMADD231SD__X0_X2_X5;     \
	VMOVSD -8 * SIZE(AO), X2;  \
	VFMADD231SD__X0_X3_X6;     \
	VMOVSD -7 * SIZE(AO), X3;  \
	VFMADD231SD__X0_X1_X7;     \
	VMOVSD -6 * SIZE(AO), X1;  \
	VFMADD231SD__X0_X2_X8;     \
	VMOVSD -5 * SIZE(AO), X2;  \
	VFMADD231SD__X0_X3_X9;     \
	VMOVSD -4 * SIZE(AO), X3;  \
	VFMADD231SD__X0_X1_X10;    \
	VMOVSD -3 * SIZE(AO), X1;  \
	VFMADD231SD__X0_X2_X11;    \
	VMOVSD -2 * SIZE(AO), X2;  \
	VFMADD231SD__X0_X3_X12;    \
	VMOVSD -1 * SIZE(AO), X3;  \
	VFMADD231SD__X0_X1_X13;    \
	ADDQ   $ 12*SIZE, AO;      \
	VFMADD231SD__X0_X2_X14;    \
	ADDQ   $ 1*SIZE, BO;       \
	VFMADD231SD__X0_X3_X15

#define SAVE1x12 \
	VMOVSD ALPHA, X0;           \
	                            \
	LEAQ   (CO1)(LDC*2), AX;    \
	                            \
	VFMADD213SD_CO1__X0_X4;     \
	VFMADD213SD_CO1_LDC__X0_X5; \
	VFMADD213SD_AX__X0_X6;      \
	VFMADD213SD_AX_LDC__X0_X7;  \
	                            \
	VMOVSD X4, (CO1);           \
	VMOVSD X5, (CO1)(LDC*1);    \
	VMOVSD X6, (AX);            \
	VMOVSD X7, (AX)(LDC*1);     \
	                            \
	LEAQ   (AX)(LDC*2), AX;     \
	LEAQ   (AX)(LDC*2), BP;     \
	                            \
	VFMADD213SD_AX__X0_X8;      \
	VFMADD213SD_AX_LDC__X0_X9;  \
	VFMADD213SD_BP__X0_X10;     \
	VFMADD213SD_BP_LDC__X0_X11; \
	                            \
	VMOVSD X8, (AX);            \
	VMOVSD X9, (AX)(LDC*1);     \
	VMOVSD X10, (BP);           \
	VMOVSD X11, (BP)(LDC*1);    \
	                            \
	LEAQ   (AX)(LDC*4), AX;     \
	LEAQ   (BP)(LDC*4), BP;     \
	                            \
	VFMADD213SD_AX__X0_X12;     \
	VFMADD213SD_AX_LDC__X0_X13; \
	VFMADD213SD_BP__X0_X14;     \
	VFMADD213SD_BP_LDC__X0_X15; \
	                            \
	VMOVSD X12, (AX);           \
	VMOVSD X13, (AX)(LDC*1);    \
	VMOVSD X14, (BP);           \
	VMOVSD X15, (BP)(LDC*1);    \
	                            \
	ADDQ   $ 1*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT4x4 \
	VXORPD Y4, Y4, Y4; \
	VXORPD Y5, Y5, Y5; \
	VXORPD Y6, Y6, Y6; \
	VXORPD Y7, Y7, Y7

#define KERNEL4x4_I \
	PREFETCHT0 A_PR1(BO);    \
	VMOVUPS__n12_AO__Y1;     \
	VMOVUPS__n16_BO__Y0;     \
	VMULPD__Y0_Y1_Y4;        \
	VPERMPD__0xb1_Y0_Y0;     \
	VMULPD__Y0_Y1_Y5;        \
	VPERMPD__0x1b_Y0_Y0;     \
	VMULPD__Y0_Y1_Y6;        \
	                         \
	ADDQ       $ 4*SIZE, AO; \
	VPERMPD__0xb1_Y0_Y0;     \
	VMULPD__Y0_Y1_Y7;        \
	VMOVUPS__n12_AO__Y1

#define KERNEL4x4_M1 \
	PREFETCHT0 A_PR1(BO);  \
	VMOVUPS__n16_BO__Y0;   \
	VFMADD231PD__Y0_Y1_Y4; \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y5; \
	VPERMPD__0x1b_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y6; \
	                       \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y7; \
	VMOVUPS__n12_AO__Y1

#define KERNEL4x4_M2 \
	VMOVUPS__n12_BO__Y0;   \
	VFMADD231PD__Y0_Y1_Y4; \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y5; \
	VPERMPD__0x1b_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y6; \
	                       \
	ADDQ $ 8*SIZE, BO;     \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y7; \
	VMOVUPS__n8_AO__Y1;    \
	ADDQ $ 8*SIZE, AO

#define KERNEL4x4_E \
	VMOVUPS__n12_BO__Y0;   \
	VFMADD231PD__Y0_Y1_Y4; \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y5; \
	VPERMPD__0x1b_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y6; \
	                       \
	ADDQ $ 8*SIZE, BO;     \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y7; \
	ADDQ $ 4*SIZE, AO

#define KERNEL4x4_SUB \
	VMOVUPS__n12_AO__Y1;   \
	VMOVUPS__n16_BO__Y0;   \
	VFMADD231PD__Y0_Y1_Y4; \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y5; \
	ADDQ $ 4*SIZE, AO;     \
	VPERMPD__0x1b_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y6; \
	ADDQ $ 4*SIZE, BO;     \
	VPERMPD__0xb1_Y0_Y0;   \
	VFMADD231PD__Y0_Y1_Y7

#define SAVE4x4 \
	VBROADCASTSD__ALPHA_Y0;     \
	                            \
	VUNPCKLPD_Y4_Y5_Y2;         \
	VUNPCKHPD_Y5_Y4_Y3;         \
	VUNPCKLPD_Y6_Y7_Y4;         \
	VUNPCKHPD_Y7_Y6_Y5;         \
	                            \
	VPERM2F128_0x31_Y4_Y2_Y6;   \
	VPERM2F128_0x31_Y5_Y3_Y7;   \
	VPERM2F128_0x20_Y2_Y4_Y4;   \
	VPERM2F128_0x20_Y3_Y5_Y5;   \
	                            \
	LEAQ (CO1)(LDC*2), AX;      \
	                            \
	VFMADD213PD_CO1__Y0_Y4;     \
	VFMADD213PD_CO1_LDC__Y0_Y5; \
	VFMADD213PD_AX__Y0_Y6;      \
	VFMADD213PD_AX_LDC__Y0_Y7;  \
	                            \
	VMOVUPS__Y4_CO1;            \
	VMOVUPS__Y5_CO1_LDC;        \
	VMOVUPS__Y6_AX;             \
	VMOVUPS__Y7_AX_LDC;         \
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
	VMOVDDUP__n12_BO__X1;  \
	VMOVUPS__n16_BO__X0;   \
	VMOVDDUP__n11_BO__X2;  \
	VFMADD231PD__X0_X1_X4; \
	VMOVDDUP__n10_BO__X3;  \
	VFMADD231PD__X0_X2_X5; \
	VMOVDDUP__n9_BO__X8;   \
	VFMADD231PD__X0_X3_X6; \
	ADDQ $ 4*SIZE, AO;     \
	VFMADD231PD__X0_X8_X7; \
	ADDQ $ 2*SIZE, BO

#define SAVE2x4 \
	VMOVDDUP__ALPHA_X0;         \
	                            \
	LEAQ (CO1)(LDC*2), AX;      \
	                            \
	VFMADD213PD_CO1__X0_X4;     \
	VFMADD213PD_CO1_LDC__X0_X5; \
	VFMADD213PD_AX__X0_X6;      \
	VFMADD213PD_AX_LDC__X0_X7;  \
	                            \
	VMOVUPS__X4_CO1;            \
	VMOVUPS__X5_CO1_LDC;        \
	VMOVUPS__X6_AX;             \
	VMOVUPS__X7_AX_LDC;         \
	VMOVUPS__X7_AX_LDC;         \
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
	VMOVSD -12 * SIZE(AO), X1; \
	VMOVSD -16 * SIZE(BO), X0; \
	VMOVSD -11 * SIZE(AO), X2; \
	VFMADD231SD__X0_X1_X4;     \
	VMOVSD -10 * SIZE(AO), X3; \
	VFMADD231SD__X0_X2_X5;     \
	VMOVSD -9 * SIZE(AO), X8;  \
	VFMADD231SD__X0_X3_X6;     \
	ADDQ   $ 4*SIZE, AO;       \
	VFMADD231SD__X0_X8_X7;     \
	ADDQ   $ 1*SIZE, BO

#define SAVE1x4 \
	VMOVSD ALPHA, X0;           \
	                            \
	LEAQ   (CO1)(LDC*2), AX;    \
	                            \
	VFMADD213SD_CO1__X0_X4;     \
	VFMADD213SD_CO1_LDC__X0_X5; \
	VFMADD213SD_AX__X0_X6;      \
	VFMADD213SD_AX_LDC__X0_X7;  \
	                            \
	VMOVSD X4, (CO1);           \
	VMOVSD X5, (CO1)(LDC*1);    \
	VMOVSD X6, (AX);            \
	VMOVSD X7, (AX)(LDC*1);     \
	                            \
	ADDQ   $ 1*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT4x2 \
	VXORPD X4, X4, X4; \
	VXORPD X5, X5, X5; \
	VXORPD X6, X6, X6; \
	VXORPD X7, X7, X7

#define KERNEL4x2_SUB \
	VMOVDDUP__n12_BO__X2;  \
	VMOVUPS__n16_BO__X0;   \
	VMOVUPS__n14_BO__X1;   \
	VMOVDDUP__n11_BO__X3;  \
	VFMADD231PD__X0_X2_X4; \
	VFMADD231PD__X1_X2_X5; \
	VFMADD231PD__X0_X3_X6; \
	VFMADD231PD__X1_X3_X7; \
	ADDQ $ 2*SIZE, AO;     \
	ADDQ $ 4*SIZE, BO

#define SAVE4x2 \
	VMOVDDUP__ALPHA_X0;           \
	                              \
	VFMADD213PD_CO1__X0_X4;       \
	VFMADD213PD_2_CO1__X0_X5;     \
	VFMADD213PD_CO1_LDC__X0_X6;   \
	VFMADD213PD_2_CO1_LDC__X0_X7; \
	                              \
	VMOVUPS__X4_CO1;              \
	VMOVUPS__X5__2_CO1;           \
	VMOVUPS__X6_CO1_LDC;          \
	VMOVUPS__X7__2_CO1_LDC;       \
	                              \
	ADDQ $ 4*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT2x2 \
	VXORPD X4, X4, X4; \
	VXORPD X6, X6, X6

#define KERNEL2x2_SUB \
	VMOVDDUP__n12_BO__X2;  \
	VMOVUPS__n16_BO__X0;   \
	VMOVDDUP__n11_BO__X3;  \
	VFMADD231PD__X0_X2_X4; \
	VFMADD231PD__X0_X3_X6; \
	ADDQ $ 2*SIZE, AO;     \
	ADDQ $ 2*SIZE, BO

#define SAVE2x2 \
	VMOVDDUP__ALPHA_X0;         \
	                            \
	VFMADD213PD_CO1__X0_X4;     \
	VFMADD213PD_CO1_LDC__X0_X6; \
	                            \
	VMOVUPS__X4_CO1;            \
	VMOVUPS__X6_CO1_LDC;        \
	                            \
	ADDQ $ 2*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT1x2 \
	VXORPD X4, X4, X4; \
	VXORPD X5, X5, X5

#define KERNEL1x2_SUB \
	VMOVSD -12 * SIZE(AO), X1; \
	VMOVSD -16 * SIZE(BO), X0; \
	VMOVSD -11 * SIZE(AO), X2; \
	VFMADD231SD__X0_X1_X4;     \
	VFMADD231SD__X0_X2_X5;     \
	ADDQ   $ 2*SIZE, AO;       \
	ADDQ   $ 1*SIZE, BO

#define SAVE1x2 \
	VMOVSD ALPHA, X0;           \
	                            \
	VFMADD213SD_CO1__X0_X4;     \
	VFMADD213SD_CO1_LDC__X0_X5; \
	                            \
	VMOVSD X4, (CO1);           \
	VMOVSD X5, (CO1)(LDC*1);    \
	                            \
	ADDQ   $ 1*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT4x1 \
	VXORPD Y4, Y4, Y4; \
	VXORPD Y5, Y5, Y5; \
	VXORPD Y6, Y6, Y6; \
	VXORPD Y7, Y7, Y7

#define KERNEL4x1 \
	VBROADCASTSD__n12_BO__Y0;   \
	VBROADCASTSD__n11_BO__Y1;   \
	VBROADCASTSD__n10_BO__Y2;   \
	VBROADCASTSD__n9_BO__Y3;    \
	                            \
	VFMADD231PD__n16_AO__Y0_Y4; \
	VFMADD231PD__n12_AO__Y1_Y5; \
	                            \
	VBROADCASTSD__n8_BO__Y0;    \
	VBROADCASTSD__n7_BO__Y1;    \
	                            \
	VFMADD231PD__n8_AO__Y2_Y6;  \
	VFMADD231PD__n4_AO__Y3_Y7;  \
	                            \
	VBROADCASTSD__n6_BO__Y2;    \
	VBROADCASTSD__n5_BO__Y3;    \
	                            \
	VFMADD231PD__0_AO__Y0_Y4;   \
	VFMADD231PD__4_AO__Y1_Y5;   \
	VFMADD231PD__8_AO__Y2_Y6;   \
	VFMADD231PD__12_AO__Y3_Y7;  \
	                            \
	ADDQ $ 8 *SIZE, AO;         \
	ADDQ $ 32*SIZE, BO

#define KERNEL4x1_SUB \
	VBROADCASTSD__n12_BO__Y2; \
	VMOVUPS__n16_BO__Y0;      \
	VFMADD231PD__Y0_Y2_Y4;    \
	ADDQ $ 1*SIZE, AO;        \
	ADDQ $ 4*SIZE, BO

#define SAVE4x1 \
	VBROADCASTSD__ALPHA_Y0; \
	                        \
	VADDPD__Y4_Y5_Y4;       \
	VADDPD__Y6_Y7_Y6;       \
	VADDPD__Y4_Y6_Y4;       \
	                        \
	VFMADD213PD_CO1__Y0_Y4; \
	                        \
	VMOVUPS__Y4_CO1;        \
	                        \
	ADDQ $ 4*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT2x1 \
	VXORPD X4, X4, X4

#define KERNEL2x1_SUB \
	VMOVDDUP__n12_BO__X2;  \
	VMOVUPS__n16_BO__X0;   \
	VFMADD231PD__X0_X2_X4; \
	ADDQ $ 1*SIZE, AO;     \
	ADDQ $ 2*SIZE, BO

#define SAVE2x1 \
	VMOVDDUP__ALPHA_X0;     \
	                        \
	VFMADD213PD_CO1__X0_X4; \
	                        \
	VMOVUPS__X4_CO1;        \
	                        \
	ADDQ $ 2*SIZE, CO1

// ****************************************************************************************
// ****************************************************************************************

#define INIT1x1 \
	VXORPD X4, X4, X4

#define KERNEL1x1_SUB \
	VMOVSD -12 * SIZE(AO), X1; \
	VMOVSD -16 * SIZE(BO), X0; \
	VFMADD231SD__X0_X1_X4;     \
	ADDQ   $ 1*SIZE, AO;       \
	ADDQ   $ 1*SIZE, BO

#define SAVE1x1 \
	VMOVSD ALPHA, X0;       \
	                        \
	VFMADD213PD_CO1__X0_X4; \
	                        \
	VMOVSD X4, (CO1);       \
	                        \
	ADDQ   $ 1*SIZE, CO1

// *****************************************************************************************
