#define MOVDDUP_(val, reg) \
	MOVSD  val, reg; \
	MOVHPD val, reg

	// VFMADD231PD SCRATCH3, SCRATCH1, RES01
#define VFMADD231PD_X3_X1_X4 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xF1; BYTE $0xB8; BYTE $0xE3

	// VFMADD231PD SCRATCH3, SCRATCH2, RES23
#define VFMADD231PD_X3_X2_X5 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xE9; BYTE $0xB8; BYTE $0xEB

	// VFMADD213PD (CO1), ALPHA, RES01
#define VFMADD213PD_CO1_X0_X4 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xF9; BYTE $0xA8; BYTE $0x27

	// VFMADD213PD (CO2), ALPHA, RES23
#define VFMADD213PD_CO2_X0_X5 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xF9; BYTE $0xA8; BYTE $0x2A

	// VFMADD231SD SCRATCH3, SCRATCH1, RES01
#define VFMADD231SD_X3_X1_X4 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xF1; BYTE $0xB9; BYTE $0xE3

	// VFMADD231SD SCRATCH3, SCRATCH2, RES23
#define VFMADD231SD_X3_X2_X5 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xE9; BYTE $0xB9; BYTE $0xEB

	// VFMADD213SD (CO1), ALPHA, RES01
#define VFMADD213SD_CO1_X0_X4 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xF9; BYTE $0xA9; BYTE $0x27

	// VFMADD213SD (CO2), ALPHA, RES23
#define VFMADD213SD_CO2_X0_X5 \
	BYTE $0xC4; BYTE $0xE2; BYTE $0xF9; BYTE $0xA9; BYTE $0x2A

#define KERNEL_2X2_INIT \
	XORPD RES01, RES01; \
	XORPD RES23, RES23

#define KERNEL_2X2_(offset) \
	MOVDDUP_(offset*SIZE(AO1), X1);     \
	MOVDDUP_((offset+1)*SIZE(AO1), X2); \
	MOVUPS offset*SIZE(BO1), X3;        \
	VFMADD231PD_X3_X1_X4;               \
	VFMADD231PD_X3_X2_X5

#define KERNEL_2X2_SAVE \
	VFMADD213PD_CO1_X0_X4; \
	VFMADD213PD_CO2_X0_X5; \
	MOVUPS RES01, (CO1);   \
	MOVUPS RES23, (CO2);   \
	ADDQ   $2*SIZE, CO1;   \
	ADDQ   $2*SIZE, CO2

#define KERNEL_1X2_INIT \
	XORPD RES01, RES01; \
	XORPD RES23, RES23

#define KERNEL_1X2 \
	MOVSD (AO1), X1;      \
	MOVSD SIZE(AO1), X2;  \
	MOVSD (BO1), X3;      \
	VFMADD231SD_X3_X1_X4; \
	VFMADD231SD_X3_X2_X5; \
	ADDQ  $1*SIZE, BO1;   \
	ADDQ  $2*SIZE, AO1

#define KERNEL_1X2_SAVE \
	VFMADD213SD_CO1_X0_X4; \
	VFMADD213SD_CO2_X0_X5; \
	MOVSD RES01, (CO1);    \
	MOVSD RES23, (CO2);    \
	ADDQ  $1*SIZE, CO1;    \
	ADDQ  $1*SIZE, CO2

#define KERNEL_2X1_INIT \
	XORPD RES01, RES01

#define KERNEL_2X1 \
	MOVDDUP_((AO1), X1);  \
	MOVUPS (BO1), X3;     \
	VFMADD231PD_X3_X1_X4; \
	ADDQ   $2*SIZE, BO1;  \
	ADDQ   $1*SIZE, AO1

#define KERNEL_2X1_SAVE \
	VFMADD213PD_CO1_X0_X4; \
	MOVUPS RES01, (CO1);   \
	ADDQ   $2*SIZE, CO1

#define KERNEL_1X1_INIT \
	XORPD RES01, RES01

#define KERNEL_1X1 \
	MOVSD (AO1), X1;      \
	MOVSD (BO1), X3;      \
	VFMADD231SD_X3_X1_X4; \
	ADDQ  $1*SIZE, BO1;   \
	ADDQ  $1*SIZE, AO1

#define KERNEL_1X1_SAVE \
	VFMADD213SD_CO1_X0_X4; \
	MOVSD RES01, (CO1);    \
	ADDQ  $1*SIZE, CO1
