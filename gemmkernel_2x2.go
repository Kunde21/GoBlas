package main

func gemmKernel_2x2(bm, bn, bk int, alpha float64, ba, bb, C []float64, ldc int) {
	var i, j, k int
	// var C0, C1, ptrba, ptrbb []float64

	c_off, c_off1, c_off2 := 0, 0, 0
	a_off, a_off1 := 0, 0
	b_off, b_off1 := 0, 0

	var res0, res1, res2, res3, load0, load1, load2, load3 float64

	// d2
	for j = 0; j < bn/2; j += 1 {
		c_off1, c_off2 = c_off, c_off+ldc
		a_off1 = a_off

		// d22
		for i = 0; i < bm/2; i += 1 {
			b_off1 = b_off

			res0 = 0
			res1 = 0
			res2 = 0
			res3 = 0
			// d22_88
			for k = 0; k < bk/4; k += 1 {
				load0 = ba[a_off1+2*0+0]
				load1 = bb[b_off1+2*0+0]
				res0 = res0 + load0*load1
				load2 = ba[a_off1+2*0+1]
				res1 = res1 + load2*load1
				load3 = bb[b_off1+2*0+1]
				res2 = res2 + load0*load3
				res3 = res3 + load2*load3

				load0 = ba[a_off1+2*1+0]
				load1 = bb[b_off1+2*1+0]
				res0 = res0 + load0*load1
				load2 = ba[a_off1+2*1+1]
				res1 = res1 + load2*load1
				load3 = bb[b_off1+2*1+1]
				res2 = res2 + load0*load3
				res3 = res3 + load2*load3

				load0 = ba[a_off1+2*2+0]
				load1 = bb[b_off1+2*2+0]
				res0 = res0 + load0*load1
				load2 = ba[a_off1+2*2+1]
				res1 = res1 + load2*load1
				load3 = bb[b_off1+2*2+1]
				res2 = res2 + load0*load3
				res3 = res3 + load2*load3

				load0 = ba[a_off1+2*3+0]
				load1 = bb[b_off1+2*3+0]
				res0 = res0 + load0*load1
				load2 = ba[a_off1+2*3+1]
				res1 = res1 + load2*load1
				load3 = bb[b_off1+2*3+1]
				res2 = res2 + load0*load3
				res3 = res3 + load2*load3

				a_off1 += 8
				b_off1 += 8
			} // end d22_8_8
			// d22_2_2
			for k = 0; k < bk&3; k += 1 {
				load0 = ba[a_off1+2*0+0]
				load1 = bb[b_off1+2*0+0]
				res0 = res0 + load0*load1
				load2 = ba[a_off1+2*0+1]
				res1 = res1 + load2*load1
				load3 = bb[b_off1+2*0+1]
				res2 = res2 + load0*load3
				res3 = res3 + load2*load3
				a_off1 += 2
				b_off1 += 2
			} // end d22_2_2
			// d22_store
			res0 = res0 * alpha
			C[c_off1+0] += res0
			res1 = res1 * alpha
			C[c_off1+1] += res1
			res2 = res2 * alpha
			C[c_off2+0] += res2
			res3 = res3 * alpha
			C[c_off2+1] += res3
			c_off1 += 2
			c_off2 += 2
		} // end d22

		// d21
		for i = 0; i < bm&1; i += 1 {
			b_off1 = b_off
			res0 = 0
			res1 = 0
			// d21_1_2
			for k = 0; k < bk; k += 1 {
				load0 = ba[a_off1+0+0]
				load1 = bb[b_off1+2*0+0]
				res0 = res0 + load0*load1
				load2 = bb[b_off1+2*0+1]
				res1 = res1 + load0*load2
				a_off1 += 1
				b_off1 += 2
			}
			// d21_1_2_save
			res0 = res0 * alpha
			C[c_off1+0] += res0
			res1 = res1 * alpha
			C[c_off2+0] += res1
			c_off1 += 1
			c_off2 += 1
		} // end d21
		k = bk << 1
		b_off += k
		i = ldc << 1
		c_off += i
	} // end d2

	// d1
	for j = 0; j < bn&1; j += 1 {
		c_off1 = c_off
		a_off1 = a_off
		// d12
		for i = 0; i < bm/2; i += 1 {
			b_off1 = b_off
			res0, res1 = 0, 0
			// d12_2_1
			for k = 0; k < bk; k += 1 {
				load0 = ba[a_off1+2*0+0]
				load1 = bb[b_off1+0+0]
				res0 = res0 + load0*load1
				load2 = ba[a_off1+2*0+1]
				res1 = res1 + load2*load1
				a_off1 += 2
				b_off1 += 1
			}
			res0 = res0 * alpha
			C[c_off1+0] += res0
			res1 = res1 * alpha
			C[c_off1+1] += res1
			c_off1 += 2
		}
		// d11
		for i = 0; i < bm&1; i += 1 {
			b_off1 = b_off
			res0 = 0
			// d11_1_1
			for k = 0; k < bk; k += 1 {
				load0 = ba[a_off1+0+0]
				load1 = bb[b_off1+0+0]
				res0 = res0 + load0*load1
				a_off1 += 1
				b_off1 += 1
			}
			res0 = res0 * alpha
			C[c_off1+0] += res0
			c_off1 += 1
		}
		k = bk << 0
		b_off += k
		c_off += ldc
	}
}
