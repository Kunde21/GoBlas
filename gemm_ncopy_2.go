package main

/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

func gemmNcopy_2(col, row int, src []float64, lda int, dst []float64) {
	var (
		i, j                        int
		src_off, src_off1, src_off2 int
		dst_off                     int
	)
	for j = row >> 1; j > 0; j-- {
		src_off1 = src_off
		src_off2 = src_off + lda
		src_off += 2 * lda

		for i = col >> 2; i > 0; i-- {
			dst[dst_off+0] = src[src_off1+0]
			dst[dst_off+1] = src[src_off2+0]
			dst[dst_off+2] = src[src_off1+1]
			dst[dst_off+3] = src[src_off2+1]
			dst[dst_off+4] = src[src_off1+2]
			dst[dst_off+5] = src[src_off2+2]
			dst[dst_off+6] = src[src_off1+3]
			dst[dst_off+7] = src[src_off2+3]
			src_off1 += 4
			src_off2 += 4
			dst_off += 8
		}
		for i = col & 3; i > 0; i-- {
			dst[dst_off+0] = src[src_off1+0]
			dst[dst_off+1] = src[src_off2+0]
			src_off1++
			src_off2++
			dst_off += 2
		}
	}

	if (row & 1) > 0 {
		for i = col >> 3; i > 0; i-- {
			dst[dst_off+0] = src[src_off+0]
			dst[dst_off+1] = src[src_off+1]
			dst[dst_off+2] = src[src_off+2]
			dst[dst_off+3] = src[src_off+3]
			dst[dst_off+4] = src[src_off+4]
			dst[dst_off+5] = src[src_off+5]
			dst[dst_off+6] = src[src_off+6]
			dst[dst_off+7] = src[src_off+7]
			src_off += 8
			dst_off += 8
		}
		for i = col & 7; i > 0; i-- {
			dst[dst_off+0] = src[src_off+0]
			src_off++
			dst_off++
		}
	}
}
