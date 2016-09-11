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

func gemmTcopy_2(m, n int, a []float64, lda int, b []float64) {
	var (
		i, j                  int
		a_off, a_off1, a_off2 int
		b_off, b_off1         int
		b_off2                = m * (n & ^1)
	)

	for i = m >> 1; i > 0; i-- {
		a_off1 = a_off
		a_off2 = a_off + lda
		a_off += 2 * lda

		b_off1 = b_off
		b_off += 4

		for j = n >> 1; j > 0; j-- {
			b[b_off1+0] = a[a_off1+0]
			b[b_off1+1] = a[a_off1+1]
			b[b_off1+2] = a[a_off2+0]
			b[b_off1+3] = a[a_off2+1]
			a_off1 += 2
			a_off2 += 2
			b_off1 += m * 2
		}

		if n&1 > 0 {
			b[b_off2+0] = a[a_off1+0]
			b[b_off2+1] = a[a_off2+0]
			b_off2 += 2
		}
	}

	if m&1 > 0 {
		for j = n >> 1; j > 0; j-- {
			b[b_off+0] = a[a_off+0]
			b[b_off+1] = a[a_off+1]
			a_off += 2
			b_off += m * 2
		}

		if n&1 > 0 {
			b[b_off2+0] = a[a_off+0]
		}
	}
}
