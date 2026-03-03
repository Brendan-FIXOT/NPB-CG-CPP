/*
MIT License

Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
        GMAP website: https://gmap.pucrs.br

        Pontifical Catholic University of Rio Grande do Sul (PUCRS)
        Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------

The original NPB 3.4.1 version was written in Fortran and belongs to:
        http://www.nas.nasa.gov/Software/NPB/

Authors of the Fortran code:
        M. Yarrow
        C. Kuszmaul

------------------------------------------------------------------------------

The serial C++ version is a translation of the original NPB 3.4.1
Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER

Authors of the C++ code:
        Dalvan Griebler <dalvangriebler@gmail.com>
        Gabriell Araujo <hexenoften@gmail.com>
        Júnior Löff <loffjh@gmail.com>
*/

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * note: please observe that in the routine conj_grad three
 * implementations of the sparse matrix-vector multiply have
 * been supplied. the default matrix-vector multiply is not
 * loop unrolled. the alternate implementations are unrolled
 * to a depth of 2 and unrolled to a depth of 8. please
 * experiment with these to find the fastest for your particular
 * architecture. if reporting timing results, any of these three may
 * be used without penalty.
 * ---------------------------------------------------------------------
 * class specific parameters:
 * it appears here for reference only.
 * these are their values, however, this info is imported in the npbparams.h
 * include file, which is written by the sys/setparams.c program.
 * ---------------------------------------------------------------------
 */

#include <cmath>
#include <immintrin.h>
#include <vector>

static inline double hsum256_pd(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  __m128d sum = _mm_add_pd(lo, hi);
  __m128d shuf = _mm_shuffle_pd(sum, sum, 0x1);
  sum = _mm_add_sd(sum, shuf);
  return _mm_cvtsd_f64(sum);
}

static void *aligned_malloc_or_die(std::size_t alignment, std::size_t bytes) {
  void *p = nullptr;
  // posix_memalign requires alignment to be power-of-two and multiple of
  // sizeof(void*)
  int rc = posix_memalign(&p, alignment, bytes);
  if (rc != 0 || !p) {
    std::fprintf(stderr, "posix_memalign(%zu, %zu) failed (rc=%d)\n", alignment,
                 bytes, rc);
    std::abort();
  }
  return p;
}

#define NZ (NA * (NONZER + 1) * (NONZER + 1))
#define NAZ (NA * (NONZER + 1))
#define T_INIT 0
#define T_BENCH 1
#define T_CONJ_GRAD 2
#define T_LAST 3

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static int colidx[NZ];
static int rowstr[NA + 1];
static int iv[NA];
static int arow[NA];
static int acol[NAZ];
static double aelt[NAZ];
static double a[NZ];
static double x[NA + 2];
static double z[NA + 2];
static double p[NA + 2];
static double q[NA + 2];
static double r[NA + 2];
#else
// static int(*colidx) = (int *)malloc(sizeof(int) * (NZ));
// static int(*rowstr) = (int *)malloc(sizeof(int) * (NA + 1));
// static int(*iv) = (int *)malloc(sizeof(int) * (NA));
// static int(*arow) = (int *)malloc(sizeof(int) * (NA));
// static int(*acol) = (int *)malloc(sizeof(int) * (NAZ));
// static double(*aelt) = (double *)malloc(sizeof(double) * (NAZ));
// static double(*a) = (double *)malloc(sizeof(double) * (NZ));
// static double(*x) = (double *)malloc(sizeof(double) * (NA + 2));
// static double(*z) = (double *)malloc(sizeof(double) * (NA + 2));
// static double(*p) = (double *)malloc(sizeof(double) * (NA + 2));
// static double(*q) = (double *)malloc(sizeof(double) * (NA + 2));
// static double(*r) = (double *)malloc(sizeof(double) * (NA + 2));
static int *colidx = (int *)aligned_malloc_or_die(64, sizeof(int) * (NZ));
static int *rowstr = (int *)aligned_malloc_or_die(64, sizeof(int) * (NA + 1));
static int *iv = (int *)aligned_malloc_or_die(64, sizeof(int) * (NA));
static int *arow = (int *)aligned_malloc_or_die(64, sizeof(int) * (NA));
static int *acol = (int *)aligned_malloc_or_die(64, sizeof(int) * (NAZ));

static double *aelt =
    (double *)aligned_malloc_or_die(64, sizeof(double) * (NAZ));
static double *a = (double *)aligned_malloc_or_die(64, sizeof(double) * (NZ));
static double *x =
    (double *)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double *z =
    (double *)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double *p =
    (double *)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double *q =
    (double *)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double *r =
    (double *)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));

#endif
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;
static boolean timeron;

/* function prototypes */

// static void conj_grad(int colidx[],		int rowstr[],
// 		double x[],
// 		double z[],
// 		double a[],
// 		double p[],
// 		double q[],
// 		double r[],
// 		double* rnorm);
//
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifndef CG_ASSUME_ALIGNED
#  define CG_ASSUME_ALIGNED(ptr, A) (__typeof__(ptr))__builtin_assume_aligned((ptr), (A))
#endif

// Tuneable knobs (reasonable defaults)
#ifndef CG_CGITMAX
#  define CG_CGITMAX 25
#endif
#ifndef CG_PREFETCH_DIST
#  define CG_PREFETCH_DIST 48   // in "k" iterations (nonzeros) ahead
#endif

// Unrolled dot product (vector-friendly)
static inline double dot_u4(const double *__restrict a,
                            const double *__restrict b,
                            int n)
{
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;

  // Help auto-vectorizer: contiguous, no alias, fixed stride
  #pragma GCC ivdep
  for (; i + 3 < n; i += 4) {
    s0 += a[i+0] * b[i+0];
    s1 += a[i+1] * b[i+1];
    s2 += a[i+2] * b[i+2];
    s3 += a[i+3] * b[i+3];
  }
  double s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

static inline double dot_self_u4(const double *__restrict a, int n)
{
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;
  #pragma GCC ivdep
  for (; i + 3 < n; i += 4) {
    const double x0 = a[i+0], x1 = a[i+1], x2 = a[i+2], x3 = a[i+3];
    s0 += x0 * x0;
    s1 += x1 * x1;
    s2 += x2 * x2;
    s3 += x3 * x3;
  }
  double s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) s += a[i] * a[i];
  return s;
}

// CSR SpMV: q = A * p
static inline void spmv_csr(const int *__restrict rowstr,
                            const int *__restrict colidx,
                            const double *__restrict a,
                            const double *__restrict p,
                            double *__restrict q,
                            int nrow)
{
  for (int j = 0; j < nrow; ++j) {
    const int start = rowstr[j];
    const int end   = rowstr[j + 1];

    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int k = start;

    // Unroll by 4; prefetch future gathers
    #pragma GCC ivdep
    for (; k + 3 < end; k += 4) {
      const int pf = k + CG_PREFETCH_DIST;
      if (pf < end) __builtin_prefetch(&p[colidx[pf]], 0, 1);

      const int c0 = colidx[k+0];
      const int c1 = colidx[k+1];
      const int c2 = colidx[k+2];
      const int c3 = colidx[k+3];

      s0 += a[k+0] * p[c0];
      s1 += a[k+1] * p[c1];
      s2 += a[k+2] * p[c2];
      s3 += a[k+3] * p[c3];
    }

    double sum = (s0 + s1) + (s2 + s3);
    for (; k < end; ++k) {
      sum += a[k] * p[colidx[k]];
    }
    q[j] = sum;
  }
}

// Fused update: z += alpha*p; r -= alpha*q; rho = r.r
static inline double axpy_axmy_rho(const double *__restrict p,
                                  const double *__restrict q,
                                  double *__restrict z,
                                  double *__restrict r,
                                  double alpha,
                                  int n)
{
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;

  #pragma GCC ivdep
  for (; i + 3 < n; i += 4) {
    z[i+0] += alpha * p[i+0];
    z[i+1] += alpha * p[i+1];
    z[i+2] += alpha * p[i+2];
    z[i+3] += alpha * p[i+3];

    const double r0 = r[i+0] - alpha * q[i+0];
    const double r1 = r[i+1] - alpha * q[i+1];
    const double r2 = r[i+2] - alpha * q[i+2];
    const double r3 = r[i+3] - alpha * q[i+3];

    r[i+0] = r0; r[i+1] = r1; r[i+2] = r2; r[i+3] = r3;

    s0 += r0 * r0;
    s1 += r1 * r1;
    s2 += r2 * r2;
    s3 += r3 * r3;
  }

  double rho = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) {
    z[i] += alpha * p[i];
    const double ri = r[i] - alpha * q[i];
    r[i] = ri;
    rho += ri * ri;
  }
  return rho;
}

static inline void p_update(const double *__restrict r,
                            double *__restrict p,
                            double beta,
                            int n)
{
  int i = 0;
  #pragma GCC ivdep
  for (; i + 3 < n; i += 4) {
    p[i+0] = r[i+0] + beta * p[i+0];
    p[i+1] = r[i+1] + beta * p[i+1];
    p[i+2] = r[i+2] + beta * p[i+2];
    p[i+3] = r[i+3] + beta * p[i+3];
  }
  for (; i < n; ++i) p[i] = r[i] + beta * p[i];
}

static inline double residual_norm2(const double *__restrict x,
                                    const double *__restrict Az,
                                    int n)
{
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;
  #pragma GCC ivdep
  for (; i + 3 < n; i += 4) {
    const double d0 = x[i+0] - Az[i+0];
    const double d1 = x[i+1] - Az[i+1];
    const double d2 = x[i+2] - Az[i+2];
    const double d3 = x[i+3] - Az[i+3];
    s0 += d0 * d0;
    s1 += d1 * d1;
    s2 += d2 * d2;
    s3 += d3 * d3;
  }
  double s = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) {
    const double d = x[i] - Az[i];
    s += d * d;
  }
  return s;
}

// static void conj_grad(const int *__restrict colidx,
//                       const int *__restrict rowstr,
//                       const double *__restrict x,
//                       double *__restrict z,
//                       const double *__restrict a,
//                       double *__restrict p,
//                       double *__restrict q,
//                       double *__restrict r,
//                       double *__restrict rnorm)
// {
//   const int n    = lastcol - firstcol + 1;
//   const int nrow = lastrow - firstrow + 1;

//   // Alignment hints (keep them as local aliases to help the optimizer)
//   p = CG_ASSUME_ALIGNED(p, 64);
//   q = CG_ASSUME_ALIGNED(q, 64);
//   r = CG_ASSUME_ALIGNED(r, 64);
//   z = CG_ASSUME_ALIGNED(z, 64);
//   x = CG_ASSUME_ALIGNED(x, 64);
//   a = CG_ASSUME_ALIGNED(a, 64);

//   // init: q=0, z=0, r=x, p=r
//   {
//     int i = 0;
//     #pragma GCC ivdep
//     for (; i + 3 < n; i += 4) {
//       q[i+0] = 0.0; q[i+1] = 0.0; q[i+2] = 0.0; q[i+3] = 0.0;
//       z[i+0] = 0.0; z[i+1] = 0.0; z[i+2] = 0.0; z[i+3] = 0.0;
//       const double x0 = x[i+0], x1 = x[i+1], x2 = x[i+2], x3 = x[i+3];
//       r[i+0] = x0; r[i+1] = x1; r[i+2] = x2; r[i+3] = x3;
//       p[i+0] = x0; p[i+1] = x1; p[i+2] = x2; p[i+3] = x3;
//     }
//     for (; i < n; ++i) {
//       q[i] = 0.0;
//       z[i] = 0.0;
//       r[i] = x[i];
//       p[i] = x[i];
//     }
//   }

//   double rho = dot_self_u4(r, n);

//   for (int cgit = 1; cgit <= CG_CGITMAX; ++cgit) {
//     // q = A*p
//     spmv_csr(rowstr, colidx, a, p, q, nrow);

//     // d = p.q
//     const double d = dot_u4(p, q, n);

//     // alpha = rho / d
//     const double alpha = rho / d;
//     const double rho0  = rho;

//     // z,r update + rho recompute (fused)
//     rho = axpy_axmy_rho(p, q, z, r, alpha, n);

//     // beta = rho / rho0
//     const double beta = rho / rho0;

//     // p = r + beta*p
//     p_update(r, p, beta, n);
//   }

//   // Explicit residual: r = A*z (store in r), then ||x - r||
//   spmv_csr(rowstr, colidx, a, z, r, nrow);
//   const double rn2 = residual_norm2(x, r, n);
//   *rnorm = sqrt(rn2);
// }

#include <immintrin.h>
#include <cmath>

// static inline double hsum256_pd(__m256d v) {
//   __m128d lo = _mm256_castpd256_pd128(v);
//   __m128d hi = _mm256_extractf128_pd(v, 1);
//   __m128d sum = _mm_add_pd(lo, hi);
//   __m128d shuf = _mm_shuffle_pd(sum, sum, 0x1);
//   sum = _mm_add_sd(sum, shuf);
//   return _mm_cvtsd_f64(sum);
// }

static inline double dot_avx2(const double* __restrict a,
                              const double* __restrict b,
                              int n) {
  __m256d acc = _mm256_setzero_pd();
  int i = 0;
  for (; i + 3 < n; i += 4) {
    __m256d va = _mm256_load_pd(a + i);
    __m256d vb = _mm256_load_pd(b + i);
    acc = _mm256_fmadd_pd(va, vb, acc);
  }
  double s = hsum256_pd(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

static inline double dot_self_avx2(const double* __restrict a, int n) {
  __m256d acc = _mm256_setzero_pd();
  int i = 0;
  for (; i + 3 < n; i += 4) {
    __m256d va = _mm256_load_pd(a + i);
    acc = _mm256_fmadd_pd(va, va, acc);
  }
  double s = hsum256_pd(acc);
  for (; i < n; ++i) s += a[i] * a[i];
  return s;
}

// z += alpha*p ; r -= alpha*q ; return rho = r.r
static inline double update_z_r_rho_avx2(double* __restrict z,
                                         double* __restrict r,
                                         const double* __restrict p,
                                         const double* __restrict q,
                                         double alpha, int n) {
  __m256d aval = _mm256_set1_pd(alpha);
  __m256d acc  = _mm256_setzero_pd();

  int i = 0;
  for (; i + 3 < n; i += 4) {
    __m256d zv = _mm256_load_pd(z + i);
    __m256d pv = _mm256_load_pd(p + i);
    __m256d rv = _mm256_load_pd(r + i);
    __m256d qv = _mm256_load_pd(q + i);

    // z = z + alpha*p
    zv = _mm256_fmadd_pd(aval, pv, zv);
    _mm256_store_pd(z + i, zv);

    // r = r - alpha*q
    rv = _mm256_fnmadd_pd(aval, qv, rv); // rv = rv - aval*qv
    _mm256_store_pd(r + i, rv);

    // rho += r*r
    acc = _mm256_fmadd_pd(rv, rv, acc);
  }

  double rho = hsum256_pd(acc);
  for (; i < n; ++i) {
    z[i] += alpha * p[i];
    const double ri = r[i] - alpha * q[i];
    r[i] = ri;
    rho += ri * ri;
  }
  return rho;
}

// p = r + beta*p
static inline void update_p_avx2(double* __restrict p,
                                 const double* __restrict r,
                                 double beta, int n) {
  __m256d bval = _mm256_set1_pd(beta);
  int i = 0;
  for (; i + 3 < n; i += 4) {
    __m256d pv = _mm256_load_pd(p + i);
    __m256d rv = _mm256_load_pd(r + i);
    pv = _mm256_fmadd_pd(bval, pv, rv);
    _mm256_store_pd(p + i, pv);
  }
  for (; i < n; ++i) p[i] = r[i] + beta * p[i];
}



static void conj_grad(const int *__restrict colidx,
                      const int *__restrict rowstr, const double *__restrict
                      x, double *__restrict z, const double *__restrict a,
                      double *__restrict p, double *__restrict q,
                      double *__restrict r, double *__restrict rnorm) {
  int j, k;
  int cgit, cgitmax;
  double d, sum, rho, rho0, alpha, beta;

  const int n = lastcol - firstcol + 1;
  const int nrow = lastrow - firstrow + 1;

  cgitmax = 25;

  rho = 0.0;

  p = (double *)__builtin_assume_aligned(p, 64);
  q = (double *)__builtin_assume_aligned(q, 64);
  r = (double *)__builtin_assume_aligned(r, 64);
  z = (double *)__builtin_assume_aligned(z, 64);
  x = (const double *)__builtin_assume_aligned(x, 64);

  /* initialize the CG algorithm */
  for (j = 0; j < n; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
  }

  /*
   * --------------------------------------------------------------------
   * rho = r.r
   * now, obtain the norm of r: First, sum squares of r elements locally...
   * --------------------------------------------------------------------
   */
  for (j = 0; j < n; j++) {
    rho = rho + r[j] * r[j];
  }

  /* the conj grad iteration loop */
  for (cgit = 1; cgit <= cgitmax; cgit++) {
    /*
     * ---------------------------------------------------------------------
     * q = A.p
     * the partition submatrix-vector multiply: use workspace w
     * ---------------------------------------------------------------------
     *
     * note: this version of the multiply is actually (slightly: maybe %5)
     * faster on the sp2 on 16 nodes than is the unrolled-by-2 version
     * below. on the Cray t3d, the reverse is TRUE, i.e., the
     * unrolled-by-two version is some 10% faster.
     * the unrolled-by-8 version below is significantly faster
     * on the Cray t3d - overall speed of code is 1.5 times faster.
     */

 		// for(j = 0; j <nrow ; j++){
		 // 	sum = 0.0;
		 // 	for(k = rowstr[j]; k < rowstr[j+1]; k++){
		 // 		sum = sum + a[k]*p[colidx[k]];
		 // 	}
		 // 	q[j] = sum;
		 // }
		 for (j = 0; j < nrow; j++) {
    const int start = rowstr[j];
    const int end   = rowstr[j + 1];
    double sum = 0.0;

    const int PF = 32;  // un peu plus grand avec unroll

    int k = start;
    // #pragma GCC ivdep
    for (; k + 3 < end; k += 4) {
        int pf = k + PF;
        // if (pf < end)
        __builtin_prefetch(&colidx[pf], 0, 1);

        sum += a[k+0] * p[colidx[k+0]];
        sum += a[k+1] * p[colidx[k+1]];
        sum += a[k+2] * p[colidx[k+2]];
        sum += a[k+3] * p[colidx[k+3]];
    }
    for (; k < end; k++) {
        sum += a[k] * p[colidx[k]];
    }
    q[j] = sum;
}
		

    // for (j = 0; j < nrow; j++) {
    //   const int start = rowstr[j];
    //   const int end = rowstr[j + 1];
    //   int kk = start;
    //   double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
      
    //   #pragma GCC ivdep
    //   for (; kk + 3 < end; kk += 4) {
    //     s0 += a[kk] * p[colidx[kk]];
    //     s1 += a[kk + 1] * p[colidx[kk + 1]];
    //     s2 += a[kk + 2] * p[colidx[kk + 2]];
    //     s3 += a[kk + 3] * p[colidx[kk + 3]];
    //   }
    //   double sum0 = (s0 + s1) + (s2 + s3);
    //   for (; kk < end; kk++) {
    //     sum0 += a[kk] * p[colidx[kk]];
    //   }
    //   q[j] = sum0;

    // }

    /*
     * --------------------------------------------------------------------
     * obtain p.q
     * --------------------------------------------------------------------
     */
    d = 0.0;
    
    for (j = 0; j < n; j++) {
      d = d + p[j] * q[j];
    }

    /*
     * --------------------------------------------------------------------
     * obtain alpha = rho / (p.q)
     * -------------------------------------------------------------------
     */
    alpha = rho / d;

    /*
     * --------------------------------------------------------------------
     * save a temporary of rho
     * --------------------------------------------------------------------
     */
    rho0 = rho;

    /*
     * ---------------------------------------------------------------------
     * obtain z = z + alpha*p
     * and    r = r - alpha*q
     * ---------------------------------------------------------------------
     */
   // rho = 0.0;
   //  for (j = 0; j < n; j++) {
   //    z[j] = z[j] + alpha * p[j];
   //    r[j] = r[j] - alpha * q[j];
   //  }

   //  /*
   //   *
   // ---------------------------------------------------------------------
   //   * rho = r.r
   //   * now, obtain the norm of r: first, sum squares of r elements
   // locally...
   //   *
   // ---------------------------------------------------------------------
   //   */
   //  for (j = 0; j < n; j++) {
   //    rho = rho + r[j] * r[/j];
   //  }

    rho = 0.0;
#pragma GCC ivdep
    for (j = 0; j < n; j++) {
      z[j] += alpha * p[j];
      double rj = r[j] - alpha * q[j];
      r[j] = rj;
      rho += rj * rj;
    }

    /*
     * ---------------------------------------------------------------------
     * obtain beta
     * ---------------------------------------------------------------------
     */
    beta = rho / rho0;

    /*
     * ---------------------------------------------------------------------
     * p = r + beta*p
     * ---------------------------------------------------------------------
     */
    for (j = 0; j < n; j++) {
      p[j] = r[j] + beta * p[j];
    }
  } /* end of do cgit=1, cgitmax */

  /*
   * ---------------------------------------------------------------------
   * compute residual norm explicitly: ||r|| = ||x - A.z||
   * first, form A.z
   * the partition submatrix-vector multiply
   * ---------------------------------------------------------------------
   */
  sum = 0.0;
  for (j = 0; j < nrow; j++) {
    d = 0.0;
    for (k = rowstr[j]; k < rowstr[j + 1]; k++) {
      d = d + a[k] * z[colidx[k]];
    }
    r[j] = d;
  }

  /*
   * ---------------------------------------------------------------------
   * at this point, r contains A.z
   * ---------------------------------------------------------------------
   */
  for (j = 0; j < n; j++) {
    d = x[j] - r[j];
    sum = sum + d * d;
  }

  *rnorm = sqrt(sum);
}
static void conj_grad(const int *__restrict colidx,
                      const int *__restrict rowstr, const double *__restrict x,
                      double *__restrict z, const double *__restrict a,
                      double *__restrict p, double *__restrict q,
                      double *__restrict r, double *__restrict rnorm);
// static void conj_grad(const int *__restrict colidx,
//                       const int *__restrict rowstr, const double *__restrict x,
//                       double *__restrict z, const double *__restrict a,
//                       double *__restrict p, double *__restrict q,
//                       double *__restrict r, double *__restrict rnorm) {
//   const int n = lastcol - firstcol + 1;
//   const int nrow = lastrow - firstrow + 1;
//   const int cgitmax = 25;

//   // alignment assumptions (true with your aligned_malloc_or_die)
//   p = (double *)__builtin_assume_aligned(p, 64);
//   q = (double *)__builtin_assume_aligned(q, 64);
//   r = (double *)__builtin_assume_aligned(r, 64);
//   z = (double *)__builtin_assume_aligned(z, 64);
//   x = (const double *)__builtin_assume_aligned(x, 64);

//   const double *__restrict aa = a;
//   const int *__restrict cidx = colidx;
//   const int *__restrict rstr = rowstr;
//   double *__restrict pp = p;
//   double *__restrict qq = q;
//   double *__restrict rr = r;
//   double *__restrict zz = z;

//   // init
//   for (int j = 0; j < n; j++) {
//     qq[j] = 0.0;
//     zz[j] = 0.0;
//     rr[j] = x[j];
//     pp[j] = rr[j];
//   }

//   // rho = r.r
//   double rho = 0.0;
//   for (int j = 0; j < n; j++) {
//     rho += rr[j] * rr[j];
//   }

//   for (int cgit = 1; cgit <= cgitmax; cgit++) {

//     // q = A * p  (SpMV)
//     for (int j = 0; j < nrow; j++) {
//       const int start = rstr[j];
//       const int end = rstr[j + 1];

//       double sum = 0.0;

//       // mild unroll by 4 (often neutral, sometimes + a few %)
//       int k = start;
//       for (; k + 3 < end; k += 4) {
//         sum += aa[k + 0] * pp[cidx[k + 0]];
//         sum += aa[k + 1] * pp[cidx[k + 1]];
//         sum += aa[k + 2] * pp[cidx[k + 2]];
//         sum += aa[k + 3] * pp[cidx[k + 3]];
//       }
//       for (; k < end; k++) {
//         sum += aa[k] * pp[cidx[k]];
//       }

//       qq[j] = sum;
//     }

//     // d = p.q
//     double d = 0.0;
//     for (int j = 0; j < n; j++) {
//       d += pp[j] * qq[j];
//     }

//     const double alpha = rho / d;
//     const double rho0 = rho;

//     // fused: z += alpha*p ; r -= alpha*q ; rho = r.r
//     rho = 0.0;
// #pragma GCC ivdep
//     for (int j = 0; j < n; j++) {
//       zz[j] += alpha * pp[j];
//       const double rj = rr[j] - alpha * qq[j];
//       rr[j] = rj;
//       rho += rj * rj;
//     }

//     const double beta = rho / rho0;

//     // p = r + beta*p
// #pragma GCC ivdep
//     for (int j = 0; j < n; j++) {
//       pp[j] = rr[j] + beta * pp[j];
//     }
//   }

//   // r = A * z
//   for (int j = 0; j < nrow; j++) {
//     const int start = rstr[j];
//     const int end = rstr[j + 1];
//     double sum = 0.0;
//     int k = start;
//     for (; k + 3 < end; k += 4) {
//       sum += aa[k + 0] * zz[cidx[k + 0]];
//       sum += aa[k + 1] * zz[cidx[k + 1]];
//       sum += aa[k + 2] * zz[cidx[k + 2]];
//       sum += aa[k + 3] * zz[cidx[k + 3]];
//     }
//     for (; k < end; k++) {
//       sum += aa[k] * zz[cidx[k]];
//     }
//     rr[j] = sum;
//   }

//   // rnorm = ||x - r||
//   double s = 0.0;
// #pragma GCC ivdep
//   for (int j = 0; j < n; j++) {
//     const double diff = x[j] - rr[j];
//     s += diff * diff;
//   }
//   *rnorm = std::sqrt(s);
// }
// static void conj_grad(const int *__restrict colidx,
//                       const int *__restrict rowstr,
//                       const double *__restrict x,
//                       double *__restrict z,
//                       const double *__restrict a,
//                       double *__restrict p,
//                       double *__restrict q,
//                       double *__restrict r,
//                       double *__restrict rnorm)
// {
//     int j, k;
//     int cgit, cgitmax;
//     double d, sum, rho, rho0, alpha, beta;

//     const int n    = lastcol - firstcol + 1;
//     const int nrow = lastrow - firstrow + 1;

//     cgitmax = 25;
//     rho = 0.0;

//     p = (double *)__builtin_assume_aligned(p, 64);
//     q = (double *)__builtin_assume_aligned(q, 64);
//     r = (double *)__builtin_assume_aligned(r, 64);
//     z = (double *)__builtin_assume_aligned(z, 64);
//     x = (const double *)__builtin_assume_aligned(x, 64);

//     /* initialize the CG algorithm */
//     for (j = 0; j < n; j++) {
//         q[j] = 0.0;
//         z[j] = 0.0;
//         r[j] = x[j];
//         p[j] = r[j];
//     }

//     /*
//      * rho = r.r
//      */
//     for (j = 0; j < n; j++) {
//         rho = rho + r[j] * r[j];
//     }

//     /* the conj grad iteration loop */
//     for (cgit = 1; cgit <= cgitmax; cgit++) {

//         /*
//          * q = A.p
//          */
//         for (j = 0; j < nrow; j++) {
//             sum = 0.0;
//             for (k = rowstr[j]; k < rowstr[j + 1]; k++) {
//                 sum = sum + a[k] * p[colidx[k]];
//             }
//             q[j] = sum;
//         }

//         /*
//          * obtain p.q
//          */
//         d = 0.0;
//         for (j = 0; j < n; j++) {
//             d = d + p[j] * q[j];
//         }

//         /*
//          * obtain alpha
//          */
//         alpha = rho / d;

//         /*
//          * save rho
//          */
//         rho0 = rho;

//         /*
//          * fused:
//          * z = z + alpha*p
//          * r = r - alpha*q
//          * rho = r.r
//          */
//         rho = 0.0;

// #pragma GCC ivdep
//         for (j = 0; j < n; j++) {
//             z[j] += alpha * p[j];
//             double rj = r[j] - alpha * q[j];
//             r[j] = rj;
//             rho += rj * rj;
//         }

//         beta = rho / rho0;

//         for (j = 0; j < n; j++) {
//             p[j] = r[j] + beta * p[j];
//         }
//     }

//     sum = 0.0;

//     for (j = 0; j < nrow; j++) {
//         d = 0.0;
//         for (k = rowstr[j]; k < rowstr[j + 1]; k++) {
//             d = d + a[k] * z[colidx[k]];
//         }
//         r[j] = d;
//     }

//     #pragma GCC ivdep
//     for (j = 0; j < n; j++) {
//         d = x[j] - r[j];
//         sum = sum + d * d;
//     }

//     *rnorm = sqrt(sum);
// }
// static void conj_grad(const int* __restrict colidx,
//                       const int* __restrict rowstr,
//                       const double* __restrict x,
//                       double* __restrict z,
//                       const double* __restrict a,
//                       double* __restrict p,
//                       double* __restrict q,
//                       double* __restrict r,
//                       double* __restrict rnorm)
// {
//   const int n    = lastcol - firstcol + 1;
//   const int nrow = lastrow - firstrow + 1;
//   const int cgitmax = 25;

//   // Only keep these if allocations are truly 64B aligned.
//   p = (double*)__builtin_assume_aligned(p, 64);
//   q = (double*)__builtin_assume_aligned(q, 64);
//   r = (double*)__builtin_assume_aligned(r, 64);
//   z = (double*)__builtin_assume_aligned(z, 64);
//   x = (const double*)__builtin_assume_aligned(x, 64);

//   // init: q=0, z=0, r=x, p=x
//   for (int j = 0; j < n; j++) {
//     q[j] = 0.0;
//     z[j] = 0.0;
//     r[j] = x[j];
//     p[j] = x[j];
//   }

//   // rho = r.r
//   double rho = 0.0;
// #if defined(__AVX2__)
//   {
//     __m256d acc = _mm256_setzero_pd();
//     int j = 0;
//     for (; j + 3 < n; j += 4) {
//       __m256d rv = _mm256_load_pd(r + j);
//       acc = _mm256_fmadd_pd(rv, rv, acc);
//     }
//     rho = hsum256_pd(acc);
//     for (; j < n; j++) rho += r[j] * r[j];
//   }
// #else
//   for (int j = 0; j < n; j++) rho += r[j] * r[j];
// #endif

//   for (int cgit = 1; cgit <= cgitmax; cgit++) {

//     // -------------------------
//     // q = A * p   (SpMV)  UNROLL x8
//     // -------------------------
//     for (int j = 0; j < nrow; j++) {
//       const int start = rowstr[j];
//       const int end   = rowstr[j + 1];

//       double s0=0.0, s1=0.0, s2=0.0, s3=0.0;
//       double s4=0.0, s5=0.0, s6=0.0, s7=0.0;

//       int k = start;

// #if CG_PREFETCH_DIST > 0
//       // small warm prefetch
//       if (k + CG_PREFETCH_DIST < end) {
//         __builtin_prefetch(a + k + CG_PREFETCH_DIST, 0, 1);
//         __builtin_prefetch(colidx + k + CG_PREFETCH_DIST, 0, 1);
//         __builtin_prefetch(p + colidx[k + CG_PREFETCH_DIST], 0, 1);
//       }
// #endif

//       for (; k + 7 < end; k += 8) {
// #if CG_PREFETCH_DIST > 0
//         const int pf = k + CG_PREFETCH_DIST;
//         if (pf < end) {
//           __builtin_prefetch(a + pf, 0, 1);
//           __builtin_prefetch(colidx + pf, 0, 1);
//           __builtin_prefetch(p + colidx[pf], 0, 1);
//         }
// #endif
//         const int i0 = colidx[k + 0];
//         const int i1 = colidx[k + 1];
//         const int i2 = colidx[k + 2];
//         const int i3 = colidx[k + 3];
//         const int i4 = colidx[k + 4];
//         const int i5 = colidx[k + 5];
//         const int i6 = colidx[k + 6];
//         const int i7 = colidx[k + 7];

//         s0 += a[k + 0] * p[i0];
//         s1 += a[k + 1] * p[i1];
//         s2 += a[k + 2] * p[i2];
//         s3 += a[k + 3] * p[i3];
//         s4 += a[k + 4] * p[i4];
//         s5 += a[k + 5] * p[i5];
//         s6 += a[k + 6] * p[i6];
//         s7 += a[k + 7] * p[i7];
//       }

//       double sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));

//       for (; k < end; k++) {
//         sum += a[k] * p[colidx[k]];
//       }
//       q[j] = sum;
//     }

//     // d = p.q
//     double d = 0.0;
// #if defined(__AVX2__)
//     {
//       __m256d acc = _mm256_setzero_pd();
//       int j = 0;
//       for (; j + 3 < n; j += 4) {
//         __m256d pv = _mm256_load_pd(p + j);
//         __m256d qv = _mm256_load_pd(q + j);
//         acc = _mm256_fmadd_pd(pv, qv, acc);
//       }
//       d = hsum256_pd(acc);
//       for (; j < n; j++) d += p[j] * q[j];
//     }
// #else
//     for (int j = 0; j < n; j++) d += p[j] * q[j];
// #endif

//     const double alpha = rho / d;
//     const double rho0  = rho;

//     // fused: z += alpha*p ; r -= alpha*q ; rho = r.r
//     rho = 0.0;
// #if defined(__AVX2__)
//     {
//       __m256d aval = _mm256_set1_pd(alpha);
//       __m256d acc  = _mm256_setzero_pd();
//       int j = 0;
//       for (; j + 3 < n; j += 4) {
//         __m256d zv = _mm256_load_pd(z + j);
//         __m256d pv = _mm256_load_pd(p + j);
//         __m256d rv = _mm256_load_pd(r + j);
//         __m256d qv = _mm256_load_pd(q + j);

//         // z = z + alpha*p
//         zv = _mm256_fmadd_pd(aval, pv, zv);
//         _mm256_store_pd(z + j, zv);

//         // r = r - alpha*q
//         __m256d rq = _mm256_fmadd_pd(aval, qv, _mm256_setzero_pd());
//         rv = _mm256_sub_pd(rv, rq);
//         _mm256_store_pd(r + j, rv);

//         // rho += r*r
//         acc = _mm256_fmadd_pd(rv, rv, acc);
//       }
//       rho = hsum256_pd(acc);
//       for (; j < n; j++) {
//         z[j] += alpha * p[j];
//         const double rj = r[j] - alpha * q[j];
//         r[j] = rj;
//         rho += rj * rj;
//       }
//     }
// #else
//     for (int j = 0; j < n; j++) {
//       z[j] += alpha * p[j];
//       const double rj = r[j] - alpha * q[j];
//       r[j] = rj;
//       rho += rj * rj;
//     }
// #endif

//     const double beta = rho / rho0;

//     // p = r + beta*p
// #if defined(__AVX2__)
//     {
//       __m256d bval = _mm256_set1_pd(beta);
//       int j = 0;
//       for (; j + 3 < n; j += 4) {
//         __m256d rv = _mm256_load_pd(r + j);
//         __m256d pv = _mm256_load_pd(p + j);
//         pv = _mm256_fmadd_pd(bval, pv, rv);
//         _mm256_store_pd(p + j, pv);
//       }
//       for (; j < n; j++) p[j] = r[j] + beta * p[j];
//     }
// #else
//     for (int j = 0; j < n; j++) p[j] = r[j] + beta * p[j];
// #endif
//   }

//   // -------------------------
//   // r = A * z  (SpMV)  (keep scalar, can also unroll like above)
//   // -------------------------
//   for (int j = 0; j < nrow; j++) {
//     double sum = 0.0;
//     for (int k = rowstr[j]; k < rowstr[j + 1]; k++) {
//       sum += a[k] * z[colidx[k]];
//     }
//     r[j] = sum;
//   }

//   // rnorm = ||x - r||
//   double s = 0.0;
// #if defined(__AVX2__)
//   {
//     __m256d acc = _mm256_setzero_pd();
//     int j = 0;
//     for (; j + 3 < n; j += 4) {
//       __m256d xv = _mm256_load_pd(x + j);
//       __m256d rv = _mm256_load_pd(r + j);
//       __m256d dv = _mm256_sub_pd(xv, rv);
//       acc = _mm256_fmadd_pd(dv, dv, acc);
//     }
//     s = hsum256_pd(acc);
//     for (; j < n; j++) {
//       const double diff = x[j] - r[j];
//       s += diff * diff;
//     }
//   }
// #else
//   for (int j = 0; j < n; j++) {
//     const double diff = x[j] - r[j];
//     s += diff * diff;
//   }
// #endif

//   *rnorm = std::sqrt(s);
// }

// Sort each CSR row by colidx, permuting a[] the same way.
// This improves spatial locality when reading p[colidx[k]].
static void csr_sort_rows_by_col(int nrow, const int *__restrict rowstr,
                                 int *__restrict colidx, double *__restrict a) {
  // temporary buffer reused per row (avoid realloc churn)
  std::vector<std::pair<int, double>> buf;
  buf.reserve(256); // will grow if needed

  for (int r = 0; r < nrow; ++r) {
    const int start = rowstr[r];
    const int end = rowstr[r + 1];
    const int nnz = end - start;
    if (nnz <= 1)
      continue;

    if ((int)buf.size() < nnz)
      buf.resize(nnz);

    for (int k = 0; k < nnz; ++k) {
      buf[k].first = colidx[start + k];
      buf[k].second = a[start + k];
    }

    std::sort(buf.begin(), buf.begin() + nnz,
              [](const auto &x, const auto &y) { return x.first < y.first; });

    for (int k = 0; k < nnz; ++k) {
      colidx[start + k] = buf[k].first;
      a[start + k] = buf[k].second;
    }
  }
}
#include <algorithm>
#include <cstdint>
#include <vector>

// Build undirected adjacency from CSR (0..n-1)
static void build_undirected_adjacency_from_csr(int n,
                                                const int *__restrict rowstr,
                                                const int *__restrict colidx,
                                                std::vector<int> &adj_offsets,
                                                std::vector<int> &adj) {
  // Count degrees (undirected, avoid self-loops)
  std::vector<int> deg(n, 0);

  for (int i = 0; i < n; ++i) {
    for (int k = rowstr[i]; k < rowstr[i + 1]; ++k) {
      int j = colidx[k];
      if (j == i)
        continue;
      // add i->j
      deg[i]++;
      // add j->i
      deg[j]++;
    }
  }

  adj_offsets.resize(n + 1);
  adj_offsets[0] = 0;
  for (int i = 0; i < n; ++i)
    adj_offsets[i + 1] = adj_offsets[i] + deg[i];
  adj.resize(adj_offsets[n]);

  // Fill adjacency (with duplicates possible; we’ll unique per node later)
  std::vector<int> cursor = adj_offsets;
  for (int i = 0; i < n; ++i) {
    for (int k = rowstr[i]; k < rowstr[i + 1]; ++k) {
      int j = colidx[k];
      if (j == i)
        continue;

      adj[cursor[i]++] = j;
      adj[cursor[j]++] = i;
    }
  }

  // Sort + unique each adjacency list to remove duplicates
  for (int i = 0; i < n; ++i) {
    int begin = adj_offsets[i];
    int end = adj_offsets[i + 1];
    std::sort(adj.begin() + begin, adj.begin() + end);
    auto it = std::unique(adj.begin() + begin, adj.begin() + end);
    int new_end = (int)std::distance(adj.begin(), it);

    // compact by shifting tail left if needed
    // easiest: just mark size changes; compact globally afterward
    // We'll do a compact pass building new arrays.
    // For simplicity (and because this is one-time setup), rebuild compact:
  }

  // Rebuild compact adjacency (one-time cost)
  std::vector<int> new_offsets(n + 1, 0);
  for (int i = 0; i < n; ++i) {
    int begin = adj_offsets[i];
    int end = adj_offsets[i + 1];
    std::sort(adj.begin() + begin, adj.begin() + end);
    int new_end = (int)std::distance(
        adj.begin(), std::unique(adj.begin() + begin, adj.begin() + end));
    new_offsets[i + 1] = new_offsets[i] + (new_end - begin);
  }

  std::vector<int> new_adj(new_offsets[n]);
  for (int i = 0; i < n; ++i) {
    int begin = adj_offsets[i];
    int end = adj_offsets[i + 1];
    std::sort(adj.begin() + begin, adj.begin() + end);
    auto it = std::unique(adj.begin() + begin, adj.begin() + end);
    int out_begin = new_offsets[i];
    for (auto p = adj.begin() + begin; p != it; ++p) {
      new_adj[out_begin++] = *p;
    }
  }

  adj_offsets.swap(new_offsets);
  adj.swap(new_adj);
}

// Compute RCM permutation.
// Output: perm[new_index] = old_index
// Also returns pinv[old_index] = new_index
static void rcm_ordering_from_adjacency(int n,
                                        const std::vector<int> &adj_offsets,
                                        const std::vector<int> &adj,
                                        std::vector<int> &perm,
                                        std::vector<int> &pinv) {
  std::vector<uint8_t> visited(n, 0);

  // Precompute degrees
  std::vector<int> degree(n);
  for (int i = 0; i < n; ++i) {
    degree[i] = adj_offsets[i + 1] - adj_offsets[i];
  }

  perm.clear();
  perm.reserve(n);

  std::vector<int> queue;
  queue.reserve(n);

  std::vector<int> neighbors;
  neighbors.reserve(256);

  // Helper: pick next start node among unvisited with minimal degree
  // (heuristic)
  auto pick_start = [&]() -> int {
    int best = -1;
    int best_deg = 0x7fffffff;
    for (int i = 0; i < n; ++i) {
      if (!visited[i]) {
        int d = degree[i];
        if (d < best_deg) {
          best_deg = d;
          best = i;
        }
      }
    }
    return best;
  };

  while ((int)perm.size() < n) {
    int start = pick_start();
    if (start < 0)
      break;

    // Standard Cuthill–McKee BFS from start
    queue.clear();
    queue.push_back(start);
    visited[start] = 1;

    for (size_t qi = 0; qi < queue.size(); ++qi) {
      int v = queue[qi];
      perm.push_back(v);

      // collect unvisited neighbors
      neighbors.clear();
      int begin = adj_offsets[v];
      int end = adj_offsets[v + 1];
      for (int e = begin; e < end; ++e) {
        int u = adj[e];
        if (!visited[u])
          neighbors.push_back(u);
      }

      // visit neighbors in increasing degree
      std::sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
        int da = degree[a], db = degree[b];
        if (da != db)
          return da < db;
        return a < b;
      });

      for (int u : neighbors) {
        if (!visited[u]) {
          visited[u] = 1;
          queue.push_back(u);
        }
      }
    }
  }

  // Reverse for RCM
  std::reverse(perm.begin(), perm.end());

  // Build inverse
  pinv.resize(n);
  for (int newi = 0; newi < n; ++newi) {
    pinv[perm[newi]] = newi;
  }
}

// Apply symmetric permutation: A' = P * A * P^T
// perm[new] = old, pinv[old] = new
static void csr_apply_symmetric_permutation(int n, const int *__restrict rowstr,
                                            const int *__restrict colidx,
                                            const double *__restrict a,
                                            const std::vector<int> &pinv,
                                            std::vector<int> &rowstr2,
                                            std::vector<int> &colidx2,
                                            std::vector<double> &a2) {
  const int nnz = rowstr[n];

  // Count nnz per new row
  rowstr2.assign(n + 1, 0);

  for (int old_i = 0; old_i < n; ++old_i) {
    int new_i = pinv[old_i];
    int count = rowstr[old_i + 1] - rowstr[old_i];
    rowstr2[new_i + 1] = count;
  }

  // Prefix sum
  rowstr2[0] = 0;
  for (int i = 0; i < n; ++i)
    rowstr2[i + 1] += rowstr2[i];

  colidx2.resize(nnz);
  a2.resize(nnz);

  // Fill new CSR
  // We'll write row by row; need per-row cursor
  std::vector<int> cursor(rowstr2.begin(), rowstr2.end());

  for (int old_i = 0; old_i < n; ++old_i) {
    int new_i = pinv[old_i];
    int out_k = cursor[new_i];

    for (int k = rowstr[old_i]; k < rowstr[old_i + 1]; ++k) {
      int old_j = colidx[k];
      int new_j = pinv[old_j];

      colidx2[out_k] = new_j;
      a2[out_k] = a[k];
      ++out_k;
    }
    cursor[new_i] = out_k;
  }

  // Sort columns within each new row (important after permute)
  for (int i = 0; i < n; ++i) {
    int begin = rowstr2[i];
    int end = rowstr2[i + 1];
    int len = end - begin;
    if (len <= 1)
      continue;

    // sort pairs (col, val)
    std::vector<std::pair<int, double>> tmp(len);
    for (int t = 0; t < len; ++t)
      tmp[t] = {colidx2[begin + t], a2[begin + t]};

    std::sort(tmp.begin(), tmp.end(),
              [](auto &x, auto &y) { return x.first < y.first; });

    for (int t = 0; t < len; ++t) {
      colidx2[begin + t] = tmp[t].first;
      a2[begin + t] = tmp[t].second;
    }
  }
}

// Convenience wrapper: compute RCM from CSR and apply it in-place to
// (rowstr,colidx,a)
static void csr_apply_rcm_inplace(int n, int *__restrict rowstr,
                                  int *__restrict colidx,
                                  double *__restrict a) {
  // 1) build adjacency
  std::vector<int> adj_offsets, adj;
  build_undirected_adjacency_from_csr(n, rowstr, colidx, adj_offsets, adj);

  // 2) compute RCM perm
  std::vector<int> perm, pinv;
  rcm_ordering_from_adjacency(n, adj_offsets, adj, perm, pinv);

  // 3) apply symmetric permutation to CSR -> into temp buffers
  std::vector<int> rowstr2, colidx2;
  std::vector<double> a2;
  csr_apply_symmetric_permutation(n, rowstr, colidx, a, pinv, rowstr2, colidx2,
                                  a2);

  // 4) copy back
  // rowstr size n+1
  for (int i = 0; i < n + 1; ++i)
    rowstr[i] = rowstr2[i];
  // nnz = rowstr[n]
  const int nnz = rowstr2[n];
  for (int k = 0; k < nnz; ++k) {
    colidx[k] = colidx2[k];
    a[k] = a2[k];
  }
}
static void csr_apply_rcm_inplace_with_perm(int n, int *__restrict rowstr,
                                            int *__restrict colidx,
                                            double *__restrict a,
                                            std::vector<int> &perm,
                                            std::vector<int> &pinv) {
  std::vector<int> adj_offsets, adj;
  build_undirected_adjacency_from_csr(n, rowstr, colidx, adj_offsets, adj);

  rcm_ordering_from_adjacency(n, adj_offsets, adj, perm, pinv);

  std::vector<int> rowstr2, colidx2;
  std::vector<double> a2;
  csr_apply_symmetric_permutation(n, rowstr, colidx, a, pinv, rowstr2, colidx2,
                                  a2);

  for (int i = 0; i < n + 1; ++i)
    rowstr[i] = rowstr2[i];
  const int nnz = rowstr2[n];
  for (int k = 0; k < nnz; ++k) {
    colidx[k] = colidx2[k];
    a[k] = a2[k];
  }
}
static void apply_pinv_to_vector(int n, const std::vector<int>& pinv,
                                const double* __restrict x_in,
                                double* __restrict x_out)
{
    // x_out[new_i] = x_in[old_i]
    for (int old_i = 0; old_i < n; ++old_i) {
        int new_i = pinv[old_i];
        x_out[new_i] = x_in[old_i];
    }
}

static void apply_perm_to_vector(int n, const std::vector<int>& perm,
                                const double* __restrict x_in,
                                double* __restrict x_out)
{
    // x_out[old_i] = x_in[new_i]
    for (int new_i = 0; new_i < n; ++new_i) {
        int old_i = perm[new_i];
        x_out[old_i] = x_in[new_i];
    }
}


struct SellCSigma {
    int n = 0;            // number of rows/cols
    int C = 0;            // slice height
    int sigma = 0;        // sorting window
    int nslices = 0;

    // Row permutation: new_row -> old_row (only rows are permuted)
    // For SpMV we apply the same permutation to input/output vectors in the kernel.
    std::vector<int> row_perm;     // size n: new_r -> old_r
    std::vector<int> row_iperm;    // size n: old_r -> new_r

    // Slice metadata
    std::vector<int> slice_ptr;    // size nslices+1: prefix sum into val/col arrays
    std::vector<int> slice_len;    // size nslices: max nnz per row in slice

    // SELL storage (padded)
    // Layout: for slice s, for t in [0..slice_len[s)-1], for r in [0..C-1]:
    // index = slice_ptr[s] + t*C + r
    std::vector<int>    col;
    std::vector<double> val;
};

// Build a row permutation for sigma-sorting (within blocks of sigma rows)
// Sort rows by decreasing nnz in each sigma window.
static void build_sigma_row_permutation(int n,
                                        const int* __restrict rowstr,
                                        int sigma,
                                        std::vector<int>& row_perm,
                                        std::vector<int>& row_iperm)
{
    row_perm.resize(n);
    for (int i = 0; i < n; ++i) row_perm[i] = i;

    if (sigma <= 0) sigma = n;

    for (int base = 0; base < n; base += sigma) {
        int end = std::min(base + sigma, n);
        std::sort(row_perm.begin() + base, row_perm.begin() + end,
                  [&](int a, int b) {
                      int da = rowstr[a + 1] - rowstr[a];
                      int db = rowstr[b + 1] - rowstr[b];
                      if (da != db) return da > db; // decreasing nnz
                      return a < b;
                  });
    }

    row_iperm.resize(n);
    for (int newr = 0; newr < n; ++newr) {
        row_iperm[row_perm[newr]] = newr;
    }
}

// Convert CSR -> SELL-C-σ
static SellCSigma csr_to_sell_c_sigma(int n,
                                      const int* __restrict rowstr,
                                      const int* __restrict colidx,
                                      const double* __restrict a,
                                      int C,
                                      int sigma)
{
    SellCSigma S;
    S.n = n;
    S.C = C;
    S.sigma = sigma;

    // Build sigma-sorted row permutation (new_row -> old_row)
    build_sigma_row_permutation(n, rowstr, sigma, S.row_perm, S.row_iperm);

    const int nslices = (n + C - 1) / C;
    S.nslices = nslices;
    S.slice_ptr.assign(nslices + 1, 0);
    S.slice_len.assign(nslices, 0);

    // Determine max nnz per slice after row permutation
    for (int s = 0; s < nslices; ++s) {
        int maxlen = 0;
        int base = s * C;
        for (int r = 0; r < C; ++r) {
            int new_row = base + r;
            if (new_row >= n) break;
            int old_row = S.row_perm[new_row];
            int len = rowstr[old_row + 1] - rowstr[old_row];
            if (len > maxlen) maxlen = len;
        }
        S.slice_len[s] = maxlen;
        S.slice_ptr[s + 1] = S.slice_ptr[s] + maxlen * C;
    }

    const int total = S.slice_ptr[nslices];
    S.col.assign(total, -1);
    S.val.assign(total, 0.0);

    // Fill SELL arrays
    for (int s = 0; s < nslices; ++s) {
        int base = s * C;
        int maxlen = S.slice_len[s];
        int out_base = S.slice_ptr[s];

        for (int r = 0; r < C; ++r) {
            int new_row = base + r;
            if (new_row >= n) break;
            int old_row = S.row_perm[new_row];

            int start = rowstr[old_row];
            int end   = rowstr[old_row + 1];
            int len   = end - start;

            // For t-th entry in this row, store at out_base + t*C + r
            for (int t = 0; t < len; ++t) {
                int out = out_base + t * C + r;
                S.col[out] = colidx[start + t];
                S.val[out] = a[start + t];
            }
            // padding remains (-1,0)
            (void)maxlen;
        }
    }

    return S;
}

// SELL SpMV: y = A * x
// Note: SELL stores rows permuted by S.row_perm (new_row -> old_row).
// For mathematical correctness with the original x/y ordering, we compute:
// y_new[new_row] = sum A(old_row, j) * x[j]
// then scatter to y[old_row] if you want original order.
// BUT for CG, it’s cheaper to keep vectors in SELL order.
// This kernel outputs y in SELL order (new_row order).
static void sell_spmv_new_order(const SellCSigma& S,
                                const double* __restrict x,  // x is in ORIGINAL column order
                                double* __restrict y_new)    // y in SELL row order
{
    const int n = S.n;
    const int C = S.C;

    for (int i = 0; i < n; ++i) y_new[i] = 0.0;

    for (int s = 0; s < S.nslices; ++s) {
        const int base_row = s * C;
        const int maxlen   = S.slice_len[s];
        const int base     = S.slice_ptr[s];

        for (int t = 0; t < maxlen; ++t) {
            const int off = base + t * C;
            // iterate rows within slice
#pragma GCC ivdep
            for (int r = 0; r < C; ++r) {
                int new_row = base_row + r;
                if (new_row >= n) break;
                int cj = S.col[off + r];
                if (cj >= 0) {
                    y_new[new_row] += S.val[off + r] * x[cj];
                }
            }
        }
    }
}
static void permute_to_sell_order(const SellCSigma& S,
                                  const double* __restrict v_old,
                                  double* __restrict v_new)
{
    // v_new[new_row] = v_old[old_row]
    for (int newr = 0; newr < S.n; ++newr) {
        v_new[newr] = v_old[S.row_perm[newr]];
    }
}

static void permute_from_sell_order(const SellCSigma& S,
                                    const double* __restrict v_new,
                                    double* __restrict v_old)
{
    // v_old[old_row] = v_new[new_row]
    for (int newr = 0; newr < S.n; ++newr) {
        v_old[S.row_perm[newr]] = v_new[newr];
    }
}

static void conj_grad_sell(const SellCSigma& S,
                           const double* __restrict x_old,
                           double* __restrict z_old,
                           double* __restrict p_new,
                           double* __restrict q_new,
                           double* __restrict r_new,
                           double* __restrict tmp_new,
                           double* __restrict rnorm)
{
    const int n = S.n;
    const int cgitmax = 25;

    // Convert x to SELL row order
    permute_to_sell_order(S, x_old, tmp_new); // tmp_new = x_new

    // init in SELL order:
    // q=0, z=0, r=x, p=r
    for (int i = 0; i < n; ++i) {
        q_new[i] = 0.0;
        z_old[i] = 0.0; // careful: z_old is original-order buffer in caller; we use it later
    }
    // We'll store z in SELL order inside tmp_new2; reuse p_new as z_new temporarily? better: use z_old as z_new buffer temporarily by treating it as array.
    // We'll use z_old as a SELL-ordered workspace first, then unpermute at end.
    double* __restrict z_new = z_old;

    for (int i = 0; i < n; ++i) {
        z_new[i] = 0.0;
        r_new[i] = tmp_new[i];
        p_new[i] = r_new[i];
    }

    double rho = 0.0;
    for (int i = 0; i < n; ++i) rho += r_new[i] * r_new[i];

    // CG iterations
    for (int it = 0; it < cgitmax; ++it) {

        // q = A * p   (SELL SpMV) -> q_new
        // Important: sell_spmv_new_order expects x in ORIGINAL column order.
        // Here p_new is in SELL row order. For correctness, columns must match row order too.
        // Therefore: This SELL kernel is most correct/effective AFTER you have applied a symmetric permutation (e.g., RCM) so that row/col order match.
        // If you have done symmetric permutation already, p_new indexing is valid for colidx.
        sell_spmv_new_order(S, p_new, q_new);

        double d = 0.0;
        for (int i = 0; i < n; ++i) d += p_new[i] * q_new[i];

        const double alpha = rho / d;
        const double rho0  = rho;

        rho = 0.0;
#pragma GCC ivdep
        for (int i = 0; i < n; ++i) {
            z_new[i] += alpha * p_new[i];
            const double ri = r_new[i] - alpha * q_new[i];
            r_new[i] = ri;
            rho += ri * ri;
        }

        const double beta = rho / rho0;

#pragma GCC ivdep
        for (int i = 0; i < n; ++i) {
            p_new[i] = r_new[i] + beta * p_new[i];
        }
    }

    // residual norm: ||x - A*z||
    // r = A*z
    sell_spmv_new_order(S, z_new, r_new);

    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = tmp_new[i] - r_new[i]; // x_new - r_new
        s += diff * diff;
    }
    *rnorm = std::sqrt(s);

    // Convert z back to original order for the rest of the benchmark
    permute_from_sell_order(S, z_new, z_old);
}

static int icnvrt(double x, int ipwr2);
static void makea(int n, int nz, double a[], int colidx[], int rowstr[],
                  int firstrow, int lastrow, int firstcol, int lastcol,
                  int arow[], int acol[][NONZER + 1], double aelt[][NONZER + 1],
                  int iv[]);
static void sparse(double a[], int colidx[], int rowstr[], int n, int nz,
                   int nozer, int arow[], int acol[][NONZER + 1],
                   double aelt[][NONZER + 1], int firstrow, int lastrow,
                   int nzloc[], double rcond, double shift);
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);

/* cg */
int main(int argc, char **argv) {
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
  printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION "
         "mode on\n");
#endif
  int i, j, k, it;
  double zeta;
  double rnorm;
  double norm_temp1, norm_temp2;
  double t, mflops, tmax;
  char class_npb;
  boolean verified;
  double zeta_verify_value, epsilon, err;

  char *t_names[T_LAST];

  for (i = 0; i < T_LAST; i++) {
    timer_clear(i);
  }

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = TRUE;
    t_names[T_INIT] = (char *)"init";
    t_names[T_BENCH] = (char *)"benchmk";
    t_names[T_CONJ_GRAD] = (char *)"conjgd";
    fclose(fp);
  } else {
    timeron = FALSE;
  }

  timer_start(T_INIT);

  firstrow = 0;
  lastrow = NA - 1;
  firstcol = 0;
  lastcol = NA - 1;

  if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10.0) {
    class_npb = 'S';
    zeta_verify_value = 8.5971775078648;
  } else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12.0) {
    class_npb = 'W';
    zeta_verify_value = 10.362595087124;
  } else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20.0) {
    class_npb = 'A';
    zeta_verify_value = 17.130235054029;
  } else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60.0) {
    class_npb = 'B';
    zeta_verify_value = 22.712745482631;
  } else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110.0) {
    class_npb = 'C';
    zeta_verify_value = 28.973605592845;
  } else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500.0) {
    class_npb = 'D';
    zeta_verify_value = 52.514532105794;
  } else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500.0) {
    class_npb = 'E';
    zeta_verify_value = 77.522164599383;
  } else {
    class_npb = 'U';
  }

  printf(
      "\n\n NAS Parallel Benchmarks 4.1 Serial C++ version - CG Benchmark\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations: %5d\n", NITER);

  naa = NA;
  nzz = NZ;

  /* initialize random number generator */
  tran = 314159265.0;
  amult = 1220703125.0;
  zeta = randlc(&tran, amult);

  makea(naa, nzz, a, colidx, rowstr, firstrow, lastrow, firstcol, lastcol, arow,
        (int(*)[NONZER + 1])(void *)acol, (double(*)[NONZER + 1])(void *)aelt,
        iv);

  /*
   * ---------------------------------------------------------------------
   * note: as a result of the above call to makea:
   * values of j used in indexing rowstr go from 0 --> lastrow-firstrow
   * values of colidx which are col indexes go from firstcol --> lastcol
   * so:
   * shift the col index vals from actual (firstcol --> lastcol)
   * to local, i.e., (0 --> lastcol-firstcol)
   * ---------------------------------------------------------------------
   */
  for (j = 0; j < lastrow - firstrow + 1; j++) {
    for (k = rowstr[j]; k < rowstr[j + 1]; k++) {
      colidx[k] = colidx[k] - firstcol;
    }
  }
  const int nrow = lastrow - firstrow + 1;
  const int n = lastcol - firstcol + 1;

  // Option A: only row-sort (cheap)
  // csr_sort_rows_by_col(nrow, rowstr, colidx, a);
  // csr_apply_rcm_inplace(n, rowstr, colidx, a);

  std::vector<int> perm, pinv;
  csr_apply_rcm_inplace_with_perm(n, rowstr, colidx, a, perm, pinv);

  // Permute x into new ordering (use a temp buffer; you already have z/q/r/p)
  apply_pinv_to_vector(n, pinv, x, z);
  for (int i = 0; i < n; ++i)
    x[i] = z[i];

  // Then row-sort (optional)
// csr_sort_rows_by_col(n, rowstr, colidx, a);

// Build SELL
const int C = 16;      // try 8, 16, 32
const int sigma = 256; // try 64, 128, 256, 512
// SellCSigma S = csr_to_sell_c_sigma(n, rowstr, colidx, a, C, sigma);
// work buffers in SELL order:
static double* p_new = p; // reuse
static double* q_new = q;
static double* r_new = r;
static double* tmp_new = z; // reuse z as temp (careful with your use pattern)

// Instead of conj_grad(...)
// conj_grad_sell(S, x, z, p_new, q_new, r_new, tmp_new, &rnorm);

  // Now run benchmark normally in permuted space.
  // (z, p, q, r are work vectors anyway)

  /* set starting vector to (1, 1, .... 1) */
  for (i = 0; i < NA + 1; i++) {
    x[i] = 1.0;
  }
  for (j = 0; j < lastcol - firstcol + 1; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = 0.0;
    p[j] = 0.0;
  }
  zeta = 0.0;

  /*
   * -------------------------------------------------------------------
   * ---->
   * do one iteration untimed to init all code and data page tables
   * ----> (then reinit, start timing, to niter its)
   * -------------------------------------------------------------------*/
  for (it = 1; it <= 1; it++) {
    /* the call to the conjugate gradient routine */
    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
    // conj_grad_sell(S, x, z, p_new, q_new, r_new, tmp_new, &rnorm);

    /*
     * --------------------------------------------------------------------
     * zeta = shift + 1/(x.z)
     * so, first: (x.z)
     * also, find norm of z
     * so, first: (z.z)
     * --------------------------------------------------------------------
     */
    norm_temp1 = 0.0;
    norm_temp2 = 0.0;
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      norm_temp1 = norm_temp1 + x[j] * z[j];
      norm_temp2 = norm_temp2 + z[j] * z[j];
    }
    norm_temp2 = 1.0 / sqrt(norm_temp2);

    /* normalize z to obtain x */
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      x[j] = norm_temp2 * z[j];
    }
  } /* end of do one iteration untimed */

  /* set starting vector to (1, 1, .... 1) */
  for (i = 0; i < NA + 1; i++) {
    x[i] = 1.0;
  }
  zeta = 0.0;

  timer_stop(T_INIT);

  printf(" Initialization time = %15.3f seconds\n", timer_read(T_INIT));

  timer_start(T_BENCH);

  /*
   * --------------------------------------------------------------------
   * ---->
   * main iteration for inverse power method
   * ---->
   * --------------------------------------------------------------------
   */
  for (it = 1; it <= NITER; it++) {
    /* the call to the conjugate gradient routine */
    if (timeron) {
      timer_start(T_CONJ_GRAD);
    }
    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
    if (timeron) {
      timer_stop(T_CONJ_GRAD);
    }

    /*
     * --------------------------------------------------------------------
     * zeta = shift + 1/(x.z)
     * so, first: (x.z)
     * also, find norm of z
     * so, first: (z.z)
     * --------------------------------------------------------------------
     */
    norm_temp1 = 0.0;
    norm_temp2 = 0.0;
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      norm_temp1 = norm_temp1 + x[j] * z[j];
      norm_temp2 = norm_temp2 + z[j] * z[j];
    }
    norm_temp2 = 1.0 / sqrt(norm_temp2);
    zeta = SHIFT + 1.0 / norm_temp1;
    if (it == 1) {
      printf("\n   iteration           ||r||                 zeta\n");
    }
    // printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);

    /* normalize z to obtain x */
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      x[j] = norm_temp2 * z[j];
    }
  } /* end of main iter inv pow meth */

  timer_stop(T_BENCH);

  /*
   * --------------------------------------------------------------------
   * end of timed section
   * --------------------------------------------------------------------
   */

  t = timer_read(T_BENCH);

  printf(" Benchmark completed\n");

  epsilon = 1.0e-10;
  if (class_npb != 'U') {
    err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
    if (err <= epsilon) {
      verified = TRUE;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %20.13e\n", zeta);
      printf(" Error is   %20.13e\n", err);
    } else {
      verified = FALSE;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.13e\n", zeta);
      printf(" The correct zeta is %20.13e\n", zeta_verify_value);
    }
  } else {
    verified = FALSE;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }
  if (t != 0.0) {
    mflops = (double)(2.0 * NITER * NA) *
             (3.0 + (double)(NONZER * (NONZER + 1)) +
              25.0 * (5.0 + (double)(NONZER * (NONZER + 1))) + 3.0) /
             t / 1000000.0;
  } else {
    mflops = 0.0;
  }
  c_print_results(
      (char *)"CG", class_npb, NA, 0, 0, NITER, t, mflops,
      (char *)"          floating point", verified, (char *)NPBVERSION,
      (char *)COMPILETIME, (char *)COMPILERVERSION, (char *)CS1, (char *)CS2,
      (char *)CS3, (char *)CS4, (char *)CS5, (char *)CS6, (char *)CS7);

  /*
   * ---------------------------------------------------------------------
   * more timers
   * ---------------------------------------------------------------------
   */
  if (timeron) {
    tmax = timer_read(T_BENCH);
    if (tmax == 0.0) {
      tmax = 1.0;
    }
    printf("  SECTION   Time (secs)\n");
    for (i = 0; i < T_LAST; i++) {
      t = timer_read(i);
      if (i == T_INIT) {
        printf("  %8s:%9.3f\n", t_names[i], t);
      } else {
        printf("  %8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t * 100.0 / tmax);
        if (i == T_CONJ_GRAD) {
          t = tmax - t;
          printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", t, t * 100.0 / tmax);
        }
      }
    }
  }

  return 0;
}

/*
 * ---------------------------------------------------------------------
 * floating point arrays here are named as in NPB1 spec discussion of
 * CG algorithm
 * ---------------------------------------------------------------------
 */

// static void conj_grad(int colidx[],
// 		int rowstr[],
// 		double x[],
// 		double z[],
// 		double a[],
// 		double p[],
// 		double q[],
// 		double r[],
// 		double* rnorm){
// 	int j, k;
// 	int cgit, cgitmax;
// 	double d, sum, rho, rho0, alpha, beta;

// 	cgitmax = 25;

// 	rho = 0.0;

// 	/* initialize the CG algorithm */
// 	for(j = 0; j < naa+1; j++){
// 		q[j] = 0.0;
// 		z[j] = 0.0;
// 		r[j] = x[j];
// 		p[j] = r[j];
// 	}

// 	/*
// 	 * --------------------------------------------------------------------
// 	 * rho = r.r
// 	 * now, obtain the norm of r: First, sum squares of r elements
// locally...
// 	 * --------------------------------------------------------------------
// 	 */
// 	for(j = 0; j < lastcol - firstcol + 1; j++){
// 		rho = rho + r[j]*r[j];
// 	}

// 	/* the conj grad iteration loop */
// 	for(cgit = 1; cgit <= cgitmax; cgit++){
// 		/*
// 		 *
// ---------------------------------------------------------------------
// 		 * q = A.p
// 		 * the partition submatrix-vector multiply: use workspace w
// 		 *
// ---------------------------------------------------------------------
// 		 *
// 		 * note: this version of the multiply is actually (slightly:
// maybe %5)
// 		 * faster on the sp2 on 16 nodes than is the unrolled-by-2
// version
// 		 * below. on the Cray t3d, the reverse is TRUE, i.e., the
// 		 * unrolled-by-two version is some 10% faster.
// 		 * the unrolled-by-8 version below is significantly faster
// 		 * on the Cray t3d - overall speed of code is 1.5 times faster.
// 		 */
// 		for(j = 0; j < lastrow - firstrow + 1; j++){
// 			sum = 0.0;
// 			for(k = rowstr[j]; k < rowstr[j+1]; k++){
// 				sum = sum + a[k]*p[colidx[k]];
// 			}
// 			q[j] = sum;
// 		}

// 		/*
// 		 *
// --------------------------------------------------------------------
// 		 * obtain p.q
// 		 *
// --------------------------------------------------------------------
// 		 */
// 		d = 0.0;
// 		for (j = 0; j < lastcol - firstcol + 1; j++) {
// 			d = d + p[j]*q[j];
// 		}

// 		/*
// 		 *
// --------------------------------------------------------------------
// 		 * obtain alpha = rho / (p.q)
// 		 *
// -------------------------------------------------------------------
// 		 */
// 		alpha = rho / d;

// 		/*
// 		 *
// --------------------------------------------------------------------
// 		 * save a temporary of rho
// 		 *
// --------------------------------------------------------------------
// 		 */
// 		rho0 = rho;

// 		/*
// 		 *
// ---------------------------------------------------------------------
// 		 * obtain z = z + alpha*p
// 		 * and    r = r - alpha*q
// 		 *
// ---------------------------------------------------------------------
// 		 */
// 		rho = 0.0;
// 		for(j = 0; j < lastcol - firstcol + 1; j++){
// 			z[j] = z[j] + alpha*p[j];
// 			r[j] = r[j] - alpha*q[j];
// 		}

// 		/*
// 		 *
// ---------------------------------------------------------------------
// 		 * rho = r.r
// 		 * now, obtain the norm of r: first, sum squares of r elements
// locally...
// 		 *
// ---------------------------------------------------------------------
// 		 */
// 		for(j = 0; j < lastcol - firstcol + 1; j++){
// 			rho = rho + r[j]*r[j];
// 		}

// 		/*
// 		 *
// ---------------------------------------------------------------------
// 		 * obtain beta
// 		 *
// ---------------------------------------------------------------------
// 		 */
// 		beta = rho / rho0;

// 		/*
// 		 *
// ---------------------------------------------------------------------
// 		 * p = r + beta*p
// 		 *
// ---------------------------------------------------------------------
// 		 */
// 		for(j = 0; j < lastcol - firstcol + 1; j++){
// 			p[j] = r[j] + beta*p[j];
// 		}
// 	} /* end of do cgit=1, cgitmax */

// 	/*
// 	 * ---------------------------------------------------------------------
// 	 * compute residual norm explicitly: ||r|| = ||x - A.z||
// 	 * first, form A.z
// 	 * the partition submatrix-vector multiply
// 	 * ---------------------------------------------------------------------
// 	 */
// 	sum = 0.0;
// 	for(j = 0; j < lastrow - firstrow + 1; j++){
// 		d = 0.0;
// 		for(k = rowstr[j]; k < rowstr[j+1]; k++){
// 			d = d + a[k]*z[colidx[k]];
// 		}
// 		r[j] = d;
// 	}

// 	/*
// 	 * ---------------------------------------------------------------------
// 	 * at this point, r contains A.z
// 	 * ---------------------------------------------------------------------
// 	 */
// 	for(j = 0; j < lastcol-firstcol+1; j++){
// 		d   = x[j] - r[j];
// 		sum = sum + d*d;
// 	}

// 	*rnorm = sqrt(sum);
// }

/*
 * ---------------------------------------------------------------------
 * scale a double precision number x in (0,1) by a power of 2 and chop it
 * ---------------------------------------------------------------------
 */
static int icnvrt(double x, int ipwr2) { return (int)(ipwr2 * x); }

/*
 * ---------------------------------------------------------------------
 * generate the test problem for benchmark 6
 * makea generates a sparse matrix with a
 * prescribed sparsity distribution
 *
 * parameter    type        usage
 *
 * input
 *
 * n            i           number of cols/rows of matrix
 * nz           i           nonzeros as declared array size
 * rcond        r*8         condition number
 * shift        r*8         main diagonal shift
 *
 * output
 *
 * a            r*8         array for nonzeros
 * colidx       i           col indices
 * rowstr       i           row pointers
 *
 * workspace
 *
 * iv, arow, acol i
 * aelt           r*8
 * ---------------------------------------------------------------------
 */
static void makea(int n, int nz, double a[], int colidx[], int rowstr[],
                  int firstrow, int lastrow, int firstcol, int lastcol,
                  int arow[], int acol[][NONZER + 1], double aelt[][NONZER + 1],
                  int iv[]) {
  int iouter, ivelt, nzv, nn1;
  int ivc[NONZER + 1];
  double vc[NONZER + 1];

  /*
   * --------------------------------------------------------------------
   * nonzer is approximately  (int(sqrt(nnza /n)));
   * --------------------------------------------------------------------
   * nn1 is the smallest power of two not less than n
   * --------------------------------------------------------------------
   */
  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  /*
   * -------------------------------------------------------------------
   * generate nonzero positions and save for the use in sparse
   * -------------------------------------------------------------------
   */
  for (iouter = 0; iouter < n; iouter++) {
    nzv = NONZER;
    sprnvc(n, nzv, nn1, vc, ivc);
    vecset(n, vc, ivc, &nzv, iouter + 1, 0.5);
    arow[iouter] = nzv;
    for (ivelt = 0; ivelt < nzv; ivelt++) {
      acol[iouter][ivelt] = ivc[ivelt] - 1;
      aelt[iouter][ivelt] = vc[ivelt];
    }
  }

  /*
   * ---------------------------------------------------------------------
   * ... make the sparse matrix from list of elements with duplicates
   * (iv is used as  workspace)
   * ---------------------------------------------------------------------
   */
  sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, aelt, firstrow, lastrow,
         iv, RCOND, SHIFT);
}

/*
 * ---------------------------------------------------------------------
 * rows range from firstrow to lastrow
 * the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
 * ---------------------------------------------------------------------
 */
static void sparse(double a[], int colidx[], int rowstr[], int n, int nz,
                   int nozer, int arow[], int acol[][NONZER + 1],
                   double aelt[][NONZER + 1], int firstrow, int lastrow,
                   int nzloc[], double rcond, double shift) {
  int nrows;

  /*
   * ---------------------------------------------------
   * generate a sparse matrix from a list of
   * [col, row, element] tri
   * ---------------------------------------------------
   */
  int i, j, j1, j2, nza, k, kk, nzrow, jcol;
  double size, scale, ratio, va;
  boolean goto_40;

  /*
   * --------------------------------------------------------------------
   * how many rows of result
   * --------------------------------------------------------------------
   */
  nrows = lastrow - firstrow + 1;

  /*
   * --------------------------------------------------------------------
   * ...count the number of triples in each row
   * --------------------------------------------------------------------
   */
  for (j = 0; j < nrows + 1; j++) {
    rowstr[j] = 0;
  }
  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza] + 1;
      rowstr[j] = rowstr[j] + arow[i];
    }
  }
  rowstr[0] = 0;
  for (j = 1; j < nrows + 1; j++) {
    rowstr[j] = rowstr[j] + rowstr[j - 1];
  }
  nza = rowstr[nrows] - 1;

  /*
   * ---------------------------------------------------------------------
   * ... rowstr(j) now is the location of the first nonzero
   * of row j of a
   * ---------------------------------------------------------------------
   */
  if (nza > nz) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %d, %d\n", nza, nz);
    exit(EXIT_FAILURE);
  }

  /*
   * ---------------------------------------------------------------------
   * ... preload data pages
   * ---------------------------------------------------------------------
   */
  for (j = 0; j < nrows; j++) {
    for (k = rowstr[j]; k < rowstr[j + 1]; k++) {
      a[k] = 0.0;
      colidx[k] = -1;
    }
    nzloc[j] = 0;
  }

  /*
   * ---------------------------------------------------------------------
   * ... generate actual values by summing duplicates
   * ---------------------------------------------------------------------
   */
  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)(n)));
  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza];

      scale = size * aelt[i][nza];
      for (nzrow = 0; nzrow < arow[i]; nzrow++) {
        jcol = acol[i][nzrow];
        va = aelt[i][nzrow] * scale;

        /*
         * --------------------------------------------------------------------
         * ... add the identity * rcond to the generated matrix to bound
         * the smallest eigenvalue from below by rcond
         * --------------------------------------------------------------------
         */
        if (jcol == j && j == i) {
          va = va + rcond - shift;
        }

        goto_40 = FALSE;
        for (k = rowstr[j]; k < rowstr[j + 1]; k++) {
          if (colidx[k] > jcol) {
            /*
             * ----------------------------------------------------------------
             * ... insert colidx here orderly
             * ----------------------------------------------------------------
             */
            for (kk = rowstr[j + 1] - 2; kk >= k; kk--) {
              if (colidx[kk] > -1) {
                a[kk + 1] = a[kk];
                colidx[kk + 1] = colidx[kk];
              }
            }
            colidx[k] = jcol;
            a[k] = 0.0;
            goto_40 = TRUE;
            break;
          } else if (colidx[k] == -1) {
            colidx[k] = jcol;
            goto_40 = TRUE;
            break;
          } else if (colidx[k] == jcol) {
            /*
             * --------------------------------------------------------------
             * ... mark the duplicated entry
             * -------------------------------------------------------------
             */
            nzloc[j] = nzloc[j] + 1;
            goto_40 = TRUE;
            break;
          }
        }
        if (goto_40 == FALSE) {
          printf("internal error in sparse: i=%d\n", i);
          exit(EXIT_FAILURE);
        }
        a[k] = a[k] + va;
      }
    }
    size = size * ratio;
  }

  /*
   * ---------------------------------------------------------------------
   * ... remove empty entries and generate final results
   * ---------------------------------------------------------------------
   */
  for (j = 1; j < nrows; j++) {
    nzloc[j] = nzloc[j] + nzloc[j - 1];
  }

  for (j = 0; j < nrows; j++) {
    if (j > 0) {
      j1 = rowstr[j] - nzloc[j - 1];
    } else {
      j1 = 0;
    }
    j2 = rowstr[j + 1] - nzloc[j];
    nza = rowstr[j];
    for (k = j1; k < j2; k++) {
      a[k] = a[nza];
      colidx[k] = colidx[nza];
      nza = nza + 1;
    }
  }
  for (j = 1; j < nrows + 1; j++) {
    rowstr[j] = rowstr[j] - nzloc[j - 1];
  }
  nza = rowstr[nrows] - 1;
}

/*
 * ---------------------------------------------------------------------
 * generate a sparse n-vector (v, iv)
 * having nzv nonzeros
 *
 * mark(i) is set to 1 if position i is nonzero.
 * mark is all zero on entry and is reset to all zero before exit
 * this corrects a performance bug found by John G. Lewis, caused by
 * reinitialization of mark on every one of the n calls to sprnvc
 * ---------------------------------------------------------------------
 */
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]) {
  int nzv, ii, i;
  double vecelt, vecloc;

  nzv = 0;

  while (nzv < nz) {
    vecelt = randlc(&tran, amult);

    /*
     * --------------------------------------------------------------------
     * generate an integer between 1 and n in a portable manner
     * --------------------------------------------------------------------
     */
    vecloc = randlc(&tran, amult);
    i = icnvrt(vecloc, nn1) + 1;
    if (i > n) {
      continue;
    }

    /*
     * --------------------------------------------------------------------
     * was this integer generated already?
     * --------------------------------------------------------------------
     */
    boolean was_gen = FALSE;
    for (ii = 0; ii < nzv; ii++) {
      if (iv[ii] == i) {
        was_gen = TRUE;
        break;
      }
    }
    if (was_gen) {
      continue;
    }
    v[nzv] = vecelt;
    iv[nzv] = i;
    nzv = nzv + 1;
  }
}

/*
 * --------------------------------------------------------------------
 * set ith element of sparse vector (v, iv) with
 * nzv nonzeros to val
 * --------------------------------------------------------------------
 */
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val) {
  int k;
  boolean set;

  set = FALSE;
  for (k = 0; k < *nzv; k++) {
    if (iv[k] == i) {
      v[k] = val;
      set = TRUE;
    }
  }
  if (set == FALSE) {
    v[*nzv] = val;
    iv[*nzv] = i;
    *nzv = *nzv + 1;
  }
}
