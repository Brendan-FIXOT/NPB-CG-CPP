# NPB SER Optimizations

### Initial run

```bash
 Benchmark completed
 VERIFICATION SUCCESSFUL
 Zeta is     2.2712745482631e+01
 Error is    6.2567753977913e-16


 CG Benchmark Completed
 class_npb       =                        B
 Size            =                    75000
 Iterations      =                       75
 Time in seconds =                    55.88
 Mop/s total     =                   979.08
 Operation type  =           floating point
 Verification    =               SUCCESSFUL
 Version         =                      4.1
 Compiler ver    =                   13.3.0
 Compile date    =              25 Feb 2026

 Compile options:
    CC           = g++ -std=c++14
    CLINK        = $(CC)
    C_LIB        = -lm 
    C_INC        = -I../common 
    CFLAGS       = -O3 -mcmodel=medium
    CLINKFLAGS   = -O3 -mcmodel=medium
    RAND         = randdp

```


More aggressive compilation options didn't change anything in terms of performance, and fastmath didn't yield any noticable preicison error.
```text
CFLAGS = -O3 -march=native -mtune=native --fp-contract=fast -fno-math-errno -fno-trapping-math -funroll-loops -flto -mcmodel=small -fopt-info-vec-optimized -fopt-info-vec-missed
CLINKFLAGS = -O3 -march=native -mtune=native -flto -mcmodel=small
```



Added restrict keyword since pointers aren't overlapping.
Fused two loops in one and vectorized, benchmark went from 85s to 80s. Also replaced the lastcol - firstcol + 1 with const variables.


```cpp
// Old code
		rho = 0.0;
		for(j = 0; j < n; j++){
			z[j] = z[j] + alpha*p[j];
			r[j] = r[j] - alpha*q[j];
		}
		for(j = 0; j < n; j++){
			rho = rho + r[j]*r[j];
		}

//Fused in one single loop
		rho = 0.0;
    //Guarantees that there is no carry over and can vectorize
		#pragma GCC ivdep 
		for (j = 0; j < n; j++){
			z[j] += alpha * p[j];
			double rj = r[j] -alpha*q[j];
			r[j] = rj;
			rho += rj*rj;
		}
```




```bash
 Benchmark completed
 VERIFICATION SUCCESSFUL
 Zeta is     2.2712745482631e+01
 Error is    6.2567753977913e-16


 CG Benchmark Completed
 class_npb       =                        B
 Size            =                    75000
 Iterations      =                       75
 Time in seconds =                    54.64
 Mop/s total     =                  1001.28
 Operation type  =           floating point
 Verification    =               SUCCESSFUL
 Version         =                      4.1
 Compiler ver    =                   13.3.0
 Compile date    =              25 Feb 2026

 Compile options:
    CC           = g++ -std=c++14
    CLINK        = $(CC)
    C_LIB        = -lm 
    C_INC        = -I../common 
    CFLAGS       = -O3 -mcmodel=medium
    CLINKFLAGS   = -O3 -mcmodel=medium
    RAND         = randdp
```

Not a big win.


Enforcing memory alignement and assuming alignment
```cpp
static void* aligned_malloc_or_die(std::size_t alignment, std::size_t bytes) {
  void* p = nullptr;
  // posix_memalign requires alignment to be power-of-two and multiple of sizeof(void*)
  int rc = posix_memalign(&p, alignment, bytes);
  if (rc != 0 || !p) {
    std::fprintf(stderr, "posix_memalign(%zu, %zu) failed (rc=%d)\n",
                 alignment, bytes, rc);
    std::abort();
  }
  return p;
}
static int*    colidx = (int*)   aligned_malloc_or_die(64, sizeof(int)    * (NZ));
static int*    rowstr = (int*)   aligned_malloc_or_die(64, sizeof(int)    * (NA + 1));
static int*    iv     = (int*)   aligned_malloc_or_die(64, sizeof(int)    * (NA));
static int*    arow   = (int*)   aligned_malloc_or_die(64, sizeof(int)    * (NA));
static int*    acol   = (int*)   aligned_malloc_or_die(64, sizeof(int)    * (NAZ));

static double* aelt   = (double*)aligned_malloc_or_die(64, sizeof(double) * (NAZ));
static double* a      = (double*)aligned_malloc_or_die(64, sizeof(double) * (NZ));
static double* x      = (double*)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double* z      = (double*)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double* p      = (double*)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double* q      = (double*)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));
static double* r      = (double*)aligned_malloc_or_die(64, sizeof(double) * (NA + 2));

```

```bash
 Benchmark completed
 VERIFICATION SUCCESSFUL
 Zeta is     2.2712745482631e+01
 Error is    6.2567753977913e-16


 CG Benchmark Completed
 class_npb       =                        B
 Size            =                    75000
 Iterations      =                       75
 Time in seconds =                    54.33
 Mop/s total     =                  1006.89
 Operation type  =           floating point
 Verification    =               SUCCESSFUL
 Version         =                      4.1
 Compiler ver    =                   13.3.0
 Compile date    =              25 Feb 2026

 Compile options:
    CC           = g++ -std=c++14
    CLINK        = $(CC)
    C_LIB        = -lm 
    C_INC        = -I../common 
    CFLAGS       = -O3 -mcmodel=medium
    CLINKFLAGS   = -O3 -mcmodel=medium
    RAND         = randdp
```

Not a big win either.


We start using a Full RCM implementation + CSR symetric permutation

```bash
   iteration           ||r||                 zeta
 Benchmark completed
 VERIFICATION SUCCESSFUL
 Zeta is     2.2712745482632e+01
 Error is    4.7864331793103e-14


 CG Benchmark Completed
 class_npb       =                        B
 Size            =                    75000
 Iterations      =                       75
 Time in seconds =                    53.26
 Mop/s total     =                  1027.30
 Operation type  =           floating point
 Verification    =               SUCCESSFUL
 Version         =                      4.1
 Compiler ver    =                   13.3.0
 Compile date    =              26 Feb 2026

 Compile options:
    CC           = g++ -std=c++14
    CLINK        = $(CC)
    C_LIB        = -lm 
    C_INC        = -I../common 
    CFLAGS       = -O3 -mcmodel=medium 
    CLINKFLAGS   = -O3 -mcmodel=medium 
    RAND         = randdp
```


On utilise les flags de compilation -mavx et -mfma

```bash
 NAS Parallel Benchmarks 4.1 Serial C++ version - CG Benchmark

 Size:       75000
 Iterations:    75
 Initialization time =           5.159 seconds

   iteration           ||r||                 zeta
 Benchmark completed
 VERIFICATION SUCCESSFUL
 Zeta is     2.2712745482632e+01
 Error is    4.7864331793103e-14


 CG Benchmark Completed
 class_npb       =                        B
 Size            =                    75000
 Iterations      =                       75
 Time in seconds =                    52.06
 Mop/s total     =                  1050.84
 Operation type  =           floating point
 Verification    =               SUCCESSFUL
 Version         =                      4.1
 Compiler ver    =                   13.3.0
 Compile date    =              26 Feb 2026

 Compile options:
    CC           = g++ -std=c++14
    CLINK        = $(CC)
    C_LIB        = -lm 
    C_INC        = -I../common 
    CFLAGS       = -O3 -mcmodel=medium -mavx -mfma
    CLINKFLAGS   = -O3 -mcmodel=medium -mavx -mfma
    RAND         = randdp
```




