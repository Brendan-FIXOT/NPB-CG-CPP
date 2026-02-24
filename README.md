# Commandes

## NPB-SER

### Compilation sur la taille B (demandé dans les consignes pour CG)
cd NPB-SER
make clean
make cg CLASS=B

### Vérif que le binaire existe
ls -lh bin/cg.B

### Run + temps + RAM
/usr/bin/time -v ./bin/cg.B

## NPB-SER

### Compilation sur la taille B (demandé dans les consignes pour CG)
cd NPB-OMP
make clean
make cg CLASS=B

### Vérif binaire
ls -lh bin/cg.B

### Run 1 thread
export OMP_NUM_THREADS=1
/usr/bin/time -v ./bin/cg.B

### Run 8 threads (adapte à tes coeurs physiques)
export OMP_NUM_THREADS=8
/usr/bin/time -v ./bin/cg.B

## NPB-PSTL

### Compilation sur la taille B (demandé dans les consignes pour CG)
cd NPB-PSTL
make clean
make cg CLASS=B

### Vérif binaire
ls -lh bin/cg.B

### Run Multi-Threads
/usr/bin/time -v ./bin/cg.B

### Run 1 thread
export TBB_NUM_THREADS=1
/usr/bin/time -v ./bin/cg.B

### Run 8 threads
export TBB_NUM_THREADS=8
/usr/bin/time -v ./bin/cg.B