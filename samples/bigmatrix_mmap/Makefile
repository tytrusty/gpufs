CUDA_DIR=/usr/local/cuda
LIBCUDA=-L$(CUDA_DIR)/lib64 -lcudart -lcudadevrt
CUDAINC=-I$(CUDA_DIR)/include
CC=/usr/bin/g++

GPUFSDIR?=../../
REG?=32

CXXFLAGS=-fopenmp -g --std=c++11
GPUFSINC= $(GPUFSDIR)/include

ifneq ($(dbg),1)
        MACROS=-DDEBUG_NOINLINE="" -DBLOCK_SIZE="32"
        CUFLAGS = --std=c++11 -O3 -rdc=true -maxrregcount=$(REG) -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_61
        LINKFLAGS = -O3 -rdc=true -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_61
        NVCC=$(CUDA_DIR)/bin/nvcc $(CUDAINC) $(MACROS)
        GPUFSLIBS = -lgpufs -L$(GPUFSDIR)/lib
else
        MACROS=-DDEBUG_NOINLINE="__noinline__" -DBLOCK_SIZE="32" -DMALLOC_STATS -DTIMING_STATS -DDEBUG
        CUFLAGS = --std=c++11 -O0 -rdc=true -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_61
        LINKFLAGS = -O0 -rdc=true -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_61
        NVCC=$(CUDA_DIR)/bin/nvcc $(CUDA_INCLUDE) $(MACROS)
        GPUFSLIBS = -lgpufsd -L$(GPUFSDIR)/lib
endif



MAIN = bigmatrix_cpu nofs nofs_naive nofs_managed nofs_unpinned fs

all: $(MAIN)

bigmatrix_cpu: 
	$(CXX) -o bigmatrix_cpu bigmatrix_cpu.cpp -O2 -fopenmp 

nofs: bigmatrix_nofs.cu
	$(NVCC) $(CUFLAGS) $(GPUFSLIBS) $(LIBCUDA) -o nofs bigmatrix_nofs.cu -I$(GPUFSINC)

nofs_unpinned: 
	$(NVCC) $(CUFLAGS) $(GPUFSLIBS) $(LIBCUDA) -o nofs_unpinned bigmatrix_nofs_unpinned.cu -I$(GPUFSINC)

nofs_managed: 
	$(NVCC) $(CUFLAGS) $(GPUFSLIBS) $(LIBCUDA) -o nofs_managed bigmatrix_nofs_managed.cu -I$(GPUFSINC)

nofs_naive:bigmatrix_nofs_naive.cu
	$(NVCC) $(CUFLAGS) $(GPUFSLIBS) $(LIBCUDA) -o nofs_naive bigmatrix_nofs_naive.cu -I$(GPUFSINC)

fs: fs.cu
	$(NVCC) $(GPUFSLIBS) $(LIBCUDA) $(CUFLAGS) -I$(GPUFSINC) -o fs fs.cu

bigmatrix_mmap.o: bigmatrix_mmap.cu
	$(NVCC) $(LINKFLAGS) $(GPUFSLIBS) $(LIBCUDA) -I$(GPUFSINC) -c $<

clean:
	\rm -f $(MAIN) *.o *~



