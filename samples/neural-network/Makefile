###########################################################
## USER SPECIFIC DIRECTORIES ##

# REF: https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable/blob/master/Makefile

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

# GPUFS dirs
GPUFS_ROOT_DIR=../../
GPUFSINC=$(GPUFS_ROOT_DIR)/include
GPUFSLIBS = -lgpufs -L$(GPUFS_ROOT_DIR)/lib


##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
MACROS=-DDEBUG_NOINLINE="" -DBLOCK_SIZE="32"
NVCC=$(CUDA_ROOT_DIR)/bin/nvcc $(MACROS)
NVCC_FLAGS=--std=c++11 -O3 -rdc=true -maxrregcount=$(REG) -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_61
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcudadevrt 

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = ./src/

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = nn

# Object files:
OBJS = $(OBJ_DIR)/bce_cost.o $(OBJ_DIR)/coordinates_dataset.o $(OBJ_DIR)/linear_layer.o $(OBJ_DIR)/main.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/neural_network.o $(OBJ_DIR)/relu_activation.o $(OBJ_DIR)/shape.o $(OBJ_DIR)/sigmoid_activation.o


##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) -I$(GPUFSINC) $(GPUFSLIBS) 

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.hh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) -I$(GPUFSINC)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) -I$(GPUFSINC)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)





