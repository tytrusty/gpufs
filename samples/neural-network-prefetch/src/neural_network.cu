#include "neural_network.hh"
#include "nn_exception.hh"

// #define FILENAME_SIZE 32
// #include "fs_globals.cu.h"
#include "fs_calls.cu.h"
#include "host_loop.h"


// Prefetch file into memory
// 
__global__ void cuda_prefetch(char* A, char* fn, int filesize)
{
	int zfd = 0;
	zfd = gopen(fn, O_GRDONLY);

	for (size_t me = blockIdx.x * FS_BLOCKSIZE; me < filesize;
			me += FS_BLOCKSIZE * gridDim.x)
	{
		int toRead = min((unsigned int) FS_BLOCKSIZE,
				(unsigned int) (filesize - me));
		if (toRead != gread(zfd, me, toRead, (uchar*)A))
		{
			assert(NULL);
		}
	}
	gclose(zfd);
} 

NeuralNetwork::NeuralNetwork(float learning_rate, int filesize, int num_files, volatile GPUGlobals* gpuGlobals) :
	learning_rate(learning_rate), filesize(filesize), num_files(num_files), gpuGlobals(gpuGlobals) 
{ 
	cudaMalloc(&d_input_fn, FILENAME_SIZE);
	cudaMalloc(&d_prefetch_A, filesize * num_files);
}

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

void NeuralNetwork::prefetch(char* fn, int fidx) {
	int n = strlen(fn);
	cudaMemcpy(d_input_fn, fn, n+1, cudaMemcpyHostToDevice);

    //int
    int nblocks = 2;
    int nthreads = 512;
    cuda_prefetch<<<nblocks, nthreads,  0, gpuGlobals->streamMgr->kernelStream>>> ((char*)d_prefetch_A+fidx*filesize, d_input_fn, filesize);
	run_gpufs_handler(gpuGlobals,0);
}


Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;
	for (auto layer : layers) {
		Z = layer->forward(Z);
	}
	Y = Z;
	return Y;
} 

Matrix NeuralNetwork::forward(int fidx, Shape A_shape) {
	Matrix Z = layers[0]->forward((float*)(d_prefetch_A+fidx*filesize), A_shape);

	for (int i = 1; i < layers.size(); ++i) {
		Z = layers[i]->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);
	Matrix error = bce_cost.dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
	}
	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
