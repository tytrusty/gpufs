#pragma once

#include <vector>
#include "nn_layer.hh"
#include "bce_cost.hh"
#include "fs_initializer.cu.h"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	BCECost bce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;
	
	char* d_input_fn;
    char* d_prefetch_A;
    int filesize;
    int num_files;

	volatile GPUGlobals* gpuGlobals = NULL;
public:
	NeuralNetwork(float learning_rate, int filesize, int num_files, volatile GPUGlobals* gpuGlobals);
	~NeuralNetwork();

    void prefetch(char* fn, int fidx);
	Matrix forward(int fidx, Shape A_shape);
	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
