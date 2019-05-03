#pragma once
#include "nn_layer.hh"
#include "fs_initializer.cu.h"

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix dA;

	Matrix A;
	Shape A_shape;
	char* A_fn = 0;
	bool is_fs_layer = false;
	volatile GPUGlobals* gpuGlobals = NULL;

	void initializeBiasWithZeros();
	void initializeWeightsRandomly();

	void computeAndStoreBackpropError(Matrix& dZ);
	void computeAndStoreLayerOutput();
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape W_shape);
	LinearLayer(std::string name, Shape W_shape, volatile GPUGlobals* gpuGlobals);
	~LinearLayer();

	Matrix& forward(Matrix& A);
	Matrix& forward(char* A_fn, Shape A_shape);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;
};
