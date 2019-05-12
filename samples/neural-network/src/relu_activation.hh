#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Matrix& forward(Matrix& Z);

	// unused
	Matrix& forward(char* A_fn, Shape A_shape) { return A; }

	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};