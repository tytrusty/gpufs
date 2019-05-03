#pragma once

#include <vector>
#include "nn_layer.hh"
#include "bce_cost.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	BCECost bce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;
	
	char* d_input_fn;
	

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(char* A_fn, Shape A_shape);
	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
