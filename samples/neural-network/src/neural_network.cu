#include "neural_network.hh"
#include "nn_exception.hh"

#define FILENAME_SIZE 32

//CUDA_SAFE_CALL(cudaMemcpy(d_filename, h_filename, n + 1, cudaMemcpyHostToDevice));


NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ 
	cudaMalloc(&d_input_fn, FILENAME_SIZE);
}

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;
	for (auto layer : layers) {
		Z = layer->forward(Z);
	}
	Y = Z;
	return Y;
} 

Matrix NeuralNetwork::forward(char* A_fn, Shape A_shape) {
	int n = strlen(A_fn);
	cudaMemcpy(d_input_fn, A_fn, n+1, cudaMemcpyHostToDevice);

	Matrix Z = layers[0]->forward(d_input_fn, A_shape);

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
