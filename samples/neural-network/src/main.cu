#include <iostream>
#include <time.h>

#include "fs_initializer.cu.h"
#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"
#include "sigmoid_activation.hh"
#include "nn_exception.hh"
#include "bce_cost.hh"

#include "coordinates_dataset.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

void init_device_app()
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30);
}

int main() {
	// GPUFS setup 
	int device = 0;
	char* gpudev = getenv("GPUDEVICE");
	if (gpudev != NULL)
		device = atoi(gpudev);

	cudaSetDevice(device);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	printf("Running on device %d: \"%s\"\n", device, deviceProp.name);
	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals);
	init_device_app();

	srand( time(NULL) );

	CoordinatesDataset dataset(100, 21);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30), gpuGlobals));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

    printf("Beginning training\n");
	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

        // Not concerned with actually training, moreso evaluating effect
        // of using gpufs for loading input matrix on runtime performance,
        // so just usin junk
		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward("input", Shape(100, 2)); // dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}
	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}
