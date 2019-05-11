#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"
#include "sigmoid_activation.hh"
#include "nn_exception.hh"
#include "bce_cost.hh"
#include <ctime>
#include <cstdio>
#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#include "coordinates_dataset.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {
    printf("Running regular cuda nn\n");

	srand( time(NULL) );

	CoordinatesDataset dataset(10000, 21);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

    auto t1 = Clock::now();

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward(dataset.getBatch("input"));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
                        << ", Time (s): "
                        << (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t1).count()) / 1000.0
						<< std::endl;
            t1 = Clock::now();
		}
	}

	// compute accuracy
	//Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	//Y.copyDeviceToHost();

	//float accuracy = computeAccuracy(
	//		Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	//std::cout 	<< "Accuracy: " << accuracy << std::endl;

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
