#include <iostream>
#include <vector>
#include <cmath>
#include "../include/Layer.hpp"
#include "../include/DenseLayer.hpp"
#include "../include/ActivationFunction.hpp"
#include "../include/LossFunction.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/Utils.hpp"

int main() {
	NeuralNetwork model(std::make_unique<MeanSquaredError>());

	model.addLayer(std::make_unique<DenseLayer>(2, 10));
	model.addActivation(std::make_unique<ReLU>());
	
	model.addLayer(std::make_unique<DenseLayer>(10, 1));
	model.addActivation(std::make_unique<Sigmoid>());

	std::vector<std::vector<double>> inputs = {
        	{0, 0},
        	{0, 1},
        	{1, 0},
        	{1, 1}
   	};

	std::vector<std::vector<double>> targets = {
        	{0},
        	{1},
        	{1},
        	{0}
   	 };

	double learningRate = 0.1;
    	int epochs = 5000000;

	for (int epoch = 0; epoch < epochs; epoch++) {
		double totalLoss = 0.0;

		for (size_t i = 0; i < inputs.size(); i++) {
			double loss = model.train(inputs[i], targets[i], learningRate);
			totalLoss += loss;
		}
			
		if (epoch % 1000 == 0) {
			std::cout << "Epoch: " << epoch << " | Loss: " << totalLoss << std::endl;
		}
	}

	return 0;
}
