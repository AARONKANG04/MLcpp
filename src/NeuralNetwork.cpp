#include "../include/NeuralNetwork.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(std::unique_ptr<LossFunction> lossFunction) : lossFunction(std::move(lossFunction)), layerCount(0) {}

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer) {
	layers.push_back({layerCount++, std::move(layer)});
}

void NeuralNetwork::addActivation(std::unique_ptr<ActivationFunction> activation) {
	if (layers.empty()) {
		throw std::runtime_error("Cannot add activation function without adding a layer.");
	}
	
	activations.push_back({layerCount - 1, std::move(activation)});
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
	std::vector<double> currentInput = input;
	for (int i = 0; i < layers.size(); i++) {
		std::vector<double> layerOutput = layers[i].second->forward(currentInput);
		if (i < activations.size()) {
			layerOutput = activations[i].second->forward(layerOutput);
		}
		currentInput = layerOutput;
	}	
	return currentInput;
}

void NeuralNetwork::backward(const std::vector<double>& output, const std::vector<double>& target) {
	std::vector<double> currentGradient = lossFunction->backward(output, target);
	for (int i = layers.size() - 1; i >= 0; i--) {
		if (i < activations.size()) {
			currentGradient = activations[i].second->backward(currentGradient);
		}
	
		currentGradient = layers[i].second->backward(currentGradient);
	}

}

void NeuralNetwork::updateWeights(double learningRate) {
	for (auto& layer : layers) {
		layer.second->updateWeights(learningRate);
	}
}

int NeuralNetwork::getNumberOfLayers() const {
	return layerCount;
}


double NeuralNetwork::train(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
	std::vector<double> output = forward(input);

	double loss = lossFunction->forward(output, target);

	backward(output, target);

	updateWeights(learningRate);

	return loss;
}


std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
	return forward(input);
}




