#include "../include/DenseLayer.hpp"
#include "../include/Utils.hpp"
#include <random>
#include <algorithm>
#include <iostream>

DenseLayer::DenseLayer(int inputSize, int outputSize)
	: inputSize(inputSize), outputSize(outputSize) {
	initializeWeights();
	biases.resize(outputSize, 0.0);
	weightGradients.resize(outputSize, std::vector<double>(inputSize, 0.0));
	biasGradients.resize(outputSize, 0.0);
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
	lastInput = input;
	lastOutput = matrixVectorMultiply(weights, input);

	for (int i = 0; i < outputSize; i++) {
		lastOutput[i] += biases[i];
	}

	return lastOutput;
}

std::vector<double> DenseLayer::backward(const std::vector<double>& gradient) {
	std::vector<double> inputGradient(inputSize, 0.0);

	for (int i = 0; i < inputSize; i++) {
		for (int j = 0; j < outputSize; j++) {
			inputGradient[i] = weights[j][i] * gradient[j];
		}
	}

	this->weightGradients.resize(outputSize, std::vector<double>(inputSize, 0.0));
   	this->biasGradients.resize(outputSize, 0.0);

	for (int i = 0; i < outputSize; i++) {
		for (int j = 0; j < inputSize; j++) {
			weightGradients[i][j] = gradient[i] * lastInput[j];
		}
		biasGradients[i] = gradient[i];
	}

	return inputGradient;
}

void DenseLayer::updateWeights(double learningRate) {
	for (int i = 0; i < outputSize; i++) {
		for (int j = 0; j < inputSize; j++) {
			weights[i][j] -= learningRate * weightGradients[i][j];
		}
		biases[i] -= learningRate * biasGradients[i];
	}
}

void DenseLayer::setInputShape(int inputSize) {
	this->inputSize = inputSize;
	weights.resize(outputSize, std::vector<double>(inputSize));
	initializeWeights();	
}

int DenseLayer::getOutputSize() const {
	return outputSize;
}

void DenseLayer::initializeWeights() {
	std::random_device rd;
	std::mt19937 gen(rd());
	double scale = std::sqrt(2.0 / (inputSize + outputSize));
	std::normal_distribution<> d(0.0, scale);

	weights.resize(outputSize, std::vector<double>(inputSize));
	for (int i = 0; i < outputSize; i++) {
		for (int j = 0; j < inputSize; j++) {
			weights[i][j] = d(gen);
		}
	}	
}














