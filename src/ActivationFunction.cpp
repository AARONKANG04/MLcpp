#include "../include/ActivationFunction.hpp"
#include <algorithm>


//Relu----------------------------

std::vector<double> ReLU::forward(const std::vector<double>& input) {
	lastOutput.resize(input.size());
	for (size_t i = 0; i < input.size(); i++) {
		lastOutput[i] = std::max(0.0, input[i]);
	}
	return lastOutput;
}

std::vector<double> ReLU::backward(const std::vector<double>& outputGradient) {
	std::vector<double> inputGradient(outputGradient.size());
	for (size_t i = 0; i < outputGradient.size(); i++) {
		inputGradient[i] = (lastOutput[i] > 0) ? outputGradient[i] : 0.0;
	}
	return inputGradient;
}

//Sigmoid-------------------------

std::vector<double> Sigmoid::forward(const std::vector<double>& input) {
	lastOutput.resize(input.size());
	for (size_t i = 0; i < input.size(); i++) {
		lastOutput[i] = 1.0 / (1.0 + std::exp(-input[i]));
	}
	return lastOutput;
}

std::vector<double> Sigmoid::backward(const std::vector<double>& outputGradient) {
	std::vector<double> inputGradient(outputGradient.size());
	for (size_t i = 0; i < outputGradient.size(); i++) {
		double sigmoidOutput = lastOutput[i];
		inputGradient[i] = outputGradient[i] * sigmoidOutput * (1.0 - sigmoidOutput);
	}
	return inputGradient;
}


//Tanh----------------------------

std::vector<double> Tanh::forward(const std::vector<double>& input) {
	lastOutput.resize(input.size());
	for (size_t i = 0; i < input.size(); i++) {
		lastOutput[i] = std::tanh(input[i]);
	}
	return lastOutput;
}

std::vector<double> Tanh::backward(const std::vector<double>& outputGradient) {
	std::vector<double> inputGradient(outputGradient.size());
	for (size_t i = 0; i < outputGradient.size(); i++) {
		double tanhOutput = lastOutput[i];
		inputGradient[i] = outputGradient[i] * (1.0 - tanhOutput * tanhOutput);
	}
	return inputGradient;
}














