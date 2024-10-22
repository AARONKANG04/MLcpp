#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "Layer.hpp"

#include <vector>

class DenseLayer : public Layer {
public:
	DenseLayer(int inputSize, int outputSize);

	std::vector<double> forward(const std::vector<double>& input) override;

	std::vector<double> backward(const std::vector<double>& gradient) override;

	void updateWeights(double learningRate) override;

	void setInputShape(int inputSize) override;

	int getOutputSize() const override;

private:
	int inputSize;
	int outputSize;
	std::vector<std::vector<double>> weights; // outputSize x inputSize
	std::vector<double> biases;
	std::vector<double> lastInput;
	std::vector<double> lastOutput;
	std::vector<std::vector<double>> weightGradients;
	std::vector<double> biasGradients;

	void initializeWeights();
};

#endif
