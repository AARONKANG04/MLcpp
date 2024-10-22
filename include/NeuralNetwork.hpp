#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <memory>
#include <utility>
#include "Layer.hpp"
#include "ActivationFunction.hpp"
#include "LossFunction.hpp"


class NeuralNetwork {
public:
	NeuralNetwork(std::unique_ptr<LossFunction> lossFunction);

	void addLayer(std::unique_ptr<Layer> layer);
	
	void addActivation(std::unique_ptr<ActivationFunction> activation);

	std::vector<double> forward(const std::vector<double>& input);

	void backward(const std::vector<double>& output, const std::vector<double>& target);
	
	void updateWeights(double learningRate);

	double train(const std::vector<double>& input, const std::vector<double>& target, double learningRate);

	std::vector<double> predict(const std::vector<double>& input);

	int getNumberOfLayers() const;
private:
	std::vector<std::pair<int, std::unique_ptr<Layer>>> layers;
	std::vector<std::pair<int, std::unique_ptr<ActivationFunction>>> activations;
	std::unique_ptr<LossFunction> lossFunction;
	int layerCount = 0;
};

#endif
