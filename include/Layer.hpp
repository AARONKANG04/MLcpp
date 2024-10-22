#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

class Layer {
public:
	virtual ~Layer() {}

	virtual std::vector<double> forward(const std::vector<double>& input) = 0; 

	virtual std::vector<double> backward(const std::vector<double>& gradient) = 0;

	virtual void updateWeights(double learningRate) = 0;
	
	virtual void setInputShape(int inputSize) = 0;

	virtual int getOutputSize() const = 0;
};

#endif
