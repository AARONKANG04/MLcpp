#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP
#include <vector>
#include <cmath>

class ActivationFunction {
public:

	virtual ~ActivationFunction() {}

	virtual std::vector<double> forward(const std::vector<double>& input) = 0;

	virtual std::vector<double> backward(const std::vector<double>& gradient) = 0;

protected:
	std::vector<double> lastOutput;
};

class ReLU : public ActivationFunction {
public:
	std::vector<double> forward(const std::vector<double>& input) override;
	std::vector<double> backward(const std::vector<double>& gradient) override;
};

class Sigmoid : public ActivationFunction {
public:
	std::vector<double> forward(const std::vector<double>& input) override;
        std::vector<double> backward(const std::vector<double>& gradient) override;	
};

class Tanh : public ActivationFunction {
public:
	std::vector<double> forward(const std::vector<double>& input) override;
        std::vector<double> backward(const std::vector<double>& gradient) override;
};

#endif
