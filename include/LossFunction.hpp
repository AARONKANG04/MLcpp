#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include<vector>
#include<cmath>

class LossFunction {
public:
	virtual ~LossFunction() {}

	virtual double forward(const std::vector<double>& predictions, const std::vector<double>& targets) = 0;

	virtual std::vector<double> backward(const std::vector<double>& predictions, const std::vector<double>& targets) = 0; 

};


class MeanSquaredError : public LossFunction {
public:
	double forward(const std::vector<double>& predictions, const std::vector<double>& targets) override;
	std::vector<double> backward(const std::vector<double>& predictions, const std::vector<double>& targets) override;
};


class CrossEntropyLoss : public LossFunction {
public:
	double forward(const std::vector<double>& predictions, const std::vector<double>& targets) override;
	std::vector<double> backward(const std::vector<double>& predictions, const std::vector<double>& targets) override;
};

#endif
