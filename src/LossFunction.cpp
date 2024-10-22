#include "../include/LossFunction.hpp"
#include <cmath>
#include <stdexcept>

double MeanSquaredError::forward(const std::vector<double>& predictions, const std::vector<double>& targets) {
	if (predictions.size() != targets.size()) {
		throw std::invalid_argument("Size of predictions and targets must be the same.");
	}
	
	double sum = 0.0;
	for (size_t i = 0; i < predictions.size(); i++) {
		double diff = predictions[i] - targets[i];
		sum += diff * diff;
	}
	return sum / predictions.size();
}


std::vector<double> MeanSquaredError::backward(const std::vector<double>& predictions, const std::vector<double>& targets) {
	if (predictions.size() != targets.size()) {
		throw std::invalid_argument("Size of predictions and targets must be the same.");
	}

	std::vector<double> gradient(predictions.size());
	for (size_t i = 0; i < predictions.size(); i++) {
		gradient[i] = 2.0 * (predictions[i] - targets[i]) / predictions.size();
	}
	return gradient;
}




double CrossEntropyLoss::forward(const std::vector<double>& predictions, const std::vector<double>& targets) {
	if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Size of predictions and targets must be the same.");
	}
	
	double sum = 0.0;
	for (size_t i = 0; i < predictions.size(); i++) {
		double epsilon = 1e-12;
           	double pred = std::clamp(predictions[i], epsilon, 1.0 - epsilon);
           	sum += -targets[i] * std::log(pred);
	}

	return sum;
}

std::vector<double> CrossEntropyLoss::backward(const std::vector<double>& predictions, const std::vector<double>& targets) {
	if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Size of predictions and targets must be the same.");
        }

	std::vector<double> gradient(predictions.size());
	for (size_t i = 0; i < predictions.size(); i++) {
		double epsilon = 1e-12;
            	double pred = std::clamp(predictions[i], epsilon, 1.0 - epsilon);
        	gradient[i] = -targets[i] / pred;
	}

	return gradient;
}




