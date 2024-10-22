#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector);

std::vector<std::vector<double>> matrixMatrixMultiply(const std::vector<std::vector<double>> A, const std::vector<std::vector<double>> B);

#endif
