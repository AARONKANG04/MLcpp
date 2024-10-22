#include "../include/Utils.hpp"

#include <stdexcept>

std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector) {
        if (matrix.empty() || matrix[0].size() != vector.size()) {
                throw std::invalid_argument("Matrix columns must match vector size.");
        }

        int rows = matrix.size();
        int cols = matrix[0].size();
        std::vector<double> result(rows, 0.0);

        for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                        result[i] += matrix[i][j] * vector[j];
                }
        }

        return result;
}


std::vector<std::vector<double>> matrixMatrixMultiply(const std::vector<std::vector<double>> A, const std::vector<std::vector<double>> B) {
        if (A.empty() || B.empty() || A[0].size() != B.size()) {
                throw std::invalid_argument("Matrix A's columns must match Matrix B's rows.");
        }

        int rowsA = A.size();
        int colsA = A[0].size();
        int colsB = B[0].size();
        std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));

        for (int i = 0; i < rowsA; i++) {
                for (int j = 0; j < colsB; j++) {
                        for (int k = 0; k < colsA; k++) {
                                result[i][j] += A[i][k] * B[k][j];
                        }
                }
        }

        return result;
}

