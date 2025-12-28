#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

class Matrix {
public:
    vector<vector<double>> data;
    int rows;
    int cols;
   
    Matrix(int r, int c) {
        rows = r;
        cols = c;
        data.resize(r, vector<double>(c, 0.0));
    }
    
  
    double& at(int i, int j) {
        return data[i][j];
    }
    
    
    void fill(double value) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                data[i][j] = value;
            }
        }
    }
    
  
    void randomize(double min = -1.0, double max = 1.0) {
        static bool seeded = false;
        if(!seeded) {
            srand(time(0));
            seeded = true;
        }
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                data[i][j] = min + (rand() / (double)RAND_MAX) * (max - min);
            }
        }
    }
    
   
    Matrix add(const Matrix& other) {
        Matrix result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
   
    Matrix subtract(const Matrix& other) {
        Matrix result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }
    
   
    Matrix multiply(const Matrix& other) {
        Matrix result(rows, other.cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < other.cols; j++) {
                double sum = 0;
                for(int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }
    
    
    Matrix elementMultiply(const Matrix& other) {
        Matrix result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }
    
  
    Matrix multiplyScalar(double scalar) {
        Matrix result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }
    
  
    Matrix transpose() {
        Matrix result(cols, rows);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }
    
   
    Matrix applyFunction(double (*func)(double)) {
        Matrix result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }
    
  
    void print() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                cout << data[i][j] << " ";
            }
            cout <<endl;
        }
    }
};

#endif