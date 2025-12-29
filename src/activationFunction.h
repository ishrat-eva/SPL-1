#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "Matrix.h"
#include <cmath>
#include <vector>
using namespace std;


double relu(double x) {
    return (x > 0) ? x : 0;
}



Matrix reluMatrix(const Matrix &input) {
    Matrix output(input.rows, input.cols);
    
    for(int i = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++) {
            output.data[i][j] = relu(input.data[i][j]);
        }
    }
    
    return output;
}


double reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}


Matrix reluDerivativeMatrix(const Matrix& input) {
    Matrix output(input.rows, input.cols);
    
    for(int i = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++) {
            output.data[i][j] = reluDerivative(input.data[i][j]);
        }
    }
    
    return output;
}



vector<double> softmax(const vector<double>& input) {
    vector<double> output(input.size());
    
    
    double maxVal = input[0];
    for(int i = 1; i < input.size(); i++) {
        if(input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    
  
    double sum = 0.0;
    for(int i = 0; i < input.size(); i++) {
        output[i] = exp(input[i] - maxVal);
        sum += output[i];
    }
    
 
    for(int i = 0; i < input.size(); i++) {
        output[i] = output[i] / sum;
    }
    
    return output;
}


void printProbabilities(const vector<double>& probs) {

    cout << "\nClass Probabilities:" << endl;
    
    
    int predicted = 0;
    double maxProb = probs[0];
    
    for(int i = 0; i < probs.size(); i++) {
        cout << "  Digit " << i << ": " << (probs[i] * 100.0) << "%" << endl;
        
        
        if(probs[i] > maxProb) {
            maxProb = probs[i];
            predicted = i;
        }
    }

    
    cout << "\nPredicted: " << predicted << " (Confidence: " << (maxProb * 100.0) << "%)" << std::endl;
}


#endif