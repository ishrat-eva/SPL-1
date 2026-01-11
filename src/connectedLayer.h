#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "Matrix.h"
#include "activationFunction.h"
#include <vector>
using namespace std;

class FullyConnectedLayer   {
public:
    int inputSize;
    int outputSize;
    
    Matrix weights;
    Matrix biases;
    
    Matrix lastInput;
    Matrix lastOutput;
    
    FullyConnectedLayer(int in, int out) 
        : inputSize(in), 
          outputSize(out),
          weights(out, in),
          biases(out, 1),
          lastInput(1, 1),
          lastOutput(1, 1)
    {

        weights.randomize(-0.5, 0.5);
        biases.fill(0.0);
    }


  vector<double> forward(vector<double> input) {
        // Convert input to matrix
        lastInput = Matrix(input.size(), 1);
        for(int i = 0; i < input.size(); i++) {
            lastInput.at(i, 0) = input[i];
        }
    
 
        
        
        lastOutput = weights.multiply(lastInput);
        for(int i = 0; i < outputSize; i++) {
            lastOutput.at(i, 0) += biases.at(i, 0);
        }
        
        
        lastOutput = reluMatrix(lastOutput);
        
        vector<double> output(outputSize);
        for(int i = 0; i < outputSize; i++) {
            output[i] = lastOutput.at(i, 0);
        }
        return output;
    }

    
    vector<double> forwardLinear(vector<double> input) {
        lastInput = Matrix(input.size(), 1);
        for(int i = 0; i < input.size(); i++) {
            lastInput.at(i, 0) = input[i];
        }
        
        lastOutput = weights.multiply(lastInput);
        for(int i = 0; i < outputSize; i++) {
            lastOutput.at(i, 0) += biases.at(i, 0);
        }
        
        vector<double> output(outputSize);
        for(int i = 0; i < outputSize; i++) {
            output[i] = lastOutput.at(i, 0);
        }
        return output;
    }
    
    
    void updateWeights(std::vector<double> errors, double learningRate)   {
      
        Matrix errorMatrix(errors.size(), 1);
        for(int i = 0; i < errors.size(); i++) {
            errorMatrix.at(i, 0) = errors[i];
        }
       
        Matrix weightChanges = errorMatrix.multiply(lastInput.transpose());
        weightChanges = weightChanges.multiplyScalar(learningRate);
        
       
        weights = weights.subtract(weightChanges);
        
        
        Matrix biasChanges = errorMatrix.multiplyScalar(learningRate);
        biases = biases.subtract(biasChanges);
    }
};

#endif