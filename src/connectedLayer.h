#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "Matrix.h"
#include "activationFunction.h"
#include <vector>
using namespace std;

class FullyConnectedLayer {
public:
    int inputSize, outputSize;
    Matrix weights, biases;
    Matrix lastInput, lastPreActivation;
    bool usedRelu; 

    FullyConnectedLayer(int in, int out)
        : inputSize(in), outputSize(out),
          weights(out, in), biases(out, 1),
          lastInput(1,1), lastPreActivation(1,1),
          usedRelu(false)
    {
        double scale = sqrt(2.0 / in);
        for(int i = 0; i < out; i++)
            for(int j = 0; j < in; j++)
                weights.data[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
        biases.fill(0.0);
    }

    vector<double> forward(vector<double> input) {
        usedRelu = true;
        lastInput = Matrix(input.size(), 1);
        for(int i = 0; i < (int)input.size(); i++)
            lastInput.data[i][0] = input[i];

        Matrix z = weights.multiply(lastInput);
        for(int i = 0; i < outputSize; i++)
            z.data[i][0] += biases.data[i][0];
        lastPreActivation = z;

        Matrix a = reluMatrix(z);
        vector<double> out(outputSize);
        for(int i = 0; i < outputSize; i++)
            out[i] = a.data[i][0];
        return out;
    }

    vector<double> forwardLinear(vector<double> input) {
        usedRelu = false;
        lastInput = Matrix(input.size(), 1);
        for(int i = 0; i < (int)input.size(); i++)
            lastInput.data[i][0] = input[i];

        Matrix z = weights.multiply(lastInput);
        for(int i = 0; i < outputSize; i++)
            z.data[i][0] += biases.data[i][0];
        lastPreActivation = z;

        vector<double> out(outputSize);
        for(int i = 0; i < outputSize; i++)
            out[i] = z.data[i][0];
        return out;
    }
    vector<double> updateWeights(vector<double> errors, double learningRate) {
        vector<double> delta(outputSize);
        for(int i = 0; i < outputSize; i++) {
            if(usedRelu)
                delta[i] = errors[i] * reluDerivative(lastPreActivation.data[i][0]);
            else
                delta[i] = errors[i]; 
        }

        vector<double> dInput(inputSize, 0.0);
        for(int j = 0; j < inputSize; j++)
            for(int i = 0; i < outputSize; i++)
                dInput[j] += delta[i] * weights.data[i][j];

        for(int i = 0; i < outputSize; i++) {
            for(int j = 0; j < inputSize; j++)
                weights.data[i][j] -= learningRate * delta[i] * lastInput.data[j][0];
            biases.data[i][0] -= learningRate * delta[i];
        }

        return dInput;
    }
};

#endif