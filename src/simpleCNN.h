
#ifndef SIMPLE_CNN_H
#define SIMPLE_CNN_H

#include "Matrix.h"
#include "convolutionalLayer.h"
#include "poolingLayer.h"
#include "connectedLayer.h"
#include "activationFunction.h"
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

class SimpleCNN {
public:
    ConvolutionalLayer conv1;
    PoolingLayer pool1;
    FullyConnectedLayer fc1;
    FullyConnectedLayer fc2;

    SimpleCNN()
        : conv1(8, 5, 28, 28),
          pool1(2, 24, 24),
          fc1(1152, 128),
          fc2(128, 10)
    {
        cout << "  1. Input:       28x28x1 (grayscale image)" << endl;
        cout << "  2. Conv Layer:  24x24x8 (8 feature maps, 5x5 filters)" << endl;
        cout << "  3. Pool Layer:  12x12x8 (2x2 max pooling)" << endl;
        cout << "  4. Flatten:     1152 neurons" << endl;
        cout << "  5. FC Layer 1:  128 neurons " << endl;
        cout << "  6. FC Layer 2:  10 neurons" << endl;
    }

    vector<double> predict(vector<double> imageVector) {
        Matrix image(28, 28);
        for(int i = 0; i < 28; i++)
            for(int j = 0; j < 28; j++)
                image.data[i][j] = imageVector[i*28+j];

        vector<Matrix> convOut = conv1.forward(image);
        vector<Matrix> poolOut = pool1.forward(convOut);

        vector<double> flat;
        for(auto& m : poolOut)
            for(int i = 0; i < 12; i++)
                for(int j = 0; j < 12; j++)
                    flat.push_back(m.data[i][j]);

        vector<double> fc1Out = fc1.forward(flat);
        vector<double> fc2Out = fc2.forwardLinear(fc1Out);
        return softmax(fc2Out);
    }

    void trainOne(vector<double> imageVector, int correctLabel, double learningRate) {
        Matrix image(28, 28);
        for(int i = 0; i < 28; i++)
            for(int j = 0; j < 28; j++)
                image.data[i][j] = imageVector[i*28+j];

        vector<Matrix> convOut = conv1.forward(image);
        vector<Matrix> poolOut = pool1.forward(convOut);

        vector<double> flat;
        for(auto& m : poolOut)
            for(int i = 0; i < 12; i++)
                for(int j = 0; j < 12; j++)
                    flat.push_back(m.data[i][j]);

        vector<double> fc1Out = fc1.forward(flat);
        vector<double> fc2Out = fc2.forwardLinear(fc1Out);
        vector<double> probs  = softmax(fc2Out);

     
        vector<double> dLoss(10);
        for(int i = 0; i < 10; i++)
            dLoss[i] = probs[i] - (i == correctLabel ? 1.0 : 0.0);

        vector<double> dFc1Out = fc2.updateWeights(dLoss, learningRate);
        vector<double> dFlat   = fc1.updateWeights(dFc1Out, learningRate);

        vector<Matrix> dPoolOut(8, Matrix(12, 12));
        int idx = 0;
        for(int ch = 0; ch < 8; ch++)
            for(int i = 0; i < 12; i++)
                for(int j = 0; j < 12; j++)
                    dPoolOut[ch].data[i][j] = dFlat[idx++];

        vector<Matrix> dConvOut = pool1.backward(dPoolOut);
        conv1.backward(dConvOut, learningRate);
    }

    int getPredictedDigit(vector<double> probs) {
        int best = 0;
        for(int i = 1; i < 10; i++)
            if(probs[i] > probs[best]) best = i;
        return best;
    }

    double getConfidence(vector<double> probs) {
        double best = probs[0];
        for(int i = 1; i < 10; i++)
            if(probs[i] > best) best = probs[i];
        return best;
    }
    bool saveWeights(string filename) {
        ofstream file(filename, ios::binary);
        if(!file.is_open()) {
            cout << "Cannot open file for saving: " << filename << endl;
            return false;
        }

        for(int f = 0; f < conv1.numFilters; f++) {
            for(int i = 0; i < conv1.filterSize; i++)
                for(int j = 0; j < conv1.filterSize; j++)
                    file.write((char*)&conv1.filters[f].data[i][j], sizeof(double));
            file.write((char*)&conv1.biases[f], sizeof(double));
        }

        for(int i = 0; i < fc1.outputSize; i++) {
            for(int j = 0; j < fc1.inputSize; j++)
                file.write((char*)&fc1.weights.data[i][j], sizeof(double));
            file.write((char*)&fc1.biases.data[i][0], sizeof(double));
        }

        for(int i = 0; i < fc2.outputSize; i++) {
            for(int j = 0; j < fc2.inputSize; j++)
                file.write((char*)&fc2.weights.data[i][j], sizeof(double));
            file.write((char*)&fc2.biases.data[i][0], sizeof(double));
        }

        file.close();
        cout << "Weights saved to: " << filename << endl;
        return true;
    }

    bool loadWeights(string filename) {
        ifstream file(filename, ios::binary);
        if(!file.is_open()) {
            cout << "No saved weights found at: " << filename << endl;
            return false;
        }

        for(int f = 0; f < conv1.numFilters; f++) {
            for(int i = 0; i < conv1.filterSize; i++)
                for(int j = 0; j < conv1.filterSize; j++)
                    file.read((char*)&conv1.filters[f].data[i][j], sizeof(double));
            file.read((char*)&conv1.biases[f], sizeof(double));
        }

        for(int i = 0; i < fc1.outputSize; i++) {
            for(int j = 0; j < fc1.inputSize; j++)
                file.read((char*)&fc1.weights.data[i][j], sizeof(double));
            file.read((char*)&fc1.biases.data[i][0], sizeof(double));
        }

        for(int i = 0; i < fc2.outputSize; i++) {
            for(int j = 0; j < fc2.inputSize; j++)
                file.read((char*)&fc2.weights.data[i][j], sizeof(double));
            file.read((char*)&fc2.biases.data[i][0], sizeof(double));
        }

        file.close();
        cout << "Weights loaded from: " << filename << endl;
        return true;
    }
};

#endif