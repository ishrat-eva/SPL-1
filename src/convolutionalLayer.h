#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "Matrix.h"
#include "activationFunction.h"
#include <vector>
using namespace std;


class ConvolutionalLayer {
public:
    int numFilters;      
    int filterSize;      
    int inputWidth;      
    int inputHeight;     
    int outputWidth;     
    int outputHeight;    
    
    vector<Matrix> filters;  
    vector<double> biases;  

  Matrix lastInput;
    vector<Matrix> lastPreActivation; 

           ConvolutionalLayer(int numF, int fSize, int inW, int inH)
        : numFilters(numF), filterSize(fSize),
          inputWidth(inW), inputHeight(inH),
          lastInput(inH, inW)
    {
        
   
        outputWidth = inputWidth - filterSize + 1;
        outputHeight = inputHeight - filterSize + 1;
       
 


        for(int i = 0; i < numFilters; i++) {
            Matrix f(filterSize, filterSize);
            
            double scale = sqrt(2.0 / (filterSize * filterSize));
            for(int r = 0; r < filterSize; r++)
                for(int c = 0; c < filterSize; c++)
                    f.data[r][c] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
            filters.push_back(f);
            biases.push_back(0.0);
        }

        cout << "Convolutional Layer: " << inputWidth << "x" << inputHeight
             << "->" << outputWidth << "x" << outputHeight
             << " (" << numFilters << " filters)" << endl;
    }

    vector<Matrix> forward(Matrix input) {
        lastInput = input;
        lastPreActivation.clear();
        vector<Matrix> outputs;

        for(int f = 0; f < numFilters; f++) {
            Matrix preAct(outputHeight, outputWidth);
            for(int i = 0; i < outputHeight; i++) {
            for(int j = 0; j < outputWidth; j++) {
             double sum = biases[f];
                for(int fi = 0; fi < filterSize; fi++)
                    for(int fj = 0; fj < filterSize; fj++)
                     sum += input.data[i+fi][j+fj] * filters[f].data[fi][fj];
                    preAct.data[i][j] = sum;
             }
         }
     lastPreActivation.push_back(preAct);
     Matrix activated(outputHeight, outputWidth);
         for(int i = 0; i < outputHeight; i++)
            for(int j = 0; j < outputWidth; j++)
               activated.data[i][j] = relu(preAct.data[i][j]);
            outputs.push_back(activated);
        }
        return outputs;
    }

    void backward(vector<Matrix> dOut, double learningRate) {
        for(int f = 0; f < numFilters; f++) {
            Matrix delta(outputHeight, outputWidth);
            for(int i = 0; i < outputHeight; i++)
                for(int j = 0; j < outputWidth; j++)
                    delta.data[i][j] = dOut[f].data[i][j]
                        * reluDerivative(lastPreActivation[f].data[i][j]);

            for(int fi = 0; fi < filterSize; fi++) {
                for(int fj = 0; fj < filterSize; fj++) {
                    double grad = 0.0;
                    for(int i = 0; i < outputHeight; i++)
                    for(int j = 0; j < outputWidth; j++)
                     grad += delta.data[i][j] * lastInput.data[i+fi][j+fj];
                    filters[f].data[fi][fj] -= learningRate * grad;
                }
            }

     double biasGrad = 0.0;
         for(int i = 0; i < outputHeight; i++)
            for(int j = 0; j < outputWidth; j++)
              biasGrad += delta.data[i][j];
            biases[f] -= learningRate * biasGrad;
        }
    }
};
 

#endif




  

 


