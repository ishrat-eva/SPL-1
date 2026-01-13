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
    
  
    ConvolutionalLayer(int numF, int fSize, int inW, int inH) {
        numFilters = numF;
        filterSize = fSize;
        inputWidth = inW;
        inputHeight = inH;
        
   
        outputWidth = inputWidth - filterSize + 1;
        outputHeight = inputHeight - filterSize + 1;
        
       
        for(int i = 0; i < numFilters; i++) {
            Matrix filter(filterSize, filterSize);
            filter.randomize(-0.1, 0.1);  
            filters.push_back(filter);
            biases.push_back(0.0);        
        }
        
        cout << "Convolutional Layer: " << inputWidth << "x" << inputHeight 
                  << "->" << outputWidth << "x" << outputHeight 
                  << " (" << numFilters << " filters)" <<endl;
    }
    
   
    vector<Matrix> forward(Matrix input) {
        vector<Matrix> outputs;
        
     
        for(int f = 0; f < numFilters; f++) {
            Matrix featureMap(outputHeight, outputWidth);
            
            for(int i = 0; i < outputHeight; i++) {
                for(int j = 0; j < outputWidth; j++) {
                  
                    double sum = 0.0;
                    
                for(int fi = 0; fi < filterSize; fi++) {
                    for(int fj = 0; fj < filterSize; fj++) {
                            sum += input.data[i + fi][j + fj] * filters[f].data[fi][fj];
                        }
                    }
                    
                
                    featureMap.data[i][j] = relu(sum + biases[f]);
                }
            }
            
            outputs.push_back(featureMap);
        }
        
        return outputs;
    }
};

#endif