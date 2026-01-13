#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "Matrix.h"
#include <vector>
using namespace std;



class PoolingLayer {
public:
    int poolSize;        
    int inputWidth;      
    int inputHeight;
    int outputWidth;     
    int outputHeight;
    

    PoolingLayer(int pSize, int inW, int inH) {
        poolSize = pSize;
        inputWidth = inW;
        inputHeight = inH;
        
        outputWidth = inputWidth / poolSize;
        outputHeight = inputHeight / poolSize;
        
        cout << "Pooling Layer: " << inputWidth << "x" << inputHeight  << "->" << outputWidth << "x" << outputHeight 
                  << " (pool: " << poolSize << "x" << poolSize << ")" << endl;
    }
  
    vector<Matrix> forward(vector<Matrix> inputs) {
        vector<Matrix> outputs;
        
      
        for(int ch = 0; ch < inputs.size(); ch++) {
            Matrix output(outputHeight, outputWidth);
            
         
            for(int i = 0; i < outputHeight; i++) {
                for(int j = 0; j < outputWidth; j++) {
                    double maxValue = -999999.0;
                    
                for(int pi = 0; pi < poolSize; pi++) {
                    for(int pj = 0; pj < poolSize; pj++) {
                        int inputI = i * poolSize + pi;
                        int inputJ = j * poolSize + pj;
                        double value = inputs[ch].data[inputI][inputJ];
                    if(value > maxValue) {
                        maxValue = value;
                            }
                        }
                    }
                    
                    output.data[i][j] = maxValue;
                }
            }
            
            outputs.push_back(output);
        }
        
        return outputs;
    }
};

#endif