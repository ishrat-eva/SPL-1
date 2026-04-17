#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "Matrix.h"
#include <vector>
using namespace std;

class PoolingLayer {
public:
    int poolSize;
    int inputWidth, inputHeight;
    int outputWidth, outputHeight;

    vector<vector<vector<pair<int,int>>>> maxPos; 

    PoolingLayer(int pSize, int inW, int inH)
        : poolSize(pSize), inputWidth(inW), inputHeight(inH)
    {
        outputWidth  = inputWidth  / poolSize;
        outputHeight = inputHeight / poolSize;
        cout << "Pooling Layer: " << inputWidth << "x" << inputHeight
             << "->" << outputWidth << "x" << outputHeight
             << " (pool: " << poolSize << "x" << poolSize << ")" << endl;
    }

    vector<Matrix> forward(vector<Matrix> inputs) {
        int numCh = inputs.size();
        maxPos.assign(numCh,
            vector<vector<pair<int,int>>>(outputHeight,
                vector<pair<int,int>>(outputWidth, {0,0})));

        vector<Matrix> outputs;
        for(int ch = 0; ch < numCh; ch++) {
            Matrix out(outputHeight, outputWidth);
            for(int i = 0; i < outputHeight; i++) {
             for(int j = 0; j < outputWidth; j++) {
                 double maxVal = -1e18;
                int mi = i*poolSize, mj = j*poolSize;
            for(int pi = 0; pi < poolSize; pi++) {
                for(int pj = 0; pj < poolSize; pj++) {
                 int ii = i*poolSize+pi, jj = j*poolSize+pj;
                     if(inputs[ch].data[ii][jj] > maxVal) {
                      maxVal = inputs[ch].data[ii][jj];
                         mi = ii; mj = jj;
                          }
                       }
                 }
             out.data[i][j] = maxVal;
              maxPos[ch][i][j] = {mi, mj};
             }
          }
      outputs.push_back(out);
    }
 return outputs;
}

   
    vector<Matrix> backward(vector<Matrix> dOut) {
        int numCh = dOut.size();
        vector<Matrix> dIn(numCh, Matrix(inputHeight, inputWidth));
        for(int ch = 0; ch < numCh; ch++) {
            for(int i = 0; i < outputHeight; i++) {
                for(int j = 0; j < outputWidth; j++) {
                    auto [mi, mj] = maxPos[ch][i][j];
                    dIn[ch].data[mi][mj] += dOut[ch].data[i][j];
                }
            }
        }
        return dIn;
    }
};

#endif