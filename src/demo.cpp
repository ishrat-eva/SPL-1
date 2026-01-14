#include <iostream>
#include <vector>
#include "Matrix.h"
#include "MNISTLoader.h"
#include "connectedLayer.h"
#include "convolutionalLayer.h"
#include "poolingLayer.h"

using namespace std;

int getPrediction(const vector<double>& out) {
    int idx = 0;
    for(int i = 1; i < out.size(); i++)
        if(out[i] > out[idx]) idx = i;
    return idx;
}

int main() {
    cout << "\nHandwritten Digit Recognition\n\n";

    MNISTLoader test;
    test.loadImages("data/t10k-images-idx3-ubyte");
    test.loadLabels("data/t10k-labels-idx1-ubyte");
    cout << "Loaded " << test.numImages << " test images\n";

    ConvolutionalLayer conv(8, 5, 28, 28);
    PoolingLayer pool(2, 24, 24);
    FullyConnectedLayer fc(1152, 10);

    while(true) {
        int choice;
        cout << "\n1. Predict a digit";
        cout << "\n0. Exit";
        cout << "\nEnter choice: ";
        cin >> choice;

        if(choice == 0) break;
        if(choice != 1) {
            cout << "Invalid choice!\n";
            continue;
        }

        int idx;
        cout << "Enter image index (0 - " << test.numImages-1 << "): ";
        cin >> idx;

        if(idx < 0 || idx >= test.numImages) {
            cout << "Invalid index!\n";
            continue;
        }

        vector<double> imgVec = test.getImage(idx);
        int actual = test.getLabel(idx);

        Matrix img(28,28);
        for(int i = 0; i < 784; i++)
            img.data[i/28][i%28] = imgVec[i];

        vector<Matrix> c = conv.forward(img);
        vector<Matrix> p = pool.forward(c);

        vector<double> flat;
        for(auto &m : p)
            for(int i = 0; i < 12; i++)
                for(int j = 0; j < 12; j++)
                    flat.push_back(m.data[i][j]);

        vector<double> out = fc.forwardLinear(flat);
        int predicted = getPrediction(out);

        cout << "\nActual Digit   : " << actual << endl;
        cout << "Predicted Digit: " << predicted
             << " (not trained)\n";
    }

    cout << "\nProgram exited successfully.\n";
    return 0;
}
