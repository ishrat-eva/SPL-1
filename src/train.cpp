#include <iostream>
#include <fstream>
#include "Matrix.h"
#include "MNISTLoader.h"
#include "activationFunction.h"
#include "convolutionalLayer.h"
#include "poolingLayer.h"
#include "connectedLayer.h"
#include "simpleCNN.h"

using namespace std;

int main() {
    string weightsFile = "weights.bin";
    ifstream checkFile(weightsFile, ios::binary);
    bool alreadyTrained = checkFile.is_open();
    checkFile.close();

    cout << "Loading MNIST Dataset..." << endl;
    MNISTLoader trainData, testData;
    trainData.loadImages("data/train-images-idx3-ubyte");
    trainData.loadLabels("data/train-labels-idx1-ubyte");
    testData.loadImages("data/t10k-images-idx3-ubyte");
    testData.loadLabels("data/t10k-labels-idx1-ubyte");
    cout << " Dataset loaded!" << endl;
    cout << "    Training images: " << trainData.numImages << endl;
    cout << "    Test images:     " << testData.numImages << "\n" << endl;

    cout << "[2/4] Creating CNN Architecture..." << endl;
    SimpleCNN cnn;
    if(alreadyTrained) {
        cout << "\n[!] Found existing weights.bin!" << endl;
        cout << "    Loading saved weights (skipping training)..." << endl;
        cnn.loadWeights(weightsFile);

        cout << "\n Evaluating on Test Set..." << endl;
        int correct = 0;
        int testSize = testData.numImages;
        for(int i = 0; i < testSize; i++) {
            vector<double> img = testData.getImage(i);
            vector<double> probs = cnn.predict(img);
            if(cnn.getPredictedDigit(probs) == testData.getLabel(i)) correct++;
            if((i+1) % 2000 == 0)
                cout << "  [" << (i+1) << "/" << testSize << "] done..." << endl;
        }
        double acc = (correct / (double)testSize) * 100.0;
        cout << "  Test Accuracy: " << acc << "%" << endl;
        cout << "  Correct: " << correct << " / " << testSize << endl;
       
        return 0;
    }

    int    numEpochs    = 10;
    int    trainSize    = 60000;
    double learningRate = 0.005;

    cout << "\n Training Settings:" << endl;
    cout << "    Epochs:        " << numEpochs << endl;
    cout << "    Training size: " << trainSize << " images" << endl;
    cout << "    Learning rate: " << learningRate << endl;
    cout << "\n  Starting Training...\n" << endl;

    for(int epoch = 0; epoch < numEpochs; epoch++) {
        cout << "--- Epoch " << (epoch+1) << "/" << numEpochs << " ---" << endl;

        int epochCorrect = 0;
        for(int i = 0; i < trainSize; i++) {
            vector<double> image = trainData.getImage(i);
            int label = trainData.getLabel(i);
            cnn.trainOne(image, label, learningRate);

            vector<double> probs = cnn.predict(image);
            if(cnn.getPredictedDigit(probs) == label) epochCorrect++;

            if((i+1) % 5000 == 0) {
                double acc = (epochCorrect / (double)(i+1)) * 100.0;
                cout << "  [" << (i+1) << "/" << trainSize << "] Train Accuracy: " << acc << "%" << endl;
            }
        }

        int valCorrect = 0, valSize = 2000;
        for(int i = 0; i < valSize; i++) {
            vector<double> img = testData.getImage(i);
            if(cnn.getPredictedDigit(cnn.predict(img)) == testData.getLabel(i)) valCorrect++;
        }
        double valAcc = (valCorrect / (double)valSize) * 100.0;
        double trainAcc = (epochCorrect / (double)trainSize) * 100.0;
        cout << "  Epoch " << (epoch+1) << " -> Train: " << trainAcc << "% | Val: " << valAcc << "%" << endl;
        if((epoch+1) % 3 == 0) {
            learningRate *= 0.5;
            cout << "  [LR Decay] New LR: " << learningRate << endl;
        }
    }

    cout << "\nSaving trained weights..." << endl;
    cnn.saveWeights(weightsFile);
    cout << "weights.bin saved! Next run will load instantly.\n" << endl;

    cout << " Final Evaluation on Test Set..." << endl;
    int correct = 0;
    int testSize = testData.numImages;
    for(int i = 0; i < testSize; i++) {
        vector<double> img = testData.getImage(i);
        if(cnn.getPredictedDigit(cnn.predict(img)) == testData.getLabel(i)) correct++;
    }
    double finalAcc = (correct / (double)testSize) * 100.0;

    cout << "  Test Correct:  " << correct << " / " << testSize << endl;
    cout << "  Test Wrong:    " << (testSize - correct) << " / " << testSize << endl;
    cout << "  Test Accuracy: " << finalAcc << "%" << endl;

    return 0;
}



