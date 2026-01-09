#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;


int readInt(ifstream &file) {
    unsigned char bytes[4];
    file.read((char*)bytes, 4); 
    
    
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

class MNISTLoader {
public:
    vector<vector<double>> images;  
    vector<int> labels;                  
    int numImages;                            
    int imageSize;                            
    
    
    bool loadImages(const string &filename) {
        ifstream file(filename, ios::binary);
        
       
        if (!file.is_open()) {
            cerr << "ERROR: Cannot open " << filename << endl;
            cerr << "Make sure the file exists in the folder!" << endl;
            return false;
        }
        
       
        int magicNumber = readInt(file);  
        numImages = readInt(file);        
        int numRows = readInt(file);      
        int numCols = readInt(file);      
        imageSize = numRows * numCols;    
        
        cout << "Loading " << numImages << " images of size " << numRows << "x" << numCols << "..." << endl;
        
       
        images.resize(numImages);
        
        for (int i = 0; i < numImages; i++) {
            images[i].resize(imageSize);
            
            
            for (int j = 0; j < imageSize; j++) {
                unsigned char pixel;
                file.read((char*)&pixel, 1);
                
             
                images[i][j] = pixel / 255.0;
            }
            
          
            if ((i + 1) % 10000 == 0) {
                cout << "  Loaded " << (i + 1) << " images..." << endl;
            }
        }
        
        file.close();
        cout << "Successfully loaded all images!" << endl;
        return true;
    }
    
   
    bool loadLabels(const string &filename) {
      
          ifstream file(filename, ios::binary);
   
        if (!file.is_open()) {
            cout << "ERROR: Cannot open " << filename << endl;
            cout << "Make sure the file exists in the 'data' folder!" << endl;
            return false;
        }
        
       
        int magicNumber = readInt(file); 
        int numLabels = readInt(file);   
        
        cout << "Loading " << numLabels << " labels..." << endl;
        
        
        labels.resize(numLabels);
        
        for (int i = 0; i < numLabels; i++) {
            unsigned char label;
            file.read((char*)&label, 1);
            labels[i] = (int)label;  
        }
        
        file.close();
        cout << "Successfully loaded all labels!" << endl;
        return true;
    }
    
    
    vector<double> getImage(int index) {
        return images[index];
    }
    
   
    int getLabel(int index) {
        return labels[index];
    }
    
  
    void printImage(int index) {

    cout << "  Digit: " << labels[index] << endl;
   
    
    for(int row = 0; row < 28; row++) {
        for(int col = 0; col < 28; col++) {
            double pixel = images[index][row * 28 + col];
            
            if(pixel > 0.5)      cout << "##";
            else if(pixel > 0.2) cout << "..";
            else                 cout << "  ";
        }
        cout << endl;
    }

}
    
   
    int getNumImages() {
        return numImages;
    }
};

#endif