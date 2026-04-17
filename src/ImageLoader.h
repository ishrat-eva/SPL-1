
#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

class ImageLoader {
public:
    

    static std::vector<double> loadFromFile(std::string filename) {
        vector<double> image(784, 0.0);
        ifstream file(filename);
        
        if(!file.is_open()) {
            cout << "ERROR: Cannot open " << filename << endl;
            return image;
        }
        
        cout << "Loading: " << filename << endl;
        
       
        for(int i = 0; i < 784; i++) {
            file >> image[i];
        }
        
        file.close();
        cout << "[+] Loaded!" << std::endl;
        return image;
    }
    

    static void saveToFile(std::vector<double> image, std::string filename) {
        ofstream file(filename);
        
        
        for(int i = 0; i < 784; i++) {
            file << image[i] << std::endl;
        }
        
        file.close();
        cout << "[+] Saved to: " << filename << std::endl;
    }
    
    
    static vector<double> makeSeven() {
        vector<double> image(784, 0.0);
        
      
        for(int j = 5; j < 23; j++) {
            image[3 * 28 + j] = 1.0;
        }
        
     
        for(int i = 4; i < 25; i++) {
            int j = 22 - (i - 4) / 2;
            image[i * 28 + j] = 1.0;
        }
        
        return image;
    }
    
    static  vector<double> makeZero() {
        vector<double> image(784, 0.0);
        
      
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                int dx = i - 14;
                int dy = j - 14;
                int dist = dx*dx + dy*dy;
                
                if(dist > 64 && dist < 144) {
                    image[i * 28 + j] = 1.0;
                }
            }
        }
        
        return image;
    }
    
    
    static vector<double> makeOne() {
        vector<double> image(784, 0.0);
        
        
        for(int i = 3; i < 25; i++) {
            image[i * 28 + 14] = 1.0;
        }
        
        return image;
    }
    
   
    static void showImage(std::vector<double> image) {

        
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                double pixel = image[i * 28 + j];
                
                if(pixel > 0.5) cout << "##";
                else if(pixel > 0.2) cout << "..";
                else std::cout << "  ";
            }
            cout << endl;
        }
        

    }
};

#endif