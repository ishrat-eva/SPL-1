#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <filesystem>

#include "Matrix.h"
#include "MNISTLoader.h"
#include "activationFunction.h"
#include "convolutionalLayer.h"
#include "poolingLayer.h"
#include "connectedLayer.h"
#include "Micrgenerator.h"
#include "simpleCNN14.h"
using namespace std;

namespace fs = filesystem;
using namespace std;

static const string WEIGHTS_OUT      = "weights_micr_real.bin";
static const string SYMBOL_FOLDER    = "symbols"; 
static const int    AUG_PER_REAL     = 40;
static const int    SYNTH_PER_SYMBOL = 500;
static const int    EPOCHS           = 15;
static const double LR_INIT          = 0.003;
 

static double randF(mt19937& rng, double lo, double hi) {
    return lo + (double)(rng()-rng.min())/(rng.max()-rng.min())*(hi-lo);
}
 
static vector<double> gaussBlur(const vector<double>& img, double sig) {
    vector<double> out(784, 0.0);
    double k[3][3], s=0;
    for(int r=-1;r<=1;r++) for(int c=-1;c<=1;c++) {
        k[r+1][c+1]=exp(-(r*r+c*c)/(2*sig*sig)); s+=k[r+1][c+1];
    }
    for(int r=0;r<28;r++) for(int c=0;c<28;c++) {
        double v=0;
        for(int kr=-1;kr<=1;kr++) for(int kc=-1;kc<=1;kc++) {
            int nr=r+kr, nc=c+kc;
            if(nr>=0&&nr<28&&nc>=0&&nc<28) v+=img[nr*28+nc]*k[kr+1][kc+1];
        }
        out[r*28+c]=v/s;
    }
    return out;
}
 
static vector<double> rotateImg(const vector<double>& img, double deg) {
    vector<double> out(784, 0.0);
    double a=deg*M_PI/180.0, ca=cos(a), sa=sin(a), cx=13.5, cy=13.5;
    for(int r=0;r<28;r++) for(int c=0;c<28;c++) {
        double dr=r-cx, dc=c-cy;
        int sr=(int)round(dr*ca+dc*sa+cx);
        int sc=(int)round(-dr*sa+dc*ca+cy);
        if(sr>=0&&sr<28&&sc>=0&&sc<28) out[r*28+c]=img[sr*28+sc];
    }
    return out;
}
 
static vector<double> scaleImg(const vector<double>& img, double f) {
    vector<double> out(784, 0.0);
    double cx=13.5, cy=13.5;
    for(int r=0;r<28;r++) for(int c=0;c<28;c++) {
        int sr=(int)round((r-cx)/f+cx);
        int sc=(int)round((c-cy)/f+cy);
        if(sr>=0&&sr<28&&sc>=0&&sc<28) out[r*28+c]=img[sr*28+sc];
    }
    return out;
}
 
static vector<double> shiftImg(const vector<double>& img, int dr, int dc) {
    vector<double> out(784, 0.0);
    for(int r=0;r<28;r++) for(int c=0;c<28;c++) {
        int nr=r+dr, nc=c+dc;
        if(nr>=0&&nr<28&&nc>=0&&nc<28) out[nr*28+nc]=img[r*28+c];
    }
    return out;
}
 
static void addNoise(vector<double>& img, double p, mt19937& rng) {
    uniform_real_distribution<double> d(0.0,1.0);
    for(int i=0;i<784;i++) {
        double r=d(rng);
        if(r<p) img[i]=1.0; else if(r<p*2) img[i]=0.0;
    }
}
 
static vector<double> augment(const vector<double>& src, mt19937& rng) {
    auto img=src;
    img=gaussBlur(img, randF(rng,0.3,1.0));
    img=rotateImg(img, randF(rng,-8.0,8.0));
    img=scaleImg(img,  randF(rng,0.85,1.15));
    img=shiftImg(img,  (int)randF(rng,-2,2),(int)randF(rng,-2,2));
    addNoise(img, 0.01, rng);
    double mx=*max_element(img.begin(),img.end());
    if(mx>0.01) for(auto& v:img) v/=mx;
    return img;
}
 

static vector<vector<double>> loadFolder(const string& folder) {
    vector<vector<double>> samples;
    if(!fs::exists(folder)) {
        cout << "  [!] Folder not found: " << folder << endl;
        return samples;
    }
    for(auto& e : fs::directory_iterator(folder)) {
        if(e.path().extension()!=".txt") continue;
        ifstream f(e.path().string());
        if(!f.is_open()) continue;
        vector<double> img(784, 0.0);
        bool ok=true;
        for(int i=0;i<784;i++) {
            if(!(f>>img[i])) { ok=false; break; }
        }
        if(ok) samples.push_back(img);
    }
    return samples;
}
 

int main() {
   
    mt19937 rng(42);
 
    cout << "[1/5] Loading MNIST..." << endl;
    MNISTLoader trainData, testData;
    if(!trainData.loadImages("data/train-images-idx3-ubyte") ||
       !trainData.loadLabels("data/train-labels-idx1-ubyte") ||
       !testData.loadImages ("data/t10k-images-idx3-ubyte")  ||
       !testData.loadLabels ("data/t10k-labels-idx1-ubyte")) {
        cerr << " MNIST load failed! data/ folder check koro." << endl;
        return 1;
    }
    cout << "MNIST: " << trainData.numImages << " samples\n" << endl;
 
    struct SymInfo { string folder; int label; const char* name; int genIdx; };
    vector<SymInfo> syms = {
        { SYMBOL_FOLDER+"/transit", 10, "Transit", 0 },
        { SYMBOL_FOLDER+"/amount",  11, "Amount",  1 },
        { SYMBOL_FOLDER+"/onus",    12, "On-Us",   2 },
    };
 
    cout << "[2/5] Loading real symbols from extracted_symbols/..." << endl;
    vector<vector<vector<double>>> realSamples(3);
    for(int i=0;i<3;i++) {
        realSamples[i]=loadFolder(syms[i].folder);
        cout << "  " << syms[i].name << " : "
             << realSamples[i].size() << " samples" << endl;
    }
    cout << endl;
 
    cout << "[3/5] Building dataset..." << endl;
    vector<vector<double>> symImages;
    vector<int>            symLabels;
 
    for(int i=0;i<3;i++) {
        int realN=(int)realSamples[i].size(), augN=0;
 
        for(auto& s:realSamples[i]) {
            symImages.push_back(s);
            symLabels.push_back(syms[i].label);
            for(int a=0;a<AUG_PER_REAL;a++) {
                symImages.push_back(augment(s,rng));
                symLabels.push_back(syms[i].label);
                augN++;
            }
        }
 
        auto synth=MICRGenerator::generateSamples(syms[i].genIdx, SYNTH_PER_SYMBOL);
        for(auto& img:synth) {
            symImages.push_back(img);
            symLabels.push_back(syms[i].label);
        }
 
        cout << "  " << syms[i].name
             << ": " << realN << " real"
             << " + " << augN << " aug"
             << " + " << SYNTH_PER_SYMBOL << " synthetic"
             << " = " << (realN+augN+SYNTH_PER_SYMBOL) << endl;
    }
 
    for(auto& img : MICRGenerator::generateSamples(3, SYNTH_PER_SYMBOL)) {
        symImages.push_back(img);
        symLabels.push_back(13);
    }
    cout << "  Dash: 0 real + " << SYNTH_PER_SYMBOL << " synthetic\n" << endl;
 
    cout << "[4/5] Combining all data..." << endl;
    vector<vector<double>> allImages;
    vector<int>            allLabels;
    for(int i=0;i<trainData.numImages;i++) {
        allImages.push_back(trainData.getImage(i));
        allLabels.push_back(trainData.getLabel(i));
    }
    for(size_t i=0;i<symImages.size();i++) {
        allImages.push_back(symImages[i]);
        allLabels.push_back(symLabels[i]);
    }
    vector<int> idx((int)allImages.size());
    iota(idx.begin(),idx.end(),0);
    shuffle(idx.begin(),idx.end(),rng);
    cout << "[+] Total: " << allImages.size()
         << " (" << trainData.numImages << " digits + "
         << symImages.size() << " symbols)\n" << endl;
 
    cout << "[5/5] Training..." << endl;
    SimpleCNN14 cnn;
    double lr=LR_INIT;
    int total=(int)allImages.size();
 
    for(int epoch=0;epoch<EPOCHS;epoch++) {
        cout << "\nEpoch " << (epoch+1) << "/" << EPOCHS
             << "  lr=" << lr << " ---" << endl;
        shuffle(idx.begin(),idx.end(),rng);
 
        int correct=0;
        for(int k=0;k<total;k++) {
            cnn.trainOne(allImages[idx[k]], allLabels[idx[k]], lr);
            auto probs=cnn.predict(allImages[idx[k]]);
            if(cnn.getPredictedClass(probs)==allLabels[idx[k]]) correct++;
            if((k+1)%10000==0)
                cout << "  [" << (k+1) << "/" << total << "]  "
                     << fixed << setprecision(1)
                     << (correct*100.0/(k+1)) << "%" << endl;
        }
 
        int vOk=0, vTotal=0;
        for(int i=0;i<1000;i++) {
            auto p=cnn.predict(testData.getImage(i));
            if(cnn.getPredictedClass(p)==testData.getLabel(i)) vOk++;
            vTotal++;
        }
        for(int i=0;i<3;i++) for(auto& s:realSamples[i]) {
            auto p=cnn.predict(s);
            if(cnn.getPredictedClass(p)==syms[i].label) vOk++;
            vTotal++;
        }
        cout << "  Train: " << (correct*100.0/total)
             << "%  |  Val: " << (vOk*100.0/vTotal) << "%" << endl;
 
        if((epoch+1)%4==0) { lr*=0.5; cout << "  LR -> " << lr << endl; }
    }
 
    cout << "\n[Saving] " << WEIGHTS_OUT << endl;
    cnn.saveWeights(WEIGHTS_OUT);
    cout << "Saved!\n" << endl;
    int dc=0;
    for(int i=0;i<testData.numImages;i++) {
        auto p=cnn.predict(testData.getImage(i));
        if(cnn.getPredictedClass(p)==testData.getLabel(i)) dc++;
    }
    cout << "  Digits : " << fixed << setprecision(1)
         << (dc*100.0/testData.numImages)
         << "% (" << dc << "/" << testData.numImages << ")" << endl;
 
    for(int i=0;i<3;i++) {
        if(realSamples[i].empty()) continue;
        int sc=0, st=(int)realSamples[i].size();
        for(auto& s:realSamples[i]) {
            auto p=cnn.predict(s);
            if(cnn.getPredictedClass(p)==syms[i].label) sc++;
        }
        cout << "  " << syms[i].name << " : "
             << (sc*100.0/st) << "% (" << sc << "/" << st << ")" << endl;
    }
  
    return 0;
}
