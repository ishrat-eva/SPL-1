#ifndef MICR_GENERATOR_H
#define MICR_GENERATOR_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class MICRGenerator {
public:

    static void setPixel(vector<double>& img, int r, int c, double val=1.0) {
        if(r>=0&&r<28&&c>=0&&c<28)
            img[r*28+c]=max(img[r*28+c],val);
    }

    static void rect(vector<double>& img, int r1, int c1, int r2, int c2) {
        for(int r=r1;r<=r2;r++)
            for(int c=c1;c<=c2;c++)
                setPixel(img,r,c);
    }


    static vector<double> makeTransit(int t=2) {
        vector<double> img(784, 0.0);
       
        rect(img,  3,  6,  24,  6+t);
        rect(img,  3, 17,  24, 17+t+2);
        rect(img,  3, 15,  3+t, 21);
        rect(img, 22, 15,  24,  21);
        return img;
    }

    static vector<double> makeAmount(int t=2) {
        vector<double> img(784, 0.0);
        rect(img, 13, 18,  24, 18+t+1);
        rect(img, 13, 16,  13+t, 21);
        return img;
    }

    static vector<double> makeOnUs(int t=2) {
        vector<double> img(784, 0.0);
        rect(img,  3, 13,  24, 13+t);
        rect(img,  8,  6,  21,  6+t);
        rect(img, 10, 19,  24, 19+t);
        rect(img, 10, 22,  24, 22+t);
        return img;
    }

    static vector<double> makeDash(int t=2) {
        vector<double> img(784, 0.0);
        rect(img, 10,  5,  20,  5+t);
        rect(img, 10, 12,  20, 12+t);
        rect(img, 10, 19,  20, 19+t);
        return img;
    }

    static vector<double> gaussianBlur(vector<double>& img, double sigma) {
        vector<double> out(784,0.0);
        double k[3][3],sum=0;
        for(int r=-1;r<=1;r++) for(int c=-1;c<=1;c++){
            k[r+1][c+1]=exp(-(r*r+c*c)/(2*sigma*sigma)); sum+=k[r+1][c+1];
        }
        for(int r=0;r<28;r++) for(int c=0;c<28;c++){
            double v=0;
            for(int kr=-1;kr<=1;kr++) for(int kc=-1;kc<=1;kc++){
                int nr=r+kr,nc=c+kc;
                if(nr>=0&&nr<28&&nc>=0&&nc<28) v+=img[nr*28+nc]*k[kr+1][kc+1];
            }
            out[r*28+c]=v/sum;
        }
        return out;
    }

    static vector<double> rotate(vector<double>& img, double deg) {
        vector<double> out(784,0.0);
        double a=deg*M_PI/180.0,ca=cos(a),sa=sin(a),cx=13.5,cy=13.5;
        for(int r=0;r<28;r++) for(int c=0;c<28;c++){
            double dr=r-cx,dc=c-cy;
            int sr=(int)round(dr*ca+dc*sa+cx);
            int sc=(int)round(-dr*sa+dc*ca+cy);
            if(sr>=0&&sr<28&&sc>=0&&sc<28) out[r*28+c]=img[sr*28+sc];
        }
        return out;
    }

    static vector<double> scaleImg(vector<double>& img, double f) {
        vector<double> out(784,0.0);
        double cx=13.5,cy=13.5;
        for(int r=0;r<28;r++) for(int c=0;c<28;c++){
            int sr=(int)round((r-cx)/f+cx);
            int sc=(int)round((c-cy)/f+cy);
            if(sr>=0&&sr<28&&sc>=0&&sc<28) out[r*28+c]=img[sr*28+sc];
        }
        return out;
    }

    static vector<double> translate(vector<double>& img, int dr, int dc) {
        vector<double> out(784,0.0);
        for(int r=0;r<28;r++) for(int c=0;c<28;c++){
            int nr=r+dr,nc=c+dc;
            if(nr>=0&&nr<28&&nc>=0&&nc<28) out[nr*28+nc]=img[r*28+c];
        }
        return out;
    }

    static vector<double> inkBleed(vector<double>& img, double amt) {
        vector<double> out=img;
        for(int r=1;r<27;r++) for(int c=1;c<27;c++)
            if(img[r*28+c]>0.5)
                for(int dr=-1;dr<=1;dr++) for(int dc=-1;dc<=1;dc++)
                    out[(r+dr)*28+(c+dc)]=max(out[(r+dr)*28+(c+dc)],amt);
        return out;
    }

    static void saltPepper(vector<double>& img, double prob) {
        for(int i=0;i<784;i++){
            double r=(double)rand()/RAND_MAX;
            if(r<prob)        img[i]=1.0;
            else if(r<prob*2) img[i]=0.0;
        }
    }

    static double randF(double lo, double hi) {
        return lo+(double)rand()/RAND_MAX*(hi-lo);
    }

    static vector<vector<double>> generateSamples(int symbolType, int count) {
        vector<vector<double>> samples;
        for(int k=0;k<count;k++){
            int thick=1+rand()%3;
            vector<double> base;
            if(symbolType==0)      base=makeTransit(thick);
            else if(symbolType==1) base=makeAmount(thick);
            else if(symbolType==2) base=makeOnUs(thick);
            else                   base=makeDash(thick);

            base=inkBleed(base,    randF(0.2,0.5));
            base=gaussianBlur(base,randF(0.5,1.2));
            base=rotate(base,      randF(-6.0,6.0));
            base=scaleImg(base,    randF(0.85,1.15));
            base=translate(base,   (int)randF(-3,3),(int)randF(-3,3));
            saltPepper(base,0.02);

            double maxV=*max_element(base.begin(),base.end());
            if(maxV>0.01) for(auto& v:base) v/=maxV;
            samples.push_back(base);
        }
        return samples;
    }
};

#endif