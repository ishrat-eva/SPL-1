
#undef UNICODE
#undef _UNICODE

#include <windows.h>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include "Matrix.h"
#include "activationFunction.h"
#include "convolutionalLayer.h"
#include "poolingLayer.h"
#include "connectedLayer.h"
#include "MICRGenerator.h"
#include "simpleCNN14.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;


#define BTN_CLEAR    1
#define BTN_TRANSIT  2
#define BTN_AMOUNT   3
#define BTN_ONUS     4
#define BTN_DASH     5
SimpleCNN14* g_cnn    = nullptr;
bool g_drawing        = false;
bool g_modelLoaded    = false;

double g_canvas[28][28] = {};
int    g_predClass  = -1;
double g_confidence = 0.0;
double g_probs[14]  = {};

int g_W = 960;
int g_H = 620;

HWND hBtnClear   = NULL;
HWND hBtnTransit = NULL;
HWND hBtnAmount  = NULL;
HWND hBtnOnUs    = NULL;
HWND hBtnDash    = NULL;

const char* CLS[14] = {
    "0","1","2","3","4","5","6","7","8","9",
    "Transit","Amount","On-Us","Dash"
};

struct Layout {
    int cellSize;
    int cvX, cvY, cvPx;  
    int rX, rY, rW, rH;   
    int btnY;             
};

Layout getLayout() {
    Layout L;
    int topH   = 65;
    int midH   = g_H - topH - 20; 
    int halfW  = (g_W - 30) / 2;
    L.cellSize = min(halfW, midH) / 28;
    if(L.cellSize < 7) L.cellSize = 7;
    L.cvPx = L.cellSize * 28;
    L.cvX  = 14;
    L.cvY  = topH;
    L.rX   = L.cvX + L.cvPx + 16;
    L.rY   = topH;
    L.rW   = g_W - L.rX - 10;
    L.rH   = L.cvPx;
    L.btnY = 0; 
    return L;
}
void clearAll() {
    memset(g_canvas, 0, sizeof(g_canvas));
    g_predClass  = -1;
    g_confidence = 0.0;
    memset(g_probs, 0, sizeof(g_probs));
}

void paintAt(int mx, int my) {
    Layout L = getLayout();
    int cx = (mx - L.cvX) / L.cellSize;
    int cy = (my - L.cvY) / L.cellSize;
    if(cx<0||cx>=28||cy<0||cy>=28) return;
    g_canvas[cy][cx] = 1.0;
    if(cx>0)  g_canvas[cy][cx-1] = min(1.0, g_canvas[cy][cx-1]+0.35);
    if(cx<27) g_canvas[cy][cx+1] = min(1.0, g_canvas[cy][cx+1]+0.35);
    if(cy>0)  g_canvas[cy-1][cx] = min(1.0, g_canvas[cy-1][cx]+0.35);
    if(cy<27) g_canvas[cy+1][cx] = min(1.0, g_canvas[cy+1][cx]+0.35);
}

void loadSymbol(int symbolType) {
    vector<double> base;
    if(symbolType == 0)      base = MICRGenerator::makeTransit(2);
    else if(symbolType == 1) base = MICRGenerator::makeAmount(2);
    else if(symbolType == 2) base = MICRGenerator::makeOnUs(2);
    else                     base = MICRGenerator::makeDash(2);

for(int r=0;r<28;r++)
    for(int c=0;c<28;c++)
        g_canvas[r][c] = base[r*28+c];
}


void runPredict() {
    if(!g_modelLoaded || !g_cnn) return;
    vector<double> flat(784);
    

    for(int r=0; r<28; r++)
        for(int c=0; c<28; c++)
            flat[r*28+c] = (g_canvas[r][c] > 0.85) ? 1.0 : 0.0;
            
    auto probs = g_cnn->predict(flat);
    g_predClass  = g_cnn->getPredictedClass(probs);
    g_confidence = g_cnn->getConfidence(probs)*100.0;
    for(int i=0; i<14; i++) g_probs[i] = probs[i];
}

void render(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);

    HDC mem = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, g_W, g_H);
    SelectObject(mem, bmp);
    HBRUSH bgBr = CreateSolidBrush(RGB(235,238,250));
    RECT all = {0,0,g_W,g_H};
    FillRect(mem, &all, bgBr);
    DeleteObject(bgBr);
    SetBkMode(mem, TRANSPARENT);


    HFONT fTitle = CreateFontA(22,0,0,0,FW_BOLD,0,0,0,ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,"Arial");
    HFONT fSub   = CreateFontA(13,0,0,0,FW_NORMAL,0,0,0,ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,"Arial");
    HFONT fNorm  = CreateFontA(14,0,0,0,FW_NORMAL,0,0,0,ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,"Arial");
    HFONT fBold  = CreateFontA(14,0,0,0,FW_BOLD,0,0,0,ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,"Arial");
    HFONT fBig   = CreateFontA(52,0,0,0,FW_BOLD,0,0,0,ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,"Arial");
    HFONT fMicr  = CreateFontA(13,0,0,0,FW_NORMAL,0,0,0,ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,"Arial");

    Layout L = getLayout();

    
    SelectObject(mem, fTitle);
    SetTextColor(mem, RGB(18,32,95));
    TextOutA(mem, 14, 8, "Handwritten Digit Recognition", 29);

    SelectObject(mem, fSub);
    if(!g_modelLoaded) {
        SetTextColor(mem, RGB(200,40,40));
        TextOutA(mem, 14, 38, "ERROR: weights_micr.bin not found! Run train_micr.exe first.", 60);
    } else {
        SetTextColor(mem, RGB(22,128,42));
        TextOutA(mem, 14, 38, "Model ready  |  Draw digit (left drag)  |  Right-click = clear  |  Use buttons for MICR symbols", 94);
    }

 
    HBRUSH wBr = CreateSolidBrush(RGB(255,255,255));
    RECT cr = {L.cvX, L.cvY, L.cvX+L.cvPx, L.cvY+L.cvPx};
    FillRect(mem, &cr, wBr);
    DeleteObject(wBr);

    for(int r=0;r<28;r++) for(int c=0;c<28;c++) {
        double v=g_canvas[r][c];
        if(v<0.01) continue;
    
        int gray = (v >= 0.5) ? 0 : (int)(255*(1.0-v*2));
        HBRUSH pb=CreateSolidBrush(RGB(gray,gray,gray));
        RECT pr={L.cvX+c*L.cellSize, L.cvY+r*L.cellSize,
                 L.cvX+(c+1)*L.cellSize, L.cvY+(r+1)*L.cellSize};
        FillRect(mem,&pr,pb);
        DeleteObject(pb);
    }

    HPEN gp=CreatePen(PS_SOLID,1,RGB(208,213,235));
    SelectObject(mem,gp);
    for(int i=0;i<=28;i++){
        MoveToEx(mem,L.cvX+i*L.cellSize,L.cvY,NULL);
        LineTo(mem,L.cvX+i*L.cellSize,L.cvY+L.cvPx);
        MoveToEx(mem,L.cvX,L.cvY+i*L.cellSize,NULL);
        LineTo(mem,L.cvX+L.cvPx,L.cvY+i*L.cellSize);
    }
    DeleteObject(gp);

    HPEN bp=CreatePen(PS_SOLID,2,RGB(65,82,180));
    SelectObject(mem,bp);
    SelectObject(mem,GetStockObject(NULL_BRUSH));
    Rectangle(mem,L.cvX-2,L.cvY-2,L.cvX+L.cvPx+2,L.cvY+L.cvPx+2);
    DeleteObject(bp);

    SelectObject(mem, fMicr);
    SetTextColor(mem, RGB(55,60,110));
    TextOutA(mem, L.cvX, L.btnY-16, "MICR Symbols:", 13);
    HBRUSH resBr=CreateSolidBrush(RGB(255,255,255));
    RECT rr={L.rX,L.rY,L.rX+L.rW,L.rY+L.rH};
    FillRect(mem,&rr,resBr);
    DeleteObject(resBr);

    HPEN rp=CreatePen(PS_SOLID,2,RGB(65,82,180));
    SelectObject(mem,rp);
    Rectangle(mem,L.rX,L.rY,L.rX+L.rW,L.rY+L.rH);
    DeleteObject(rp);

    SelectObject(mem,fBold);
    SetTextColor(mem,RGB(18,32,95));
    TextOutA(mem,L.rX+12,L.rY+10,"Prediction Result",17);

    if(g_predClass>=0) {
        SelectObject(mem,fBig);
        SetTextColor(mem,RGB(12,82,200));
        TextOutA(mem,L.rX+14,L.rY+34,CLS[g_predClass],strlen(CLS[g_predClass]));

        SelectObject(mem,fBold);
        SetTextColor(mem,RGB(22,130,42));
        ostringstream cs;
        cs<<fixed<<setprecision(1)<<g_confidence<<"% confident";
        string css=cs.str();
        TextOutA(mem,L.rX+14,L.rY+96,css.c_str(),css.size());

        HPEN dp=CreatePen(PS_SOLID,1,RGB(190,200,230));
        SelectObject(mem,dp);
        MoveToEx(mem,L.rX+10,L.rY+116,NULL); LineTo(mem,L.rX+L.rW-10,L.rY+116);
        DeleteObject(dp);

        SelectObject(mem,fNorm);
        SetTextColor(mem,RGB(52,58,78));
        TextOutA(mem,L.rX+12,L.rY+122,"All probabilities:",18);

        int barMaxW = L.rW - 128;
        int barY    = L.rY + 142;
        int barH    = max(10,(L.rH-155)/14);

        for(int i=0;i<14;i++){
            bool top=(i==g_predClass);
            SelectObject(mem,top?fBold:fNorm);
            SetTextColor(mem,top?RGB(12,82,200):RGB(62,65,85));
            TextOutA(mem,L.rX+12,barY,CLS[i],strlen(CLS[i]));

            HBRUSH bbg=CreateSolidBrush(RGB(218,222,242));
            RECT bgr={L.rX+78,barY+1,L.rX+78+barMaxW,barY+barH};
            FillRect(mem,&bgr,bbg); DeleteObject(bbg);

            int bw=(int)(g_probs[i]*barMaxW);
            if(bw>0){
                HBRUSH bf=CreateSolidBrush(top?RGB(32,102,220):RGB(132,162,228));
                RECT bfr={L.rX+78,barY+1,L.rX+78+bw,barY+barH};
                FillRect(mem,&bfr,bf); DeleteObject(bf);
            }

            ostringstream pct;
            pct<<fixed<<setprecision(1)<<(g_probs[i]*100.0)<<"%";
            string ps2=pct.str();
            SelectObject(mem,fNorm);
            SetTextColor(mem,top?RGB(12,82,200):RGB(82,88,108));
            TextOutA(mem,L.rX+82+barMaxW,barY,ps2.c_str(),ps2.size());

            barY+=barH+2;
        }
    } else {
        SelectObject(mem,fNorm);
        SetTextColor(mem,RGB(150,155,178));
        TextOutA(mem,L.rX+14,L.rY+58,"Draw a digit above,",19);
        TextOutA(mem,L.rX+14,L.rY+78,"or click a MICR",15);
        TextOutA(mem,L.rX+14,L.rY+98,"symbol button below.",20);
    }

    DeleteObject(fTitle); DeleteObject(fSub); DeleteObject(fNorm);
    DeleteObject(fBold);  DeleteObject(fBig); DeleteObject(fMicr);

    BitBlt(hdc,0,0,g_W,g_H,mem,0,0,SRCCOPY);
    DeleteObject(bmp); DeleteDC(mem);
    EndPaint(hwnd,&ps);
}

void repositionButtons() {
    if(!hBtnClear) return;
    Layout L = getLayout();
    int bY  = L.btnY;
    int bH  = 32;
    int bW  = 90;
    int gap = 10;
    int x   = L.cvX;

    SetWindowPos(hBtnTransit, NULL, x,                bY, bW, bH, SWP_NOZORDER);
    SetWindowPos(hBtnAmount,  NULL, x+(bW+gap),       bY, bW, bH, SWP_NOZORDER);
    SetWindowPos(hBtnOnUs,    NULL, x+(bW+gap)*2,     bY, bW, bH, SWP_NOZORDER);
    SetWindowPos(hBtnDash,    NULL, x+(bW+gap)*3,     bY, bW, bH, SWP_NOZORDER);
    SetWindowPos(hBtnClear,   NULL, x+(bW+gap)*4+20,  bY, bW, bH, SWP_NOZORDER);
}



LRESULT CALLBACK WndProc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    switch(msg){

    case WM_CREATE: {
        break;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_SIZE:
        g_W = LOWORD(lp);
        g_H = HIWORD(lp);
        InvalidateRect(hw,NULL,FALSE);
        break;

    case WM_COMMAND:
        break;

    case WM_LBUTTONDOWN:
        g_drawing=true;
        paintAt(LOWORD(lp),HIWORD(lp));
        InvalidateRect(hw,NULL,FALSE);
        break;

    case WM_LBUTTONUP:
        g_drawing=false;
        runPredict();
        InvalidateRect(hw,NULL,FALSE);
        break;

    case WM_MOUSEMOVE:
        if(g_drawing){
            paintAt(LOWORD(lp),HIWORD(lp));
            runPredict();
            InvalidateRect(hw,NULL,FALSE);
        }
        break;

    case WM_RBUTTONDOWN:
        clearAll();
        InvalidateRect(hw,NULL,FALSE);
        break;

    case WM_GETMINMAXINFO: {
        MINMAXINFO* mm=(MINMAXINFO*)lp;
        mm->ptMinTrackSize.x=640;
        mm->ptMinTrackSize.y=480;
        break;
    }
    case WM_PAINT:
        render(hw);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProc(hw,msg,wp,lp);
}
    int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nShow){
    g_cnn = new SimpleCNN14();

    
    g_modelLoaded = g_cnn->loadWeights("weights_micr_real.bin");
    
    
    if(!g_modelLoaded) {
        g_modelLoaded = g_cnn->loadWeights("weights_micr.bin");
    }

    WNDCLASSA wc  = {};
    wc.lpfnWndProc  = WndProc;
    wc.hInstance    = hInst;
    wc.hCursor      = LoadCursor(NULL,IDC_CROSS);
    wc.hIcon        = LoadIcon(NULL,IDI_APPLICATION);
    wc.hbrBackground= NULL;
    wc.lpszClassName= "MICRDraw";
    RegisterClassA(&wc);
    RECT wr={0,0,960,620};
    AdjustWindowRect(&wr,WS_OVERLAPPEDWINDOW,FALSE);
    HWND hw=CreateWindowA(
        "MICRDraw",
        "Handwritten Digit Recognition - SPL Project",
        WS_OVERLAPPEDWINDOW,
        80, 60,
        wr.right-wr.left, wr.bottom-wr.top,
        NULL,NULL,hInst,NULL
    );

    ShowWindow(hw,nShow);
    UpdateWindow(hw);
    repositionButtons();
    MSG msg;
    while(GetMessage(&msg,NULL,0,0)){
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    delete g_cnn;
    return 0;
}



