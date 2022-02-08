#ifndef LSADJUST_H
#define LSADJUST_H

#include <iostream>
#include <vector>

using namespace std;

class LeastSquareAdjust
{
public:
    vector<double> &X;
    vector<double> &Y;
    
    LeastSquareAdjust(vector<double> &_x, vector<double> &_y);
    virtual ~LeastSquareAdjust(){}

    void setX(vector<double> &_x);
    void setY(vector<double> &_y);
    void setThreshold(double _threshold);
    void setPoints(int _points);
    void setLimits();

    void solve();    

    double getMeanX(){ return this->meanX; }
    double getMeanY(){ return this->meanY; }
    double getA();
    double getB();
    double getMSE();
    double getSMSE();
    bool isSolved() { return this->solved; }

private:
    double meanX, meanY;
    double A, B, residual;
    bool solved;

    int begin, end;
    int points;
    double threshold;

    double computeMean(vector<double> &_vector);
    void computeB();
    void computeA();
    void setSolved(bool isSolved);
    void computeMeanSquaredResiduals(); 
    double evaluate(double point);
};

#endif
