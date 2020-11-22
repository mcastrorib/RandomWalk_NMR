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
    void setLimits();

    void solve();    

    double getMeanX(){ return this->meanX; }
    double getMeanY(){ return this->meanY; }
    double getA(){ return this->A; }
    double getB(){ return this->B; }

private:
    double meanX, meanY;
    double A, B;
    bool solved;

    int begin, end;
    double threshold;

    double computeMean(vector<double> &_vector);
    void computeB();
    void computeA();
    void setAsSolved();
    void setAsUnsolved();
};

#endif
