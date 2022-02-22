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
    bool intercept;
    bool verbose;
    
    LeastSquareAdjust(vector<double> &_x, vector<double> &_y, bool _intercept = true);
    virtual ~LeastSquareAdjust(){}

    void setX(vector<double> &_x);
    void setY(vector<double> &_y);
    void setThreshold(double _threshold);
    void setPoints(int _points);
    void setLimits();
    void setVerbose(bool _verbose) { this->verbose = _verbose; }
    bool isVerbose() { return this->verbose; }

    void solve();    

    double getMeanX(){ return this->meanX; }
    double getMeanY(){ return this->meanY; }
    double getA();
    double getB();
    double getMSE();
    double getSMSE();
    bool isSolved() { return this->solved; }
    bool hasIntercept() { return this->intercept; }

private:
    double meanX, meanY;
    double A, B, residual;
    bool solved;

    int begin, end;
    int points;
    double threshold;

    double computeMean(vector<double> &_vector);
    void computeB();
    void computeBWithIntercept();
    void computeBWithoutIntercept();
    void computeA();
    void computeAWithIntercept();
    void computeAWithoutIntercept();
    void setSolved(bool isSolved);
    void computeMeanSquaredResiduals(); 
    double evaluate(double point);
};

#endif
