#ifndef OMP_LOOP_ENABLER_H_
#define OMP_LOOP_ENABLER_H_

// include C++ standard libraries
#include <iostream>
#include <string>

using namespace std;

class OMPLoopEnabler
{
private:
    // pore position
    double start;
    double finish;

public:
    
    // Pore methods:
    // default constructors
    OMPLoopEnabler(const int tid, const int num_threads, const int num_loops);

    //copy constructors
    OMPLoopEnabler(const OMPLoopEnabler &_otherOMPLE);

    // default destructor
    virtual ~OMPLoopEnabler()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    }

    void setStart(int _start) { this->start = _start; }
    void setFinish(int _finish) { this->finish = _finish; }
    int getStart(){ return this->start; }
    int getFinish(){ return this->finish;}    
};

#endif