#ifndef MY_ALLOCATOR_H_
#define MY_ALLOCATOR_H_

// include C++ standard libraries
#include <iostream>
#include <string>

using namespace std;

class myAllocator
{
public:    
    // Pore methods:
    // default constructors
    myAllocator(){};

    //copy constructors
    myAllocator(const myAllocator &_otherAllocator);

    // default destructor
    virtual ~myAllocator()
    {
        // cout << "myAllocator object destroyed succesfully" << endl;
    }

    int* getIntArray(uint size);
    uint* getUIntArray(uint size);
    double* getDoubleArray(uint size);
    uint64_t* getUInt64Array(uint size);
      
};

#endif