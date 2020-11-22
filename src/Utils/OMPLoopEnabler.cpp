#include <iostream>
#include <omp.h>

#include "OMPLoopEnabler.h"

using namespace std;

OMPLoopEnabler::OMPLoopEnabler( const int thread_id, 
                                const int num_threads, 
                                const int num_loops)
{
    const int loops_per_thread = num_loops / num_threads;
    (*this).setStart(thread_id * loops_per_thread);
    if (thread_id == (num_threads - 1)) { (*this).setFinish(num_loops); }
    else { (*this).setFinish((*this).getStart() + loops_per_thread); }
}