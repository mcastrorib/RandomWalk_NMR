// include C++ standard libraries
#include <iostream>
#include <unistd.h>

// include CMake Configuration file
#include "RWNMR_Config.h"

// include interface header files
#include "rwnmrApp.h"

#include "myRNG.h"

using namespace std;

// Main program
int main(int argc, char *argv[])
{    	
	// rwnmrApp app(argc, argv, PROJECT_ROOT_DIR);
	// app.exec();


	cout << "RNG: " << myRNG::RNG_uint64() << endl;
    return 0;
}
