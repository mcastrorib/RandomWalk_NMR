// include C++ standard libraries
#include <iostream>
#include <unistd.h>

// include CMake Configuration file
#include "RWNMR_Config.h"

// include interface header files
#include "rwnmrApp.h"
#include "ArgsParser.h"

using namespace std;

// Main program
int main(int argc, char *argv[])
{    
	
	rwnmrApp app(argc, argv);
	app.exec();

    return 0;
}
