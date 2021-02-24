// include C++ standard libraries
#include <iostream>
#include <unistd.h>

// include CMake Configuration file
#include "RWNMR_Config.h"

// include interface header files
#include "rwnmrApp.h"

using namespace std;

// Main program
int main(int argc, char *argv[])
{    	
	rwnmrApp app(argc, argv);
	app.exec();

	// int pos = -2001;
	// int shift = 2;
	// uint cols = 900;
	// int imgPos = (pos / shift) % cols;
	// if(imgPos < 0) imgPos += cols;
	// cout << "pos = " << pos << endl;
	// cout << "shift = " << shift << endl;
	// cout << "cols = " << cols << endl;
	// cout << "imgPos = " << imgPos << endl;

    return 0;
}
