// include C++ standard libraries
#include <iostream>
#include <unistd.h>

// include CMake Configuration file
#include "NMR_RWConfig.h"

// include interface header files
#include "App.h"
#include "ArgsParser.h"

using namespace std;

// Main Program
int main(int argc, char *argv[])
{    
     ArgsParser args(argc, argv);
     App NMR_app(args);
     NMR_app.exec();

     return 0;
}
