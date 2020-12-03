#ifndef ARGS_PARSER_H_
#define ARGS_PARSER_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class ArgsParser
{
public:    
    vector<string> commands;
    vector<string> paths;
     
    // default constructors
    ArgsParser(){};
    ArgsParser(int argc, char *argv[]);

    //copy constructors
    ArgsParser(const ArgsParser &_otherParser);

    // default destructor
    virtual ~ArgsParser()
    {
        // cout << "ArgParser object destroyed succesfully" << endl;
    }

    string getCommand(uint idx) {return this->commands[idx];}
    string getPath(uint idx) {return this->paths[idx];}

      
};

#endif