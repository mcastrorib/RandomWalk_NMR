#ifndef RWNMRAPP_H_
#define RWNMRAPP_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>
#include "../Utils/ArgsParser.h"
#include "../ConfigFiles/configFiles_defs.h"
#include "../NMR_Simulation/NMR_Simulation.h"


using namespace std;

class rwnmrApp
{
public:   
    const string config_root;
    ArgsParser args;
    NMR_Simulation *NMR;

    // default constructors
    rwnmrApp(){};
    rwnmrApp(int argc, char *argv[]);
    rwnmrApp(ArgsParser _args);

    //copy constructors
    rwnmrApp(const rwnmrApp &_otherApp);

    // default destructor
    virtual ~rwnmrApp()
    {
        delete NMR;
        NMR = NULL;
        cout << "Application ended." << endl;
    }

    void buildEssentials();
    void exec();    

    void CPMG(uint command_idx);
    void PFGSE(uint command_idx);
    void GA(uint command_idx);

    string getConfigRoot() { return this->config_root; }
    NMR_Simulation& getNMR() { return (*this->NMR); }
    ArgsParser& getArgs() { return this->args; }
    string getArgsPath(uint idx) { return this->args.getPath(idx); }
};

#endif