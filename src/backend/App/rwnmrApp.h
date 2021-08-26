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
    const string project_root;
    string config_root;
    ArgsParser args;
    NMR_Simulation *NMR;

    // default constructors
    rwnmrApp(){};
    rwnmrApp(int argc, char *argv[], string _root);
    rwnmrApp(ArgsParser _args, string _root);

    //copy constructors
    rwnmrApp(const rwnmrApp &_otherApp) : project_root(_otherApp.project_root)
    {
        this->config_root = _otherApp.config_root;
        this->args = _otherApp.args;
        this->NMR = _otherApp.NMR;
    }

    // default destructor
    virtual ~rwnmrApp()
    {
        delete NMR;
        NMR = NULL;
    }

    void buildEssentials();
    void exec();    

    void CPMG(uint command_idx);
    void PFGSE(uint command_idx);
    void GA(uint command_idx);

    string getProjectRoot() { return this->project_root; }
    string getConfigRoot() { return this->config_root; }
    NMR_Simulation& getNMR() { return (*this->NMR); }
    ArgsParser& getArgs() { return this->args; }
    string getArgsPath(uint idx) { return this->args.getPath(idx); }

    void setConfigRoot(string _croot) { this->config_root = _croot; }
};

#endif