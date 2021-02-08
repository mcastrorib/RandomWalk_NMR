#ifndef APP_H_
#define APP_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>
#include "../Utils/ArgsParser.h"
#include "../ConfigFiles/configFiles_defs.h"
#include "../NMR_Simulation/NMR_Simulation.h"


using namespace std;

class App
{
public:   
    const string config_root;
    ArgsParser args;
    NMR_Simulation *NMR;

    // default constructors
    App(){};
    App(ArgsParser _args);

    //copy constructors
    App(const App &_otherApp);

    // default destructor
    virtual ~App()
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