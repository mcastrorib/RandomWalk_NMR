#ifndef APP_H_
#define APP_H_

// include C++ standard libraries
#include <iostream>
#include <string>
#include <vector>
#include "../Utils/ArgsParser.h"
#include "../NMR_Simulation/NMR_Simulation.h"

using namespace std;

#define CONFIG_ROOT "/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/config/"
#define RWNMR_CONFIG_DEFAULT "rwnmr.config"
#define UCT_CONFIG_DEFAULT "uct.config"
#define CPMG_CONFIG_DEFAULT "cpmg.config"
#define PFGSE_CONFIG_DEFAULT "pfgse.config"
#define GA_CONFIG_DEFAULT "ga.config"


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