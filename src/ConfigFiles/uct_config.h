#ifndef UCT_CONFIG_H_
#define UCT_CONFIG_H_

// include C++ standard libraries
#include <iostream>
#include <string>

using namespace std;

class uct_config
{
public:
    string config_filepath;
    string DIR_PATH;
    string FILENAME;
    uint FIRST_IDX;
    uint DIGITS;
    string EXTENSION;
    uint SLICES;
    double RESOLUTION;
    uint VOXEL_DIVISION;



    // default constructors
    uct_config(){};
    uct_config(const string configFile);

    //copy constructors
    uct_config(const uct_config &otherConfig);

    // default destructor
    virtual ~uct_config()
    {
        // cout << "OMPLoopEnabler object destroyed succesfully" << endl;
    } 

    void readConfigFile(const string configFile);
    
    // -- Read methods
    void readDirPath(string s);
    void readFilename(string s);
    void readFirstIdx(string s);
    void readDigits(string s);
    void readExtension(string s);
    void readSlices(string s); 
    void readResolution(string s);
    void readVoxelDivision(string s);

    // -- Read methods
    string getConfigFilepath() {return this->config_filepath; }
    string getDirPath(){ return this->DIR_PATH;}
    string getFilename(){ return this->FILENAME;}
    uint getFirstIdx(){ return this->FIRST_IDX;}
    uint getDigits(){ return this->DIGITS;}
    string getExtension(){ return this->EXTENSION;}
    uint getSlices(){ return this->SLICES;} 
    double getResolution(){ return this->RESOLUTION;}
    uint getVoxelDivision(){ return this->VOXEL_DIVISION;}
};

#endif