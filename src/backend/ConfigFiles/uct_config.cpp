// include C++ standard libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>

#include "uct_config.h"

using namespace std;

// default constructors
uct_config::uct_config(const string configFile) : config_filepath(configFile)
{
	this->IMG_FILES_LIST = "Empty";
	vector<string> IMG_FILES();

	string default_dirpath = CONFIG_ROOT;
	string default_filename = UCT_CONFIG_DEFAULT;
	(*this).readConfigFile(default_dirpath + default_filename);
	if(configFile != (default_dirpath + default_filename)) (*this).readConfigFile(configFile);
	if((*this).getImgFilesList() == "Empty") (*this).createImgFileList();	
	(*this).readImgFiles();
}

//copy constructors
uct_config::uct_config(const uct_config &otherConfig)
{
	this->config_filepath = otherConfig.config_filepath;
    this->DIR_PATH = otherConfig.DIR_PATH;
    this->FILENAME = otherConfig.FILENAME;
    this->FIRST_IDX = otherConfig.FIRST_IDX;
    this->DIGITS = otherConfig.DIGITS;
    this->EXTENSION = otherConfig.EXTENSION;
    this->SLICES = otherConfig.SLICES;
    this->RESOLUTION = otherConfig.RESOLUTION;
    this->VOXEL_DIVISION = otherConfig.VOXEL_DIVISION;
	this->IMG_FILES_LIST = otherConfig.IMG_FILES_LIST;
	this->IMG_FILES = otherConfig.IMG_FILES;
}

// read config file
void uct_config::readConfigFile(const string configFile)
{
	cout << "reading uct configs from file...";

    ifstream fileObject;
    fileObject.open(configFile, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

    string line;
    while(fileObject)
    {
    	getline(fileObject, line);
    	// cout << line << endl;

    	string s = line;
    	string delimiter = ": ";
		size_t pos = 0;
		string token, content;
    	while ((pos = s.find(delimiter)) != std::string::npos) 
    	{
			token = s.substr(0, pos);
			content = s.substr(pos + delimiter.length(), s.length());
			s.erase(0, pos + delimiter.length());

			if(token == "DIR_PATH")	(*this).readDirPath(content);
			else if(token == "FILENAME") (*this).readFilename(content);
			else if(token == "FIRST_IDX") (*this).readFirstIdx(content);
			else if(token == "DIGITS") (*this).readDigits(content);
			else if(token == "EXTENSION") (*this).readExtension(content);
			else if(token == "SLICES") (*this).readSlices(content);
			else if(token == "RESOLUTION") (*this).readResolution(content);
			else if(token == "VOXEL_DIVISION") (*this).readVoxelDivision(content);
			else if(token == "IMG_FILES_LIST") (*this).readImgFilesList(content);
			
		}
    } 

    cout << "Ok" << endl;
    fileObject.close();
}

void uct_config::readDirPath(string s)
{
	this->DIR_PATH = s;
}

void uct_config::readFilename(string s)
{
	this->FILENAME = s;
}

void uct_config::readFirstIdx(string s)
{
	this->FIRST_IDX = std::stoi(s);
}

void uct_config::readDigits(string s)
{
	this->DIGITS = std::stoi(s);
}

void uct_config::readExtension(string s)
{
	this->EXTENSION = s;
}

void uct_config::readSlices(string s)
{
	this->SLICES = std::stoi(s);
}

void uct_config::readResolution(string s)
{
	this->RESOLUTION = std::stod(s);
}

void uct_config::readVoxelDivision(string s)
{
	this->VOXEL_DIVISION = std::stoi(s);
}

void uct_config::readImgFilesList(string s)
{
	this->IMG_FILES_LIST = s;
}

void uct_config::readImgFiles()
{
	const string filepath = (*this).getImgFilesList();
	cout << "reading image list from file " << filepath << "...";

    ifstream fileObject;
    fileObject.open(filepath, ios::in);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

	// reserve memory for image file list
	if(this->IMG_FILES.size() > 0) this->IMG_FILES.clear();
	this->IMG_FILES.reserve((*this).getSlices());

    string line;
	uint slice = 0;
    while(fileObject)
    {
    	getline(fileObject, line);
		// cout << line << endl;
    	if(slice < (*this).getSlices()) this->IMG_FILES.push_back(line);
		slice++;
    } 
	cout << "img list size: " << this->IMG_FILES.size() << endl;
    cout << "Ok" << endl;
    fileObject.close();
}

void uct_config::createImgFileList()
{
    cout << "creating image file list...";

	string dirpath = CONFIG_ROOT;
	string filepath = dirpath + "/imgs/ImagesList.txt";

    ofstream fileObject;
    fileObject.open(filepath, ios::out);
    if (fileObject.fail())
    {
        cout << "Could not open file from disc." << endl;
        exit(1);
    }

	// constant strings
    string currentDirectory = (*this).getDirPath();
    string currentFileName = (*this).getFilename();
    string currentExtension = (*this).getExtension();
	uint firstImage = (*this).getFirstIdx();
    uint digits = (*this).getDigits();
	uint slices = (*this).getSlices();

    // variable strings
    string currentFileID;
    string currentImagePath;    

    for (uint slice = 0; slice < slices; slice++)
    {
        // identifying next image to be read
        currentFileID = (*this).convertFileIDToString(firstImage + slice, digits);
        currentImagePath = currentDirectory + currentFileName + currentFileID + currentExtension;

        fileObject << currentImagePath << endl;
    }	  

	(*this).readImgFilesList(filepath);    
    cout << "Ok. (" << time << " seconds)." << endl;
}


