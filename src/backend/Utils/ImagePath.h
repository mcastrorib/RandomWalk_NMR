#ifndef IMAGE_PATH_H
#define IMAGE_PATH_H

#include <sstream>
#include <iomanip>
#include <string>

using namespace std;

class ImagePath
{
public:
    string path;
    string filename;
    uint fileID;
    uint digits;
    uint images;
    string extension;
    string completePath;

    ImagePath(){}
    ImagePath(string _path, string _name, uint _fileID, uint _digits, uint _images, string _extension);

    virtual ~ImagePath(){}

    void setPath(string newInput){this->path = newInput;}
    void setFileName(string newInput){this->filename = newInput;}
    void setFileID(string newInput){this->fileID = std::stoi(newInput);}
    void setImages(string newInput){this->images = std::stoi(newInput);}
    void setImageExtension(string newInput){this->extension = newInput;}
    void updateCompletePath();
    void updateNumberOfDigits();

    inline string convertFileIDToString(uint id, uint digits)
    {
        stringstream result;
        result << std::setfill('0') << std::setw(digits) << id;
        return result.str();
    }
};

#endif