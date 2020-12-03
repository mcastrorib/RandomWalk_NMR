// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <cstdlib>

#include "ImagePath.h"

using namespace std;

ImagePath::ImagePath(string _path, 
                     string _name, 
                     uint _fileID, 
                     uint _digits,
                     uint _images, 
                     string _extension) : path(_path),
                                          filename(_name),
                                          fileID(_fileID),
                                          digits(_digits),
                                          images(_images),
                                          extension(_extension) 
{
    (*this).updateCompletePath();    
}

void ImagePath::updateCompletePath()
{
    (*this).updateNumberOfDigits();
    this->completePath = this->path + 
                         this->filename + 
                         (*this).convertFileIDToString(this->fileID, this->digits) + 
                         this->extension;
}

void ImagePath::updateNumberOfDigits()
{
    int finalID = this->fileID + this->images;
    int result = finalID / 10;
    int count = 1;

    while (result > 0)
    {
        count++;
        result = result / 10;
    }

    this->digits = count;
}