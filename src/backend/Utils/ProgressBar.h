#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H

#include <iostream>
#include <string>

using namespace std;

class ProgressBar 
{
    public:
        string firstPartOfpBar = "["; //Change these at will (that is why I made them public)
        string lastPartOfpBar = "]";
        string pBarFiller = "|";
        string pBarUpdater = "/-\\|";

        ProgressBar(double _total) : neededProgress(_total)
        {
            this->pBarLength = 50; //I would recommend NOT changing this
            this->currUpdateVal = 0; //Do not change
            this->currentProgress = 0.0; //Do not change
            (*this).update(0.0);
        }

        void reset(double _newTotal)
        {
            this->neededProgress = _newTotal;
            this->currUpdateVal = 0; //Do not change
            this->currentProgress = 0.0; //Do not change
            this->amountOfFiller = 0;
        }

        void update(double newProgress) 
        {
            currentProgress += newProgress;
            amountOfFiller = (int)((this->currentProgress / this->neededProgress) * (double) this->pBarLength);
        }
        
        void print() 
        {
            this->currUpdateVal %= this->pBarUpdater.length();            
            cout << "\r" //Bring cursor to start of line
            << this->firstPartOfpBar; //Print out first part of pBar
            
            //Print out current progress
            for (int a = 0; a < this->amountOfFiller; a++) 
            { 
                cout << this->pBarFiller;
            }
            cout << this->pBarUpdater[currUpdateVal];

            for (int b = 0; b < this->pBarLength - this->amountOfFiller; b++) { //Print out spaces
                cout << " ";
            }

            cout << this->lastPartOfpBar //Print out last part of progress bar
            << " (" << (int)(100*(this->currentProgress/this->neededProgress)) << "%)" //This just prints out the percent
            << flush;
            this->currUpdateVal += 1;
        }

        

    private:
        int amountOfFiller;
        int pBarLength; //I would recommend NOT changing this
        int currUpdateVal; //Do not change
        double currentProgress; //Do not change
        double neededProgress; //I would recommend NOT changing this
};

#endif