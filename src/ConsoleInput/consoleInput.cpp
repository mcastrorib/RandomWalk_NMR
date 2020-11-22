// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <cstdlib>

#include "consoleInput.h"

using namespace std;

// default values
// string ConsoleInput::simulationName = DEFAULT_NAME;
// uint ConsoleInput::numberOfImages = DEFAULT_NUMBEROFIMAGES;
// uint ConsoleInput::steps = DEFAULT_STEPS;
// double ConsoleInput::occupancy = DEFAULT_OCCUPANCY;
// uint64_t ConsoleInput::seed = DEFAULT_SEED;
// bool ConsoleInput::use_GPU = DEFAULT_GPU_USAGE;
// string ConsoleInput::T2path = DEFAULT_T2_PATH;

ConsoleInput::ConsoleInput() : simulationName(DEFAULT_NAME),
                               numberOfImages(DEFAULT_NUMBEROFIMAGES),
                               steps(DEFAULT_STEPS),
                               occupancy(DEFAULT_OCCUPANCY),
                               seed(DEFAULT_SEED),
                               use_GPU(DEFAULT_GPU_USAGE),
                               T2path(DEFAULT_T2_PATH)
{
    quit = false;
    imagePath.path = DEFAULT_IMAGE_PATH;
    imagePath.filename = DEFAULT_IMAGE_FILE;
    imagePath.fileID = DEFAULT_IMAGE_ID;
    imagePath.digits = DEFAULT_IMAGE_DIGITS;
    imagePath.extension = DEFAULT_IMAGE_EXTENSION;
    (*this).updateCompletePath();    
}

void ConsoleInput::updateCompletePath()
{
    (*this).updateNumberOfDigits();
    imagePath.completePath = imagePath.path + 
                             imagePath.filename + 
                             convertFileIDToString(imagePath.fileID, imagePath.digits) + 
                             imagePath.extension;
}

void ConsoleInput::updateNumberOfDigits()
{
    int finalID = this->imagePath.fileID + this->numberOfImages;
    int result = finalID / 10;
    int count = 1;

    while (result > 0)
    {
        count++;
        result = result / 10;
    }

    this->imagePath.digits = count;
}

void ConsoleInput::resetConsoleInput()
{
    this->numberOfImages = DEFAULT_NUMBEROFIMAGES;
    this->steps = DEFAULT_STEPS;
    this->occupancy = DEFAULT_OCCUPANCY;
    this->seed = DEFAULT_SEED;
    this->use_GPU = DEFAULT_GPU_USAGE;
    this->T2path = DEFAULT_T2_PATH;
    this->imagePath.path = DEFAULT_IMAGE_PATH;
    this->imagePath.filename = DEFAULT_IMAGE_FILE;
    this->imagePath.fileID = DEFAULT_IMAGE_ID;
    this->imagePath.digits = DEFAULT_IMAGE_DIGITS;
    this->imagePath.extension = DEFAULT_IMAGE_EXTENSION;
    (*this).updateCompletePath();    
}

bool ConsoleInput::quitProgram()
{
    string key;
    bool validAnswer = false;

    while (validAnswer != true)
    {
        cout << endl
             << "Quit program or restart another NMR simulation? [(Q)uit/(R)estart]" << endl;

        cin >> key;

        if (key == "YES" or key == "yes" or
            key == "Y" or key == "y" or
            key == "QUIT" or key == "quit" or
            key == "Q" or key == "q")
        {
            return (true);
        }
        if (key == "NO" or key == "no" or
            key == "N" or key == "n" or
            key == "RESTART" or key == "restart" or
            key == "R" or key == "r")
        {
            return (false);
        }

        cout << "Sorry, invalid answer.";
    }
}

bool ConsoleInput::menuContinue()
{
    string key;
    bool validAnswer = false;

    while (validAnswer != true)
    {
        cout << endl
             << "Continue to next step? [(Y)es/(N)o]" << endl;

        cin >> key;

        if (key == "YES" or key == "yes" or
            key == "Y" or key == "y")
        {
            return (true);
        }
        if (key == "NO" or key == "no" or
            key == "N" or key == "n")
        {
            return (false);
        }

        cout << "Sorry, invalid answer.";
    }
}

bool ConsoleInput::menuSetImage()
{
    string key = "0";
    menuChoice choice = none;

    while (choice != choice_continue)
    {
        // print menu on screen
        // clear console screen
        system("clear");
        cout << "------------------------------------------------------" << endl;
        cout << "RANDOM WALK NMR SIMULATION PROGRAM" << endl;
        cout << "Set simulation parameters:" << endl
             << endl;
        cout << "0 - NMR simulation name (NAME)" << endl;
        cout << "1 - rock image file path (IMAGEPATH)" << endl;
        cout << "2 - number of images (IMAGES)" << endl
             << endl;
        cout << "Digit 'GO' or 'START' to proceed the simulation" << endl;
        cout << "Digit 'RESET' or 'DEFAULT' to reset default options." << endl;
        cout << "Digit 'EXIT' or 'QUIT' to abort." << endl;
        cout << "------------------------------------------------------" << endl;

        // translate string to enum type "menu_choice"
        cin >> key;
        cout << endl;

        choice = getMenuChoice(key);

        // branching input menu choice options
        if (choice == choice_name or
            choice == choice_imagepath or
            choice == choice_numberOfImages or
            choice == choice_reset or
            choice == none)
        {
            menuSwitch(choice);
        }

        // return false if user choose to exit, else it can return to menu or proceed simulation
        if (choice == choice_exit)
        {
            this->quit = true;
            return (false);
        }
        else if (choice != choice_continue)
        {
            choice = menuReturn(choice);
        }
    }
    // clear console screen
    system("clear");
    return (true);
}

bool ConsoleInput::menuSetWalkers()
{
    string key = "0";
    menuChoice choice = none;

    while (choice != choice_continue)
    {
        // print menu on screen
        // clear console screen
        system("clear");
        cout << "------------------------------------------------------" << endl;
        cout << "RANDOM WALK NMR SIMULATION PROGRAM" << endl;
        cout << "Set simulation parameters:" << endl
             << endl;
        cout << "0 - number of steps (STEPS)" << endl;
        cout << "1 - walker occupancy (OCCUPANCY)" << endl;
        cout << "2 - initial seed (SEED)" << endl;
        cout << "3 - gpu usage (GPU)" << endl
             << endl;
        cout << "Digit 'GO' or 'START' to proceed the simulation" << endl;
        cout << "Digit 'RESET' or 'DEFAULT' to reset default options." << endl;
        cout << "Digit 'EXIT' or 'QUIT' to abort." << endl;
        cout << "------------------------------------------------------" << endl;

        // translate string to enum type "menu_choice"
        cin >> key;
        cout << endl;

        // hardcoded translation
        if (key == "0")
        {
            key = "3";
        }
        else if (key == "1")
        {
            key = "4";
        }
        else if (key == "2")
        {
            key = "5";
        }
        else if (key == "3")
        {
            key = "6";
        }

        choice = getMenuChoice(key);

        // branching input menu choice options
        if (choice == choice_steps or
            choice == choice_occupancy or
            choice == choice_seed or
            choice == choice_gpu or
            choice == none)
        {
            menuSwitch(choice);
        }

        // return false if user choose to exit, else it can return to menu or proceed simulation
        if (choice == choice_exit)
        {
            this->quit = true;
            return (false);
        }
        else if (choice != choice_continue)
        {
            choice = menuReturn(choice);
        }
    }
    // clear console screen
    system("clear");
    return (true);
}

bool ConsoleInput::menu()
{
    string key = "0";
    menuChoice choice = none;

    while (choice != choice_continue)
    {
        // print menu on screen
        printMenuOptions();

        // translate string to enum type "menu_choice"
        cin >> key;
        cout << endl;

        choice = getMenuChoice(key);

        // branching input menu choice options
        menuSwitch(choice);

        // return false if user choose to exit, else it can return to menu or proceed simulation
        if (choice == choice_exit)
        {
            return (false);
        }
        else if (choice != choice_continue)
        {
            choice = menuReturn(choice);
        }
    }
    // clear console screen
    system("clear");
    return (true);
}

void ConsoleInput::printMenuOptions()
{
    // initial menu
    // clear console screen
    system("clear");
    cout << "------------------------------------------------------" << endl;
    cout << "RANDOM WALK NMR SIMULATION PROGRAM" << endl;
    cout << "Set simulation parameters:" << endl
         << endl;
    cout << "0 - NMR simulation name (NAME)" << endl;
    cout << "1 - rock image file path (IMAGEPATH)" << endl;
    cout << "2 - number of images (IMAGES)" << endl;
    cout << "3 - number of steps (STEPS)" << endl;
    cout << "4 - walker occupancy (OCCUPANCY)" << endl;
    cout << "5 - initial seed (SEED)" << endl;
    cout << "6 - gpu usage (GPU)" << endl
         << endl;
    cout << "Digit 'GO' or 'START' to proceed the simulation" << endl;
    cout << "Digit 'RESET' or 'DEFAULT' to reset default options." << endl;
    cout << "Digit 'EXIT' or 'QUIT' to abort." << endl;
    cout << "------------------------------------------------------" << endl;
}

menuChoice ConsoleInput::getMenuChoice(string key)
{
    // translate input to menu choice option
    if (key == "GO" or key == "go" or key == "g" or
        key == "START" or key == "start" or key == "s")
    {
        return (choice_continue);
    }
    else if (key == "0" or
             key == "NAME" or key == "name" or key == "Name")
    {
        return (choice_name);
    }
    else if (key == "1" or
             key == "ROCK" or key == "rock" or
             key == "IMAGEPATH" or key == "imagepath" or
             key == "image" or key == "path")
    {
        return (choice_imagepath);
    }

    else if (key == "2" or
             key == "IMAGES" or key == "images" or
             key == "IMGS" or key == "imgs")
    {
        return (choice_numberOfImages);
    }

    else if (key == "3" or
             key == "STEPS" or key == "steps" or
             key == "STEP" or key == "step")
    {
        return (choice_steps);
    }

    else if (key == "4" or
             key == "OCCUPANCY" or key == "occupancy" or
             key == "OCC" or key == "occ")
    {
        return (choice_occupancy);
    }

    else if (key == "5" or
             key == "INITIAL_SEED" or key == "initial_seed" or
             key == "SEED" or key == "seed")
    {
        return (choice_seed);
    }

    else if (key == "6" or
             key == "GPU_USAGE" or key == "gpu_usage" or
             key == "GPU" or key == "gpu")
    {
        return (choice_gpu);
    }

    else if (key == "R" or key == "r" or
             key == "D" or key == "d" or
             key == "RESET" or key == "reset" or
             key == "DEFAULT" or key == "default")
    {
        return (choice_reset);
    }

    else if (key == "Q" or key == "q" or
             key == "EXIT" or key == "exit" or
             key == "QUIT" or key == "quit")
    {
        return (choice_exit);
    }

    else
    {
        return (none);
    }
}

void ConsoleInput::menuSwitch(menuChoice choice)
{
    string newInput;

    switch (choice)
    {
    case (choice_name):
        cout << "Current simulation name: " << this->simulationName << endl;
        cout << "New name: ";
        cin >> newInput;
        setSimulationName(newInput);
        break;

    case (choice_imagepath):
        cout << "Current image: " << this->imagePath.completePath << endl;
        cout << "New path to directory where image is located: ";
        cin >> newInput;
        setImagePath(newInput);
        cout << "New image file name (e.g. 'AC1_1um_3D_'): ";
        cin >> newInput;
        setImageFileName(newInput);
        cout << "New image file ID (e.g. '1', '23', '560' etc): ";
        cin >> newInput;
        setImageFileID(newInput);
        cout << "New image file extension (e.g. '.png', '.jpg' etc): ";
        cin >> newInput;
        setImageExtension(newInput);
        setRockImagePath();
        break;

    case (choice_numberOfImages):
        cout << "Current number of images: " << this->numberOfImages << endl;
        cout << "New number of images: ";
        cin >> newInput;
        setNumberOfImages(newInput);
        break;

    case (choice_steps):
        cout << "Current number of steps: " << this->steps << endl;
        cout << "New number of steps: ";
        cin >> newInput;
        setNumberOfSteps(newInput);
        break;

    case (choice_occupancy):
        cout << "Current walker occupancy: " << this->occupancy << endl;
        cout << "New walker occupancy: ";
        cin >> newInput;
        setWalkerOccupancy(newInput);
        break;

    case (choice_seed):
        cout << "Current initial seed: " << this->seed << endl;
        cout << "New initial seed: ";
        cin >> newInput;
        setInitialSeed(newInput);
        break;

    case (choice_gpu):
        cout << "Current GPU usage option: ";
        if (this->use_GPU == true)
        {
            cout << "ON" << endl;
        }
        else
            cout << "OFF" << endl;

        cout << "Use GPU to perform simulations? [Y/N] " << endl;
        cin >> newInput;
        setGpuUsage(newInput);
        break;

    case (choice_reset):
        resetConsoleInput();
        cout << "Default options reset." << endl;
        break;

    case (choice_exit):
        cout << "Ok." << endl;
        break;

    case (choice_continue):
        cout << "Ok." << endl;
        break;

    case (none):

        cout << "Sorry, invalid input. Try it again." << endl;
        sleep(1);

        break;
    }

    cout << endl;
}

menuChoice ConsoleInput::menuReturn(menuChoice choice)
{
    string key = "0";

    if (choice != none)
    {
        while (key != "Y" and key != "y")
        {
            cout << "Return to menu? [Y/N]" << endl;
            cin >> key;

            if (key == "N" || key == "n")
            {
                cout << endl;
                return (choice_continue);
            }
        }
        cout << endl;
    }
    return (none);
}

void ConsoleInput::setSimulationName(string userInput)
{
    this->simulationName = userInput;
}

void ConsoleInput::setImagePath(string userInput)
{
    this->imagePath.path = userInput;
}

void ConsoleInput::setImageFileName(string userInput)
{
    this->imagePath.filename = userInput;
}

void ConsoleInput::setImageFileID(string userInput)
{
    stringstream fileID(userInput);
    fileID >> this->imagePath.fileID;
}

void ConsoleInput::setImageExtension(string userInput)
{
    this->imagePath.extension = userInput;
}

void ConsoleInput::setRockImagePath()
{
    (*this).updateCompletePath();
    cout << "new image path: " << this->imagePath.completePath;
}

void ConsoleInput::setNumberOfImages(string userInput)
{
    stringstream images(userInput);
    images >> this->numberOfImages;
}

void ConsoleInput::setNumberOfSteps(string userInput)
{
    stringstream steps(userInput);
    steps >> this->steps;
}

void ConsoleInput::setWalkerOccupancy(string userInput)
{
    stringstream occupancy(userInput);
    occupancy >> this->occupancy;

    if (this->occupancy > 1)
    {
        this->occupancy = (int)this->occupancy % 100;
        this->occupancy = (double)this->occupancy / 100.0;
    }
}

void ConsoleInput::setInitialSeed(string userInput)
{
    stringstream seed(userInput);
    seed >> this->seed;
}

void ConsoleInput::setGpuUsage(string userInput)
{
    if (userInput == "Y" or userInput == "y")
    {
        this->use_GPU = true;
    }
    else if (userInput == "N" or userInput == "n")
    {
        this->use_GPU = false;
    }
    else
    {
        // user informed invalid answer
        cout << "Sorry, invalid answer." << endl;
    }
}
