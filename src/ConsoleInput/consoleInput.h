#ifndef CONSOLE_INPUT_H
#define CONSOLE_INPUT_H

#include <sstream>
#include <iomanip>

using namespace std;

// define default input values
#define DEFAULT_NAME "NMR_simulation"
#define DEFAULT_NUMBEROFIMAGES 1
#define DEFAULT_OCCUPANCY 1
#define DEFAULT_STEPS 40000
#define DEFAULT_SEED 0
#define DEFAULT_IMAGE_PATH "../../Images/tiny_3D/imgs/"
#define DEFAULT_IMAGE_FILE "tiny_3D_"
#define DEFAULT_IMAGE_ID 0
#define DEFAULT_IMAGE_DIGITS 3
#define DEFAULT_IMAGE_EXTENSION ".png"
#define DEFAULT_T2_PATH "../data/input_data/PSO_benchmark_2D/NMR_T2.txt"
#define DEFAULT_GPU_USAGE true

typedef struct imagePath
{
    string path;
    string filename;
    uint fileID;
    uint digits;
    string extension;
    string completePath;
} ImagePath;

typedef enum menu_choice
{
    choice_name,
    choice_imagepath,
    choice_numberOfImages,
    choice_T2path,
    choice_steps,
    choice_occupancy,
    choice_seed,
    choice_gpu,
    choice_continue,
    choice_exit,
    choice_reset,
    none
} menuChoice;

class ConsoleInput
{
public:
    string simulationName;
    uint numberOfImages;
    uint steps;
    double occupancy;
    ImagePath imagePath;
    string T2path;
    uint64_t seed;
    bool use_GPU;
    bool quit;

    // default constructor and destructor
    ConsoleInput();
    virtual ~ConsoleInput(){};

    void resetConsoleInput();

    bool menu();
    bool menuSetImage();
    bool menuSetWalkers();
    bool menuGA();
    bool quitProgram();
    bool menuContinue();

    void printMenuOptions();
    menuChoice getMenuChoice(string key);
    void menuSwitch(menuChoice _choice);
    menuChoice menuReturn(menuChoice _choice);

    void setSimulationName(string userInput);
    void setImagePath(string newInput);
    void setImageFileName(string newInput);
    void setImageFileID(string newInput);
    void setImageExtension(string newInput);
    void setRockImagePath();
    void setT2Path(string newInput);
    void setNumberOfImages(string userInput);
    void setNumberOfSteps(string userInput);
    void setWalkerOccupancy(string userInput);
    void setInitialSeed(string userInput);
    void setGpuUsage(string userInput);
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
