// include C++ standard libraries
#include <iostream>
#include <sstream>
#include <cstdint>
#include <vector>
#include <string>

#include "ArgsParser.h"

ArgsParser::ArgsParser(int argc, char *argv[])
{
    vector<string> commands();
    vector<string> paths();

    // initialize essentials
    this->commands.push_back("rwnmr");
    this->paths.push_back("default");
    this->commands.push_back("uct");
    this->paths.push_back("default");

    // read args
    if(argc > 1)
    {
        int arg_idx = 1;
        while(arg_idx < argc)
        {
            string argument(argv[arg_idx]);

            if(argument == "-rwconfig") 
            {
                
                if((arg_idx + 1) < argc)
                {
                    string new_path = argv[arg_idx+1];
                    this->paths[0] = new_path;
                }                
            }
            else if(argument == "-uctconfig")
            {                
                if((arg_idx + 1) < argc)
                {
                    string new_path = argv[arg_idx+1];
                    this->paths[1] = new_path;
                }                
            }
            else if(argument == "cpmg")
            {
                this->commands.push_back("cpmg");

                if((arg_idx + 1) < argc)
                {
                    string add_flag = argv[arg_idx + 1];
                    if(add_flag == "-config" and (arg_idx + 2) < argc)
                    {
                        string new_path = argv[arg_idx + 2];
                        this->paths.push_back(new_path);
                    } else
                    {
                        this->paths.push_back("default");
                    }
                } else
                {
                    this->paths.push_back("default");
                }                
            }
            else if(argument == "pfgse")
            {
                this->commands.push_back("pfgse");

                if((arg_idx + 1) < argc)
                {
                    string add_flag = argv[arg_idx + 1];
                    if(add_flag == "-config" and (arg_idx + 2) < argc)
                    {
                        string new_path = argv[arg_idx + 2];
                        this->paths.push_back("default");
                    } else
                    {
                        this->paths.push_back("default");
                    }  
                } else
                {
                    this->paths.push_back("default");
                }  
            }
            else if(argument == "ga")
            {
                this->commands.push_back("ga");

                if((arg_idx + 1) < argc)
                {
                    string add_flag = argv[arg_idx + 1];
                    if(add_flag == "-config" and (arg_idx + 2) < argc)
                    {
                        string new_path = argv[arg_idx + 2];
                        this->paths.push_back("default");
                    } else
                    {
                        this->paths.push_back("default");
                    }
                } else
                {
                    this->paths.push_back("default");
                }  
            }
            // increment argument
            arg_idx++;
        }
    }

}

ArgsParser::ArgsParser(const ArgsParser &_otherParser)
{
    this->commands = _otherParser.commands;
    this->paths = _otherParser.paths;    
}