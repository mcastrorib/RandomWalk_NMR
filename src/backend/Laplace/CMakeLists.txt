add_library(Laplace tikhonov.cpp 
                    ./src/nmrinv_core.cpp 
                    ./src/nmrinv.cpp 
                    ./src/TikhonovSolver.cpp 
                    ./src/vector_funcs.cpp)
                           
include_directories(./include)
include_directories(/usr/include/eigen3)    
      
target_include_directories(Laplace
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS Laplace 
                DESTINATION lib)

install(FILES tikhonov.h 
              ./include/nmrinv_core.h 
              ./include/nmrinv.h 
              ./include/TikhonovSolver.h 
              ./include/vector_funcs.h 
              DESTINATION include)