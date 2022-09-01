add_library(FileHandler fileHandler.cpp smallFunctions.cpp baseFunctions.cpp)

target_include_directories(FileHandler
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
target_compile_options(FileHandler PUBLIC -O3)
install(TARGETS FileHandler DESTINATION lib)
install(FILES fileHandler.h DESTINATION include)