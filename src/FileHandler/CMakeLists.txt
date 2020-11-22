add_library(FileHandler fileHandler.cpp smallFunctions.cpp baseFunctions.cpp)

target_include_directories(FileHandler
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS FileHandler DESTINATION lib)
install(FILES fileHandler.h DESTINATION include)