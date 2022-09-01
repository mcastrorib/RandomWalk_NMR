add_library(Utils OMPLoopEnabler.cpp myAllocator.cpp ArgsParser.cpp ImagePath.cpp)

target_include_directories(Utils
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
target_compile_options(Utils PUBLIC -O3)
install(TARGETS Utils DESTINATION lib)
install(FILES OMPLoopEnabler.h myAllocator.h ArgsParser.h ImagePath.h ProgressBar.h DESTINATION include)