add_library(Utils OMPLoopEnabler.cpp myAllocator.cpp ArgsParser.cpp ImagePath.cpp)

target_include_directories(Utils
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS Utils DESTINATION lib)
install(FILES OMPLoopEnabler.h myAllocator.h ArgsParser.h ImagePath.h DESTINATION include)