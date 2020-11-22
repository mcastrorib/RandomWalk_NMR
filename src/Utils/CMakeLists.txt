add_library(Utils OMPLoopEnabler.cpp)

target_include_directories(Utils
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS Utils DESTINATION lib)
install(FILES OMPLoopEnabler.h DESTINATION include)