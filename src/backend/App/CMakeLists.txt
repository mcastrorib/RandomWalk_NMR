add_library(App rwnmrApp.cpp)

target_include_directories(App
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS App DESTINATION lib)
install(FILES rwnmrApp.h DESTINATION include)