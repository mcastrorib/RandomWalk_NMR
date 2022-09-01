add_library(App rwnmrApp.cpp)

target_include_directories(App
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

target_compile_options(App PUBLIC -O3)
install(TARGETS App DESTINATION lib)
install(FILES rwnmrApp.h DESTINATION include)