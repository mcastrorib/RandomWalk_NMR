add_library(Walker walker.cpp)

target_include_directories(Walker
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
target_compile_options(Walker PUBLIC -O3)
install(TARGETS Walker DESTINATION lib)
install(FILES walker.h walker_defs.h DESTINATION include)