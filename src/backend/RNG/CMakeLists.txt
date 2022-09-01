add_library(RNG xorshift.cpp)

target_include_directories(RNG
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
target_compile_options(RNG PUBLIC -O3)
install(TARGETS RNG DESTINATION lib)
install(FILES randomIndex.h xorshift.h DESTINATION include)