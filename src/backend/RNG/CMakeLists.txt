add_library(RNG xorshift.cpp)

target_include_directories(RNG
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS RNG DESTINATION lib)
install(FILES randomIndex.h xorshift.h DESTINATION include)