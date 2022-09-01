add_library(BitBlock bitBlock.cpp)

target_include_directories(BitBlock
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
target_compile_options(BitBlock PUBLIC -O3)
install(TARGETS BitBlock DESTINATION lib)
install(FILES bitBlock.h DESTINATION include)