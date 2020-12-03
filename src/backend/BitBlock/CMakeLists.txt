add_library(BitBlock bitBlock.cpp)

target_include_directories(BitBlock
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS BitBlock DESTINATION lib)
install(FILES bitBlock.h DESTINATION include)