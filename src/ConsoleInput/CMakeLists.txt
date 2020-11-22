add_library(ConsoleInput consoleInput.cpp)

target_include_directories(ConsoleInput
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )


install(TARGETS ConsoleInput DESTINATION lib)
install(FILES consoleInput.h DESTINATION include)
