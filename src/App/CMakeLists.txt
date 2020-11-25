add_library(App App.cpp)

target_include_directories(App
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS App DESTINATION lib)
install(FILES App.h DESTINATION include)