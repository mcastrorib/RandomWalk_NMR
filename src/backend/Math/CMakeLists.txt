add_library(Math LeastSquareAdjust.cpp Vector3D.cpp)

target_include_directories(Math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(Math PUBLIC -O3)
install(TARGETS Math DESTINATION lib)
install(FILES LeastSquareAdjust.h Vector3D.h DESTINATION include)