add_library(ParticleSwarm appPSO.cpp PSO.cpp basic.cpp)

target_include_directories(ParticleSwarm
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS ParticleSwarm DESTINATION lib)
install(FILES PSO.h DESTINATION include)
