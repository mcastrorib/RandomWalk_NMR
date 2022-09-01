# Add CUDA Toolkit package
find_package(CUDAToolkit)

# Add library to main program
add_library(NMR_Simulation 
			NMR_Simulation.cpp 
			NMR_Simulation_cuda.cpp 
			CollisionHistogram.cpp 
			ChordLengthHistogram.cpp
			InternalField.cpp
			NMR_cpmg.cpp
			NMR_cpmg_cuda.cpp
			NMR_pfgse.cpp
			NMR_pfgse_cuda.cpp
			NMR_multitau.cpp)

# Include directory in build up
target_include_directories(NMR_Simulation
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )


# Request the library to be built with -std=c++11
# As this is public compile feature, anything that links to
# This lib will also be build with -std=c++11
target_compile_features(NMR_Simulation PUBLIC cxx_std_11)

# Explicitly state that all CUDA files in the 
# library needs to be built with -dc as the member functions 
# could be called by other libraries and executables
set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp -O3 -w")
set_source_files_properties(NMR_Simulation_cuda.h PROPERTIES LANGUAGE "CUDA")
set_source_files_properties(NMR_Simulation_cuda.cpp PROPERTIES LANGUAGE "CUDA")
set_source_files_properties(NMR_cpmg_cuda.h PROPERTIES LANGUAGE "CUDA")
set_source_files_properties(NMR_cpmg_cuda.cpp PROPERTIES LANGUAGE "CUDA")
set_source_files_properties(NMR_pfgse_cuda.h PROPERTIES LANGUAGE "CUDA")
set_source_files_properties(NMR_pfgse_cuda.cpp PROPERTIES LANGUAGE "CUDA")

# Set CUDA architecture and link to CUDA runtime environment
# For better results, add current device architecture
find_package(CUDA)
if (CUDA_FOUND)
  #Get CUDA compute capability
  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_check/cuda_script) # No suffix required
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_check/check_cuda.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(STATUS "CUDA Architecture: ${ARCH}")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    # message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
    # message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")

    set(CUDA_NVCC_FLAGS "${ARCH}")
	set_property(TARGET NMR_Simulation PROPERTY CUDA_ARCHITECTURES "${ARCH}")
    add_definitions(-DGPU) #You may not require this

  else()
    message(WARNING ${ARCH})
  endif()
endif()
# set_property(TARGET NMR_Simulation PROPERTY CUDA_ARCHITECTURES 35 50 72 75 86)
target_compile_options(NMR_Simulation PUBLIC -O3)
target_link_libraries(NMR_Simulation PRIVATE CUDA::cudart)

# Install lib in project tree
install(TARGETS NMR_Simulation DESTINATION lib)
install(FILES 
		CollisionHistogram.h
		ChordLengthHistogram.h
		InternalField.h
		NMR_Simulation.h 
		NMR_Simulation_cuda.h 
		NMR_cpmg.h
		NMR_cpmg_cuda.h
		NMR_pfgse.h
		NMR_pfgse_cuda.h
		NMR_multitau.h
		NMR_defs.h DESTINATION include)
