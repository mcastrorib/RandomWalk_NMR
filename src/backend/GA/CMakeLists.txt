add_library(GA 
			ga_core.cpp 			 
			individual.cpp
			mpi_ga_island.cpp)
			
target_include_directories(GA
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
target_compile_options(GA PUBLIC -O3)
install(TARGETS GA DESTINATION lib)
install(FILES ga_defs.h 
			  ga_core.h
			  mpi_ga_island.h
			  ga_output.h 
			  ga_parameters.h 
			  ga_problem.h 
			  population.h 			   			   
			  individual.h
			  DESTINATION include)