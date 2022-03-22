add_library(ConfigFiles rwnmr_config.cpp 
						uct_config.cpp 
						cpmg_config.cpp 
						pfgse_config.cpp
						multitau_config.cpp
						ga_config.cpp)


target_include_directories(ConfigFiles
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

install(TARGETS ConfigFiles DESTINATION lib)
install(FILES configFiles_defs.h
			  rwnmr_config.h 
			  uct_config.h 
			  cpmg_config.h 
			  pfgse_config.h
			  multitau_config.h
			  ga_config.h DESTINATION include)
