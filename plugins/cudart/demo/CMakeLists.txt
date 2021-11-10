
cuda_add_executable( add_grid.e add_grid.cu)
set_target_properties(add_grid.e PROPERTIES CUDA_RUNTIME_LIBRARY "shared")
install(TARGETS add_grid.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)

cuda_add_executable( dummy0.e dummy0.cu)
set_target_properties(dummy0.e PROPERTIES CUDA_RUNTIME_LIBRARY "shared")
install(TARGETS dummy0.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)

cuda_add_executable(launchKernel.e launchKernel.cu)
set_target_properties(launchKernel.e PROPERTIES CUDA_RUNTIME_LIBRARY "shared")
install(TARGETS launchKernel.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)

cuda_add_executable(matrixMul.e matrixMul.cu)
set_target_properties(matrixMul.e PROPERTIES CUDA_RUNTIME_LIBRARY "shared")
install(TARGETS matrixMul.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)