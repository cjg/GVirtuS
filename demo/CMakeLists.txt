find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --cudart=shared)


add_subdirectory(dummy0)
add_subdirectory(launchKernel)
add_subdirectory(matrixMul)