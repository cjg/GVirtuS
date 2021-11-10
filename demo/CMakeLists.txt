find_package(CUDA REQUIRED)
set(CUDA_RUNTIME_LIBRARY shared)


add_subdirectory(dummy0)
add_subdirectory(launchKernel)
add_subdirectory(matrixMul)
