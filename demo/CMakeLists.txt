find_package(CUDA REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")

add_subdirectory(dummy0)
add_subdirectory(launchKernel)
add_subdirectory(matrixMul)