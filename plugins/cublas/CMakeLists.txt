cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cublas)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

gvirtus_add_backend(cublas ${CUDA_VERSION}
        backend/CublasHandler.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDA_cublas_LIBRARY} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cublas ${CUDA_VERSION}
        frontend/Cublas_helper.cpp
        frontend/Cublas_level1.cpp
        frontend/Cublas_level2.cpp
        frontend/Cublas_level3.cpp
        frontend/CublasFrontend.cpp)
