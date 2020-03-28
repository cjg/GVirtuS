cmake_minimum_required(VERSION 3.10.2)
project(gvirtus-plugin-cufft)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} util 3rdparty/include)

gvirtus_add_backend(cufft ${CUDA_VERSION}
        backend/CufftHandler.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDA_cublas_LIBRARY} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cufft ${CUDA_VERSION}
        frontend/Cufft.cpp
        frontend/CufftFrontend.cpp)
