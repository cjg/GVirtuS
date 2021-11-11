cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-curand)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

gvirtus_add_backend(curand ${CUDA_VERSION}
        backend/CurandHandler.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDA_curand_LIBRARY} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(curand ${CUDA_VERSION}
        frontend/Curand.cpp
        frontend/CurandFrontend.cpp)
