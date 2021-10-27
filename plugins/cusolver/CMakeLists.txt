cmake_minimum_required(VERSION 3.10.2)
project(gvirtus-plugin-cusolver)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

gvirtus_add_backend(cusolver ${CUDA_VERSION}
        backend/CusolverHandler.cpp
)
target_link_libraries(${PROJECT_NAME} ${CUDA_cusolver_LIBRARY} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cusolver ${CUDA_VERSION}
        frontend/Cusolver.cpp
        frontend/CusolverFrontend.cpp
        frontend/Cusolver_helper.cpp
)
