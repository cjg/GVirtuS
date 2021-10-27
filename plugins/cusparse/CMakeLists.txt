cmake_minimum_required(VERSION 3.10.2)
project(gvirtus-plugin-cusparse)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

gvirtus_add_backend(cusparse ${CUDA_VERSION}
        backend/CusparseHandler.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cuspare ${CUDA_VERSION}
        frontend/Cusparse.cpp
        frontend/CusparseFrontend.cpp
	frontend/Cusparse_helper.cpp
)
