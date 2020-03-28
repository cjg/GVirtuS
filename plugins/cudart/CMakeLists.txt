cmake_minimum_required(VERSION 3.10.2)
project(gvirtus-plugin-cudart)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} util 3rdparty/include)

gvirtus_add_backend(cudart ${CUDA_VERSION}
        backend/CudaRtHandler_texture.cpp
        backend/CudaRtHandler_event.cpp
        backend/CudaRtHandler_opengl.cpp
        backend/CudaRtHandler_surface.cpp
        backend/CudaRtHandler_thread.cpp
        backend/CudaRtHandler_internal.cpp
        backend/CudaRtHandler_occupancy.cpp
        backend/CudaRtHandler_execution.cpp
        backend/CudaRtHandler_device.cpp
        backend/CudaRtHandler_stream.cpp
        backend/CudaRtHandler_memory.cpp
        backend/CudaRtHandler.cpp
        backend/CudaRtHandler_version.cpp
        backend/CudaRtHandler_error.cpp
        util/CudaUtil.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cudart ${CUDA_VERSION}
        frontend/CudaRt.cpp
        frontend/CudaRt_device.cpp
        frontend/CudaRt_error.cpp
        frontend/CudaRt_event.cpp
        frontend/CudaRt_execution.cpp
        frontend/CudaRtFrontend.cpp
        frontend/CudaRt_internal.cpp
        frontend/CudaRt_memory.cpp
        frontend/CudaRt_occupancy.cpp
        frontend/CudaRt_opengl.cpp
        frontend/CudaRt_stream.cpp
        frontend/CudaRt_surface.cpp
        frontend/CudaRt_texture.cpp
        frontend/CudaRt_thread.cpp
        frontend/CudaRt_version.cpp
        util/CudaUtil.cpp)
