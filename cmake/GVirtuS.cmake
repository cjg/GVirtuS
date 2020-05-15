include_directories(include)

if(NOT ${PROJECT_NAME} STREQUAL "gvirtus-common")
    include_directories(${GVIRTUS_HOME}/include)
    link_directories(${GVIRTUS_HOME}/lib)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGVIRTUS_HOME=\\\"${GVIRTUS_HOME}\\\"")
set(CMAKE_SKIP_RPATH ON)

find_library(LOG4CPLUS_LIBRARY log4cplus)
if(NOT LOG4CPLUS_LIBRARY)
    message(FATAL_ERROR "log4cplus library not found")
endif()

find_package(Threads REQUIRED)

if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus)
    message(STATUS "Installing ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus to ${GVIRTUS_HOME}/include")
    install(DIRECTORY include/gvirtus DESTINATION ${GVIRTUS_HOME}/include)
endif()

macro(gvirtus_install_target target_name)
    install(TARGETS ${target_name}
            LIBRARY DESTINATION ${GVIRTUS_HOME}/lib
            ARCHIVE DESTINATION ${GVIRTUS_HOME}/lib
            RUNTIME DESTINATION ${GVIRTUS_HOME}/bin
            INCLUDES DESTINATION ${GVIRTUS_HOME}/include)
endmacro()

function(gvirtus_add_backend)
    if(ARGC LESS 3)
        message(FATAL_ERROR "Usage: gvirtus_add_backend(wrapped_library version source1 ...)")
    endif()
    list(GET ARGV 0 wrapped_library)
    list(REMOVE_AT ARGV 0)
    list(GET ARGV 0 version)
    list(REMOVE_AT ARGV 0)
    if(NOT "${PROJECT_NAME}" STREQUAL "gvirtus-plugin-${wrapped_library}")
        message(FATAL_ERROR "This project should be named gvirtus-plugin-${wrapped_library}")
    endif()
    add_library(${PROJECT_NAME} SHARED
            ${ARGV})
    target_link_libraries(${PROJECT_NAME} log4cplus gvirtus-common gvirtus-communicators)
    install(TARGETS ${PROJECT_NAME}
            LIBRARY DESTINATION ${GVIRTUS_HOME}/lib)
endfunction()

function(gvirtus_add_frontend)
    if(ARGC LESS 3)
        message(FATAL_ERROR "Usage: gvirtus_add_frontend(wrapped_library version source1 ...)")
    endif()
    list(GET ARGV 0 wrapped_library)
    list(REMOVE_AT ARGV 0)
    list(GET ARGV 0 version)
    list(REMOVE_AT ARGV 0)
    add_library(${wrapped_library} SHARED
            ${ARGV})
    target_link_libraries(${wrapped_library} log4cplus gvirtus-common gvirtus-communicators gvirtus-frontend)
    set_target_properties(${wrapped_library} PROPERTIES VERSION ${version})
    string(REGEX REPLACE "\\..*" "" soversion ${version})
    set_target_properties(${wrapped_library} PROPERTIES SOVERSION ${soversion})
    string(TOUPPER "LIB${wrapped_library}_${version}" script)
    set(script "${script} {\n local:\n*;\n};")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lib${wrapped_library}.map "${script}")
    install(TARGETS ${wrapped_library}
            LIBRARY DESTINATION ${GVIRTUS_HOME}/lib/frontend)
endfunction()