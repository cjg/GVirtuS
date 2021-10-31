cmake_minimum_required(VERSION 3.18)
project("protocol-generator")

find_library(LIBCLANG_CPP_LIBRARY clang)

if(LIBCLANG_CPP_LIBRARY)
    message(STATUS "Found Clang library (${LIBCLANG_CPP_LIBRARY}")
else()
    message(STATUS "NOT found Clang library")
endif()

find_file(CLANG_AST_HEADER  clang/AST/AST.h)

if(CLANG_AST_HEADER)
    get_filename_component(LIBCLANG_INCLUDES ${CLANG_AST_HEADER} DIRECTORY)
    get_filename_component(LIBCLANG_INCLUDES ${LIBCLANG_INCLUDES} DIRECTORY)
    get_filename_component(LIBCLANG_INCLUDES ${LIBCLANG_INCLUDES} DIRECTORY)
    message(STATUS "Found Clang headers (${LIBCLANG_INCLUDES})")
else()
    message(STATUS "NOT found Clang headers")
endif()

if(LIBCLANG_CPP_LIBRARY AND CLANG_AST_HEADER)
    message(STATUS "Building ${PROJECT_NAME}")
    add_executable(${PROJECT_NAME}
            main.cpp)
    target_link_libraries(${PROJECT_NAME} ${LIBCLANG_CPP_LIBRARY})
else()
    message(STATUS "NOT building ${PROJECT_NAME}")
endif()
