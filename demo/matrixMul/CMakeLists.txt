cuda_add_executable(matrixMul.e matrixMul.cu)
target_compile_options(matrixMul.e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared>)