cuda_add_executable(launchKernel.e lauchKernel.cu)
target_compile_options(launchKernel.e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared>)