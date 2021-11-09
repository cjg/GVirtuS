cuda_add_executable( dummy0.e dummy0.cu )
target_compile_options(dummy0.e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared>)