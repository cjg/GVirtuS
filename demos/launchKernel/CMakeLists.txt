#target_compile_options((launchKernel.e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#        --cudart=shared
#        >)
#
#add_executable(launchKernel.e launchKernel.cu)