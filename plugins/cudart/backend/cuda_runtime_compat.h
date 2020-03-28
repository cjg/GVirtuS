#if !defined(__CUDA_RUNTIME_COMPAT_H__)
#define __CUDA_RUNTIME_COMPAT_H__

/** \cond impl_private */
#if !defined(__dv)

#if defined(__cplusplus)

#define __dv(v) = v

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */
/** \endcond impl_private */

extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg,
                                                        size_t size,
                                                        size_t offset);
extern __host__ cudaError_t CUDARTAPI
cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0),
                  cudaStream_t stream __dv(0));
extern __host__ __device__ unsigned cudaConfigureCall(dim3 gridDim,
                                                      dim3 blockDim,
                                                      size_t sharedMem = 0,
                                                      void *stream = 0);

template <class T>
static __inline__ __host__ cudaError_t cudaLaunch(T *func) {
  return ::cudaLaunch((const void *)func);
}

template <class T>
static __inline__ __host__ cudaError_t cudaSetupArgument(T arg, size_t offset) {
  return ::cudaSetupArgument((const void *)&arg, sizeof(T), offset);
}

#endif /* !__CUDA_RUNTIME_COMPAT_H__ */
