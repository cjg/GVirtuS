/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 */
/*
#include "Handler.h"
#include "communicator/Result.h"
#include "CudaUtil.h"
#include <cudnn.h>

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

class CudnnHandler : public Handler {
public:
    CudnnHandler();
    virtual ~CudnnHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);
*/
    /*void * RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);
    *//*
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef Result * (*CudnnRoutineHandler)(CudnnHandler *, Buffer *);
    static std::map<std::string, CudnnRoutineHandler> * mspHandlers;
    //void **pointers;
    //int nPointers;
    
    //std::map<std::string, std::string> * mpMapObject;

    //void *mpShm;
    //int mShmFd;
};

#define CUDNN_ROUTINE_HANDLER(name) Result * handle##name(CudnnHandler * pThis, Buffer * in)
#define CUDNN_ROUTINE_HANDLER_PAIR(name) make_pair("cudnn" #name, handle##name)
*/
/* CudnnHandler_Platform *//*
CUDNN_ROUTINE_HANDLER(GetVersion);
CUDNN_ROUTINE_HANDLER(Create);
CUDNN_ROUTINE_HANDLER(Destroy);
CUDNN_ROUTINE_HANDLER(GetErrorString);
CUDNN_ROUTINE_HANDLER(SetStream);
CUDNN_ROUTINE_HANDLER(GetStream);
CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptorEx);
CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyTensorDescriptor);
CUDNN_ROUTINE_HANDLER(TransformTensor);
CUDNN_ROUTINE_HANDLER(AddTensor);
CUDNN_ROUTINE_HANDLER(OpTensor);
CUDNN_ROUTINE_HANDLER(SetTensor);
CUDNN_ROUTINE_HANDLER(ScaleTensor);
CUDNN_ROUTINE_HANDLER(CreateFilterDescriptor);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v3);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v3);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v4);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v4);
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor);
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v3);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v3);
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v4);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v4);
CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor);
CUDNN_ROUTINE_HANDLER(CreateConvolutionDescriptor);
CUDNN_ROUTINE_HANDLER(SetConvolution2dDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolution2dDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolution2dForwardOutputDim);
 //_CUDNNHANDLER_H*/
/* This file is part of gVirtuS.
*  *
* gVirtuS is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
* gVirtuS is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with gVirtuS; if not, write to the Free Software
* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*
* Written by: Raffaele Montella <raffaele.montella@uniparthenope.it>,
*             Department of Science and Technologies
*                   
 * #include "Backend.h"
 *
 * class CudnnBackend : public Backend {
 * public:
 *     Handler *GetHandler();
 *     };
 *     */

#ifndef CUDNNHANDLER_H
#define CUDNNHANDLER_H

#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/Result.h>

#include <cudnn.h>

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>

#include <limits.h>
#if ( __WORDSIZE == 64 )
    #define BUILD_64   1
#endif

/*                                             
 * CudaRtHandler is used by Backend's Process(es) for storing and retrieving
 * device related data and functions. 
 * CudaRtHandler has also the method Execute() that is responsible to execute a
 * named CUDA Runtime routine unmarshalling the input parameters from the
 * provided Buffer.
 */
using namespace std;
using namespace log4cplus;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

class CudnnHandler : public gvirtus::backend::Handler {
public:
    CudnnHandler();
    virtual ~CudnnHandler();
    bool CanExecute(std::string routine);
    std::shared_ptr<Result> Execute(std::string routine, std::shared_ptr<Buffer> input_buffer);
    static void setLogLevel(Logger *logger);
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef std::shared_ptr<Result> (*CudnnRoutineHandler)(CudnnHandler *, std::shared_ptr<Buffer>);
    static std::map<std::string, CudnnRoutineHandler> * mspHandlers;
};

#define CUDNN_ROUTINE_HANDLER(name) std::shared_ptr<Result> handle##name(CudnnHandler * pThis, std::shared_ptr<Buffer> in)
#define CUDNN_ROUTINE_HANDLER_PAIR(name) make_pair("cudnn" #name, handle##name)

/* CudnnHandler.cpp */
CUDNN_ROUTINE_HANDLER(GetVersion);
CUDNN_ROUTINE_HANDLER(GetErrorString);
CUDNN_ROUTINE_HANDLER(Create);
CUDNN_ROUTINE_HANDLER(Destroy);
CUDNN_ROUTINE_HANDLER(SetStream);
CUDNN_ROUTINE_HANDLER(GetStream); 
CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptorEx);
CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptorEx);
CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetTensorSizeInBytes);
CUDNN_ROUTINE_HANDLER(DestroyTensorDescriptor);
CUDNN_ROUTINE_HANDLER(InitTransformDest);
CUDNN_ROUTINE_HANDLER(CreateTensorTransformDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensorTransformDescriptor);
CUDNN_ROUTINE_HANDLER(GetTensorTransformDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyTensorTransformDescriptor);
CUDNN_ROUTINE_HANDLER(TransformTensor);
CUDNN_ROUTINE_HANDLER(TransformTensorEx);
CUDNN_ROUTINE_HANDLER(GetFoldedConvBackwardDataDescriptors);
CUDNN_ROUTINE_HANDLER(AddTensor);
CUDNN_ROUTINE_HANDLER(CreateOpTensorDescriptor);
CUDNN_ROUTINE_HANDLER(SetOpTensorDescriptor);
CUDNN_ROUTINE_HANDLER(GetOpTensorDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyOpTensorDescriptor);
CUDNN_ROUTINE_HANDLER(OpTensor);
CUDNN_ROUTINE_HANDLER(CreateReduceTensorDescriptor);
CUDNN_ROUTINE_HANDLER(SetReduceTensorDescriptor);
CUDNN_ROUTINE_HANDLER(GetReduceTensorDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyReduceTensorDescriptor);
CUDNN_ROUTINE_HANDLER(GetReductionIndicesSize);
CUDNN_ROUTINE_HANDLER(GetReductionWorkspaceSize);
CUDNN_ROUTINE_HANDLER(ReduceTensor);
CUDNN_ROUTINE_HANDLER(SetTensor);
CUDNN_ROUTINE_HANDLER(ScaleTensor);
CUDNN_ROUTINE_HANDLER(CreateFilterDescriptor);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor);
#if CUDNN_VERSION < 6000
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v3);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v3);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v4);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v4);
#endif
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor);
#if CUDNN_VERSION < 6000
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v3);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v3);
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v4);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v4);
#endif
CUDNN_ROUTINE_HANDLER(GetFilterSizeInBytes);
CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor);
CUDNN_ROUTINE_HANDLER(TransformFilter);
CUDNN_ROUTINE_HANDLER(ReorderFilterAndBias);
CUDNN_ROUTINE_HANDLER(CreateConvolutionDescriptor);
CUDNN_ROUTINE_HANDLER(SetConvolutionMathType);
CUDNN_ROUTINE_HANDLER(GetConvolutionMathType);
CUDNN_ROUTINE_HANDLER(SetConvolutionGroupCount);
CUDNN_ROUTINE_HANDLER(GetConvolutionGroupCount);
CUDNN_ROUTINE_HANDLER(SetConvolutionReorderType);
CUDNN_ROUTINE_HANDLER(GetConvolutionReorderType);
CUDNN_ROUTINE_HANDLER(SetConvolution2dDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolution2dDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolution2dForwardOutputDim);
CUDNN_ROUTINE_HANDLER(SetConvolutionNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolutionNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolutionNdForwardOutputDim);
CUDNN_ROUTINE_HANDLER(DestroyConvolutionDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithmMaxCount);
CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithm);
CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithmEx);
#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm);
#endif
#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm_v7);
#endif
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardWorkspaceSize);
CUDNN_ROUTINE_HANDLER(ConvolutionForward);
CUDNN_ROUTINE_HANDLER(ConvolutionBiasActivationForward);
CUDNN_ROUTINE_HANDLER(ConvolutionBackwardBias);
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithmMaxCount);
CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithm);
CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithmEx);
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm);
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm_v7);
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterWorkspaceSize);
CUDNN_ROUTINE_HANDLER(ConvolutionBackwardFilter);
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithmMaxCount);
CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardDataAlgorithm);
CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardDataAlgorithmEx);
#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithm);
#endif
#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithm_v7);
#endif
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataWorkspaceSize);
CUDNN_ROUTINE_HANDLER(ConvolutionBackwardData);
CUDNN_ROUTINE_HANDLER(Im2Col);
CUDNN_ROUTINE_HANDLER(SoftmaxForward);
CUDNN_ROUTINE_HANDLER(SoftmaxBackward);
CUDNN_ROUTINE_HANDLER(CreatePoolingDescriptor);
CUDNN_ROUTINE_HANDLER(SetPooling2dDescriptor);
CUDNN_ROUTINE_HANDLER(GetPooling2dDescriptor);
CUDNN_ROUTINE_HANDLER(SetPoolingNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetPoolingNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetPoolingNdForwardOutputDim);
CUDNN_ROUTINE_HANDLER(GetPooling2dForwardOutputDim);
CUDNN_ROUTINE_HANDLER(DestroyPoolingDescriptor);
CUDNN_ROUTINE_HANDLER(PoolingForward);
CUDNN_ROUTINE_HANDLER(PoolingBackward);
CUDNN_ROUTINE_HANDLER(CreateActivationDescriptor);
CUDNN_ROUTINE_HANDLER(SetActivationDescriptor);
CUDNN_ROUTINE_HANDLER(GetActivationDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyActivationDescriptor);
CUDNN_ROUTINE_HANDLER(ActivationForward);
CUDNN_ROUTINE_HANDLER(ActivationBackward);
CUDNN_ROUTINE_HANDLER(CreateLRNDescriptor);
CUDNN_ROUTINE_HANDLER(SetLRNDescriptor);
CUDNN_ROUTINE_HANDLER(GetLRNDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyLRNDescriptor);
CUDNN_ROUTINE_HANDLER(LRNCrossChannelForward);
CUDNN_ROUTINE_HANDLER(LRNCrossChannelBackward);
CUDNN_ROUTINE_HANDLER(DivisiveNormalizationForward);
CUDNN_ROUTINE_HANDLER(DivisiveNormalizationBackward);
CUDNN_ROUTINE_HANDLER(DeriveBNTensorDescriptor);
CUDNN_ROUTINE_HANDLER(GetBatchNormalizationForwardTrainingExWorkspaceSize);
CUDNN_ROUTINE_HANDLER(GetBatchNormalizationBackwardExWorkspaceSize);
CUDNN_ROUTINE_HANDLER(GetBatchNormalizationTrainingExReserveSpaceSize);
CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardTraining);
CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardTrainingEx);
CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardInference);
CUDNN_ROUTINE_HANDLER(BatchNormalizationBackward);
CUDNN_ROUTINE_HANDLER(BatchNormalizationBackwardEx);
CUDNN_ROUTINE_HANDLER(CreateSpatialTransformerDescriptor);
CUDNN_ROUTINE_HANDLER(SetSpatialTransformerNdDescriptor);
CUDNN_ROUTINE_HANDLER(DestroySpatialTransformerDescriptor);
CUDNN_ROUTINE_HANDLER(SpatialTfGridGeneratorForward);
CUDNN_ROUTINE_HANDLER(SpatialTfGridGeneratorBackward);
CUDNN_ROUTINE_HANDLER(SpatialTfSamplerForward);
CUDNN_ROUTINE_HANDLER(SpatialTfSamplerBackward);
CUDNN_ROUTINE_HANDLER(CreateDropoutDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyDropoutDescriptor);
CUDNN_ROUTINE_HANDLER(DropoutGetStatesSize);
CUDNN_ROUTINE_HANDLER(DropoutGetReserveSpaceSize);
CUDNN_ROUTINE_HANDLER(SetDropoutDescriptor);
CUDNN_ROUTINE_HANDLER(RestoreDropoutDescriptor);
CUDNN_ROUTINE_HANDLER(GetDropoutDescriptor);
CUDNN_ROUTINE_HANDLER(DropoutForward);
CUDNN_ROUTINE_HANDLER(DropoutBackward);
CUDNN_ROUTINE_HANDLER(CreateRNNDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyRNNDescriptor);
#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v5);
//CUDNN_ROUTINE_HANDLER(GetRNNDescriptor_v5);
#endif
#if CUDNN_VERSION >= 6000
CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v6);
CUDNN_ROUTINE_HANDLER(GetRNNDescriptor_v6);
#endif
#if CUDNN_VERSION >= 8000
CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v8);
CUDNN_ROUTINE_HANDLER(GetRNNDescriptor_v8);
#endif
CUDNN_ROUTINE_HANDLER(SetRNNMatrixMathType);
CUDNN_ROUTINE_HANDLER(GetRNNMatrixMathType);
CUDNN_ROUTINE_HANDLER(SetRNNBiasMode);
CUDNN_ROUTINE_HANDLER(GetRNNBiasMode);
CUDNN_ROUTINE_HANDLER(RNNSetClip);
CUDNN_ROUTINE_HANDLER(RNNGetClip);
CUDNN_ROUTINE_HANDLER(SetRNNProjectionLayers);
CUDNN_ROUTINE_HANDLER(GetRNNProjectionLayers);
CUDNN_ROUTINE_HANDLER(CreatePersistentRNNPlan);
CUDNN_ROUTINE_HANDLER(DestroyPersistentRNNPlan);
CUDNN_ROUTINE_HANDLER(SetPersistentRNNPlan);
CUDNN_ROUTINE_HANDLER(GetRNNWorkspaceSize);
CUDNN_ROUTINE_HANDLER(GetRNNTrainingReserveSize);
CUDNN_ROUTINE_HANDLER(GetRNNParamsSize);
CUDNN_ROUTINE_HANDLER(GetRNNLinLayerMatrixParams);
CUDNN_ROUTINE_HANDLER(GetRNNLinLayerBiasParams);
CUDNN_ROUTINE_HANDLER(RNNForwardInference);
CUDNN_ROUTINE_HANDLER(RNNForwardTraining);
CUDNN_ROUTINE_HANDLER(RNNBackwardData);
CUDNN_ROUTINE_HANDLER(RNNBackwardWeights);
CUDNN_ROUTINE_HANDLER(SetRNNPaddingMode);
CUDNN_ROUTINE_HANDLER(GetRNNPaddingMode);
CUDNN_ROUTINE_HANDLER(CreateRNNDataDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyRNNDataDescriptor);
CUDNN_ROUTINE_HANDLER(SetRNNDataDescriptor);
CUDNN_ROUTINE_HANDLER(GetRNNDataDescriptor);
CUDNN_ROUTINE_HANDLER(RNNForwardTrainingEx);
CUDNN_ROUTINE_HANDLER(RNNForwardInferenceEx);
CUDNN_ROUTINE_HANDLER(RNNBackwardDataEx);
CUDNN_ROUTINE_HANDLER(RNNBackwardWeightsEx);
CUDNN_ROUTINE_HANDLER(SetRNNAlgorithmDescriptor);
CUDNN_ROUTINE_HANDLER(GetRNNForwardInferenceAlgorithmMaxCount);
CUDNN_ROUTINE_HANDLER(FindRNNForwardInferenceAlgorithmEx);
CUDNN_ROUTINE_HANDLER(GetRNNForwardTrainingAlgorithmMaxCount);
CUDNN_ROUTINE_HANDLER(FindRNNForwardTrainingAlgorithmEx);
CUDNN_ROUTINE_HANDLER(GetRNNBackwardDataAlgorithmMaxCount);
CUDNN_ROUTINE_HANDLER(FindRNNBackwardDataAlgorithmEx);
CUDNN_ROUTINE_HANDLER(GetRNNBackwardWeightsAlgorithmMaxCount);
CUDNN_ROUTINE_HANDLER(FindRNNBackwardWeightsAlgorithmEx);
CUDNN_ROUTINE_HANDLER(CreateSeqDataDescriptor);
CUDNN_ROUTINE_HANDLER(DestroySeqDataDescriptor);
CUDNN_ROUTINE_HANDLER(SetSeqDataDescriptor);
CUDNN_ROUTINE_HANDLER(GetSeqDataDescriptor);
CUDNN_ROUTINE_HANDLER(CreateAttnDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyAttnDescriptor);
CUDNN_ROUTINE_HANDLER(SetAttnDescriptor);
CUDNN_ROUTINE_HANDLER(GetAttnDescriptor);
CUDNN_ROUTINE_HANDLER(GetMultiHeadAttnBuffers);
CUDNN_ROUTINE_HANDLER(GetMultiHeadAttnWeights);
CUDNN_ROUTINE_HANDLER(MultiHeadAttnForward);
CUDNN_ROUTINE_HANDLER(MultiHeadAttnBackwardData);
CUDNN_ROUTINE_HANDLER(MultiHeadAttnBackwardWeights);
CUDNN_ROUTINE_HANDLER(CreateCTCLossDescriptor);
CUDNN_ROUTINE_HANDLER(SetCTCLossDescriptor);
CUDNN_ROUTINE_HANDLER(SetCTCLossDescriptorEx);
CUDNN_ROUTINE_HANDLER(GetCTCLossDescriptor);
CUDNN_ROUTINE_HANDLER(GetCTCLossDescriptorEx);
CUDNN_ROUTINE_HANDLER(DestroyCTCLossDescriptor);
CUDNN_ROUTINE_HANDLER(CTCLoss);
CUDNN_ROUTINE_HANDLER(GetCTCLossWorkspaceSize);
CUDNN_ROUTINE_HANDLER(CreateAlgorithmDescriptor);
CUDNN_ROUTINE_HANDLER(SetAlgorithmDescriptor);
CUDNN_ROUTINE_HANDLER(GetAlgorithmDescriptor);
CUDNN_ROUTINE_HANDLER(CopyAlgorithmDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyAlgorithmDescriptor);
CUDNN_ROUTINE_HANDLER(CreateAlgorithmPerformance);
CUDNN_ROUTINE_HANDLER(SetAlgorithmPerformance);
CUDNN_ROUTINE_HANDLER(GetAlgorithmPerformance);
CUDNN_ROUTINE_HANDLER(DestroyAlgorithmPerformance);
CUDNN_ROUTINE_HANDLER(GetAlgorithmSpaceSize);
CUDNN_ROUTINE_HANDLER(SaveAlgorithm);
CUDNN_ROUTINE_HANDLER(RestoreAlgorithm);
CUDNN_ROUTINE_HANDLER(SetCallback);
CUDNN_ROUTINE_HANDLER(GetCallback);
CUDNN_ROUTINE_HANDLER(CreateFusedOpsConstParamPack);
CUDNN_ROUTINE_HANDLER(DestroyFusedOpsConstParamPack);
CUDNN_ROUTINE_HANDLER(SetFusedOpsConstParamPackAttribute);
CUDNN_ROUTINE_HANDLER(GetFusedOpsConstParamPackAttribute);
CUDNN_ROUTINE_HANDLER(CreateFusedOpsVariantParamPack);
CUDNN_ROUTINE_HANDLER(DestroyFusedOpsVariantParamPack);
CUDNN_ROUTINE_HANDLER(SetFusedOpsVariantParamPackAttribute);
CUDNN_ROUTINE_HANDLER(GetFusedOpsVariantParamPackAttribute);
CUDNN_ROUTINE_HANDLER(CreateFusedOpsPlan);
CUDNN_ROUTINE_HANDLER(DestroyFusedOpsPlan);
CUDNN_ROUTINE_HANDLER(MakeFusedOpsPlan);
CUDNN_ROUTINE_HANDLER(FusedOpsExecute);


#endif  /* CUDNNHANDLER_H */
