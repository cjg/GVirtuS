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
 *
*/

#include <cstring>
#include <map>
#include <errno.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include "CudnnHandler.h"

using namespace std;
using namespace log4cplus;

std::map<string, CudnnHandler::CudnnRoutineHandler> * CudnnHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CudnnHandler> create_t() {
    return std::make_shared<CudnnHandler>();
}


extern "C" int HandlerInit() {
    return 0;
}

CudnnHandler::CudnnHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CudnnHandler"));
    setLogLevel(&logger);
    Initialize();
}

CudnnHandler::~CudnnHandler() {

}

void CudnnHandler::setLogLevel(Logger *logger) {
	log4cplus::LogLevel logLevel=log4cplus::INFO_LOG_LEVEL;
	char * val = getenv("GVIRTUS_LOGLEVEL");
	std::string logLevelString =(val == NULL ? std::string("") : std::string(val));
	if(logLevelString != "") {
		logLevel=std::stoi(logLevelString);
	}
	logger->setLogLevel(logLevel);
}

bool CudnnHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();

}

std::shared_ptr<Result> CudnnHandler::Execute(std::string routine, std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CudnnHandler::CudnnRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    try {
        return it->second(this, input_buffer);
    } catch (const char *ex) {
        LOG4CPLUS_DEBUG(logger,ex);
        LOG4CPLUS_DEBUG(logger,strerror(errno));
    }
    return NULL;
}

void CudnnHandler::Initialize(){
   if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudnnHandler::CudnnRoutineHandler> ();

    /* CublasHandler Query Platform Info */
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetErrorString));   
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Destroy));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetStream)); 
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensorNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorSizeInBytes));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(InitTransformDest));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformTensorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFoldedConvBackwardDataDescriptors));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(AddTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(OpTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReductionIndicesSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReductionWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ReduceTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ScaleTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor));
    #if CUDNN_VERSION < 6000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v4));
    #endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor));
    #if CUDNN_VERSION < 6000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v4));
    #endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterSizeInBytes));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformFilter));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ReorderFilterAndBias));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateConvolutionDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionGroupCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionGroupCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionReorderType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionReorderType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionNdForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyConvolutionDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionForwardAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionForwardAlgorithmEx));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithm));
#endif
#if CUDNN_VERSION >= 7000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithm_v7));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBiasActivationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardBias));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardFilterAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardFilterAlgorithmEx));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithm));
#endif
#if CUDNN_VERSION >= 7000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithm_v7));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardFilter));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardDataAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardDataAlgorithmEx));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithm));
#endif
#if CUDNN_VERSION >= 7000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithm_v7));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Im2Col));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SoftmaxForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SoftmaxBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreatePoolingDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPooling2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPooling2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPoolingNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPoolingNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPoolingNdForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPooling2dForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyPoolingDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(PoolingForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(PoolingBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ActivationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ActivationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(LRNCrossChannelForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(LRNCrossChannelBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DivisiveNormalizationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DivisiveNormalizationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DeriveBNTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationForwardTrainingExWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationBackwardExWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationTrainingExReserveSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardTraining));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardTrainingEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardInference));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationBackwardEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateSpatialTransformerDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetSpatialTransformerNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroySpatialTransformerDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfGridGeneratorForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfGridGeneratorBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfSamplerForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfSamplerBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutGetStatesSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutGetReserveSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RestoreDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetDropoutDescriptor)); 
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateRNNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyRNNDescriptor));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v5));
    //mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDescriptor_v5));
#endif

#if CUDNN_VERSION >= 6000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v6));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDescriptor_v6));
#endif
#if CUDNN_VERSION >= 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v8));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDescriptor_v8));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNMatrixMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNMatrixMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNBiasMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBiasMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNSetClip));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNGetClip));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNProjectionLayers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNProjectionLayers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreatePersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyPersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNTrainingReserveSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNParamsSize));   
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNLinLayerMatrixParams));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNLinLayerBiasParams));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardInference));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardTraining));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNPaddingMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNPaddingMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardTrainingEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardInferenceEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardDataEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardWeightsEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNForwardInferenceAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNForwardInferenceAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNForwardTrainingAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNForwardTrainingAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBackwardDataAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNBackwardDataAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBackwardWeightsAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNBackwardWeightsAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroySeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetMultiHeadAttnBuffers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetMultiHeadAttnWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnBackwardWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCTCLossDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CTCLoss));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CopyAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SaveAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RestoreAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCallback));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCallback));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsConstParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsConstParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFusedOpsConstParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFusedOpsConstParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsVariantParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsVariantParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFusedOpsVariantParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFusedOpsVariantParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MakeFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FusedOpsExecute));
}



CUDNN_ROUTINE_HANDLER(GetConvolutionMathType){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionMathType"));

     cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
     cudnnMathType_t mathType;

     cudnnStatus_t cs = cudnnGetConvolutionMathType(convDesc, &mathType);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
     try{
         out->Add<cudnnMathType_t>(mathType);
     } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
     }
     LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionMathType Executed");
     //cout << " DEBUG - cudnnGetConvolutionMathType Executed"<<endl;
     return std::make_shared<Result>(cs, out);
 }

CUDNN_ROUTINE_HANDLER(SetConvolutionReorderType){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionReorderType"));

     cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
     cudnnReorderType_t reorderType = in->Get<cudnnReorderType_t>();

     cudnnStatus_t cs = cudnnSetConvolutionReorderType(convDesc, reorderType);

     LOG4CPLUS_DEBUG(logger, "cudnnSetConvolutionReorderType Executed");
     //cout << " DEBUG - cudnnSetConvolutionReorderType Executed"<<endl;
     return std::make_shared<Result>(cs); 
 }

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardFilterAlgorithm"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t DyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;
    
    cudnnStatus_t cs = cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, DyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardFilterAlgorithm Executed");
    //cout << " DEBUG - cudnnFindConvolutionBackwardFilterAlgorithm Executed"<<endl;
    return std::make_shared<Result>(cs, out);  
 }


CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithmMaxCount){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithmMaxCount"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     int count;

     cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &count);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
     try{
         out->Add<int>(count);
     } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
     }
     LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionForwardAlgorithmMaxCount Executed");
     //cout << " DEBUG - cudnnGetConvolutionForwardAlgorithmMaxCount Executed"<<endl;
     return std::make_shared<Result>(cs, out);
 }



CUDNN_ROUTINE_HANDLER(SetConvolutionNdDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionNdDescriptor"));

     cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
     int arrayLength = in->Get<int>();
     int *padA = in->Assign<int>();
     int *filterStrideA = in->Assign<int>();
     int *dilationA = in->Assign<int>();
     cudnnConvolutionMode_t mode = in->Get<cudnnConvolutionMode_t>();
     cudnnDataType_t computeType = in->Get<cudnnDataType_t>();

     cudnnStatus_t cs = cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);

     LOG4CPLUS_DEBUG(logger, "cudnnSetConvolutionNdDescriptor Executed");   
     //cout << " DEBUG - cudnnSetConvolutionNdDescriptor Executed"<<endl;
     return std::make_shared<Result>(cs);
 }



CUDNN_ROUTINE_HANDLER(GetConvolutionNdForwardOutputDim){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionNdForwardOutputDim"));

     cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
     cudnnTensorDescriptor_t inputTensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
     int nbDims = in->Get<int>();
     int *tensorOutputDimA = in->Assign<int>();

     cudnnStatus_t cs = cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOutputDimA);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
     try{
         out->Add<int>(tensorOutputDimA);
     } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
     }
     LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionNdForwardOutputDim Executed");
     //cout << " DEBUG - cudnnGetConvolutionNdForwardOutputDim Executed"<<endl;
     return std::make_shared<Result>(cs, out);
 }

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithmEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardFilterAlgorithmEx"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>(); //INPUT
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
    void *x = in->Assign<void>(); //INPUT
    cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
    void *y = in->Assign<void>(); //INPUT
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>(); //INPUT
    cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>(); //INPUT
    void *dw = in->Assign<void>(); //INPUT/OUTPUT
    int requestedAlgoCount = in->Get<int>(); //INPUT
    int *returnedAlgoCount; //OUTPUT
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults; //OUTPUT
    void *workSpace = in->Assign<void>(); //INPUT
    size_t workSpaceSizeInBytes = in->Get<size_t>(); //INPUT
   
    cudnnStatus_t cs = cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, &perfResults, workSpace, workSpaceSizeInBytes);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<void>(dw);
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults); 
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardFilterAlgorithmEx Executed");
    return std::make_shared<Result>(cs, out);
}



CUDNN_ROUTINE_HANDLER(GetConvolution2dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dDescriptor"));

     cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
     int padh,padw,u,v,upscalex,upscaley;
     cudnnConvolutionMode_t mode;
     cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

     cudnnStatus_t cs = cudnnGetConvolution2dDescriptor(convDesc,&padh,&padw,&u,&v,&upscalex,&upscaley,&mode,&computeType);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

     try{
         out->Add(padh);
         out->Add(padw);
         out->Add(u);
         out->Add(v);
         out->Add(upscalex);
         out->Add(upscaley);
     } catch(string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
     }
     LOG4CPLUS_DEBUG(logger, "cudnnGetConvolution2dDescriptor Executed");
     //cout << "DEBUG - cudnnGetConvolution2dDescriptor Executed"<<endl;
     return std::make_shared<Result>(cs,out);
 }




CUDNN_ROUTINE_HANDLER(SetConvolutionGroupCount){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionGroupCount"));

     cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
     int groupCount = in->Get<int>();

     cudnnStatus_t cs = cudnnSetConvolutionGroupCount(convDesc, groupCount);

     LOG4CPLUS_DEBUG(logger, "cudnnSetConvolutionGroupCount Executed");   
     //cout << " DEBUG - cudnnSetConvolutionGroupCount Executed"<<endl;
     return std::make_shared<Result>(cs);
 }

CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithmEx){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionForwardAlgorithmEx"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *x = in->GetFromMarshal<void *>();
     cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
     void *w = in->GetFromMarshal<void *>();
     cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
     cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *y = in->GetFromMarshal<void *>();
     int requestedAlgoCount = in->Get<int>();
     int returnedAlgoCount;
     cudnnConvolutionFwdAlgoPerf_t perfResults;
     void *workSpace = in->GetFromMarshal<void *>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();

     cudnnStatus_t cs = cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, &returnedAlgoCount, &perfResults, workSpace, workSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->AddMarshal<void *>(y);
         out->Add<int>(returnedAlgoCount);
         out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnFindConvolutionForwardAlgorithmEx Executed");
    return std::make_shared<Result>(cs,out);  
}

CUDNN_ROUTINE_HANDLER(GetConvolutionNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionNdDescriptor"));
    
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int arrayLengthRequested = in->Get<int>();
    int arrayLength;
    int *padA = in->Assign<int>();
    int *filterStrideA = in->Assign<int>();
    int *dilationA = in->Assign<int>();
    cudnnConvolutionMode_t mode;
    cudnnDataType_t dataType;

    cudnnStatus_t cs = cudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested, &arrayLength,padA, filterStrideA, dilationA, &mode, &dataType);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<cudnnConvolutionDescriptor_t>(convDesc);
         out->Add<int>(arrayLength);
         out->Add<int>(padA);
         out->Add<int>(filterStrideA);
         out->Add<int>(dilationA);
         out->Add<cudnnConvolutionMode_t>(mode);
         out->Add<cudnnDataType_t>(dataType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionNdDescriptor Executed");
    return std::make_shared<Result>(cs,out);   
}

#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
  cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  cudnnConvolutionFwdPreference_t preference = (cudnnConvolutionFwdPreference_t)in->Get<long long int>();
    size_t memoryLimitInBytes = (size_t)in->Get<int>();

    cudnnConvolutionFwdAlgo_t algo;



  cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, &algo);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<cudnnConvolutionFwdAlgo_t>(algo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);
}
#endif

#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm_v7){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm_v7"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
  cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  int requestedAlgoCount = in->Get<int>();
  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults;

  cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<int>(returnedAlgoCount);
         out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardAlgorithm_v7  Executed");
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(GetConvolutionReorderType){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionReorderType"));
 
  cudnnConvolutionDescriptor_t convDesc= (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  cudnnReorderType_t reorderType;

  cudnnStatus_t cs = cudnnGetConvolutionReorderType(convDesc, &reorderType);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<cudnnReorderType_t>(reorderType);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionReorderType  Executed");
    return std::make_shared<Result>(cs,out); 
}

CUDNN_ROUTINE_HANDLER(SetConvolutionMathType){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionMathType"));

   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnMathType_t mathType = in->Get<cudnnMathType_t>();

   cudnnStatus_t cs = cudnnSetConvolutionMathType(convDesc, mathType);

    LOG4CPLUS_DEBUG(logger,"cudnnSetConvolutionMathType  Executed");
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ConvolutionBiasActivationForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBiasActivationForward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   void *alpha1 = in->Assign<void>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->GetFromMarshal<void *>();
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   void *w = in->GetFromMarshal<void *>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
   void *workSpace = in->GetFromMarshal<void *>();
   size_t workSpaceSizeInBytes = in->Get<size_t>();
   void *alpha2 = in->Assign<void>();
   cudnnTensorDescriptor_t zDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *z = in->GetFromMarshal<void *>();
   cudnnTensorDescriptor_t biasDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *bias = in->GetFromMarshal<void *>();
   cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->AddMarshal<void *>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnConvolutionBiasActivationForward  Executed");
    return std::make_shared<Result>(cs,out);  
}

CUDNN_ROUTINE_HANDLER(onvolutionBiasActivationForward){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBiasActivationForward"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  void *alpha1 = in->Assign<void>();
  cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  void *x = in->GetFromMarshal<void *>();
  cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
  void *w = in->GetFromMarshal<void *>();
  cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
  void *workSpace = in->GetFromMarshal<void *>();
  size_t workSpaceSizeInBytes = in->Get<size_t>();
  void *alpha2 = in->Assign<void>();
  cudnnTensorDescriptor_t zDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  void *z = in->GetFromMarshal<void *>();
  cudnnTensorDescriptor_t biasDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  void *bias = in->GetFromMarshal<void *>();
  cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
  cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  void *y = in->GetFromMarshal<void *>();

  cudnnStatus_t cs = cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);

 std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->AddMarshal<void *>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"onvolutionBiasActivationForward  Executed");
    return std::make_shared<Result>(cs,out);  
}

CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithmMaxCount){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithmMaxCount"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   int count;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &count);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<int>(count);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"GetConvolutionBackwardFilterAlgorithmMaxCount  Executed");
    return std::make_shared<Result>(cs,out);
}

#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithm"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;

    cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<int>(returnedAlgoCount);
         out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"GetConvolutionBackwardFilterAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);
}
#endif

#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm_v7){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithm_v7"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;
    
    cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);
  
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<int>(returnedAlgoCount);
         out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"GetConvolutionBackwardFilterAlgorithm_v7  Executed");
    return std::make_shared<Result>(cs,out);   
}
#endif

CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithm){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionForwardAlgorithm"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
  cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  int requestedAlgoCount = in->Get<int>();
  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults;

  cudnnStatus_t cs = cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<int>(returnedAlgoCount);
         out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"FindConvolutionForwardAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionGroupCount){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionGroupCount"));

   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   int groupCount;

   cudnnStatus_t cs = cudnnGetConvolutionGroupCount(convDesc, &groupCount);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<int>(groupCount);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionGroupCount  Executed");
    return std::make_shared<Result>(cs,out); 
}


CUDNN_ROUTINE_HANDLER(DestroyConvolutionDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardBias"));
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnDestroyConvolutionDescriptor(convDesc);

    //LOG4CPLUS_DEBUG(logger,"cudnnDestroyConvolutionDescriptor  Executed");
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ConvolutionBackwardBias){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardBias"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    float alpha = in->Get<float>();
    //const void *alpha = in->Assign<void>();
    const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    const void *dy = in->GetFromMarshal<void *>();
    float beta = in->Get<float>();
    //const void *beta = in->Assign<void>();
    const cudnnTensorDescriptor_t dbDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *db = in->GetFromMarshal<void *>(); 

    cudnnStatus_t cs = cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy, &beta,  dbDesc, db);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->AddMarshal<void *>(db);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnConvolutionBackwardBias  Executed");
    return std::make_shared<Result>(cs,out);
}


CUDNN_ROUTINE_HANDLER(ConvolutionForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionForward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   //const void *alpha = in->GetFromMarshal<void *>();

   float alpha = in->Get<float>();	


   const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const void *x = in->GetFromMarshal<void *>();
   const cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   const void *w = in->GetFromMarshal<void *>();
   const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
   void *workSpace = in->GetFromMarshal<void *>();
   size_t workSpaceSizeInBytes = in->Get<size_t>();
   float beta =in->Get<float>();
   const cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->GetFromMarshal<void *>();


   cudnnStatus_t cs = cudnnConvolutionForward(handle, &alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, &beta, yDesc, y);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->AddMarshal<void *>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnConvolutionForward  Executed");
    return std::make_shared<Result>(cs,out);  
}


CUDNN_ROUTINE_HANDLER(ConvolutionBackwardFilter){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardFilter"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  //const void *alpha = in->Assign<void>();
  float alpha = in->Get<float>();
  const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  const void *x = in->GetFromMarshal<void *>();
  const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  const void *dy = in->GetFromMarshal<void *>();
  const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  cudnnConvolutionBwdFilterAlgo_t algo = in->Get<cudnnConvolutionBwdFilterAlgo_t>();
  void *workSpace = in->GetFromMarshal<void *>();
  size_t workSpaceSizeInBytes = in->Get<size_t>();
  float beta = in->Get<float>();
  //const void *beta = in->Assign<void>();
  const cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
  void *dw = in->GetFromMarshal<void *>();

  cudnnStatus_t cs = cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, &beta,dwDesc, dw);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->AddMarshal<void *>(dw);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnConvolutionBackwardFilter  Executed");
    return std::make_shared<Result>(cs,out);  
}


CUDNN_ROUTINE_HANDLER(GetConvolution2dForwardOutputDim){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dForwardOutputDim"));
   
   const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t inputTensor = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const cudnnFilterDescriptor_t filterDesc  = (cudnnFilterDescriptor_t)in->Get<long long int>();
   int n;
   int c;
   int h;
   int w;

   cudnnStatus_t cs = cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n ,&c, &h, &w);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<int>(n);
         out->Add<int>(c);
         out->Add<int>(h);
         out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolution2dForwardOutputDim  Executed");
    return std::make_shared<Result>(cs,out);  
}


CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterWorkspaceSize){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterWorkspaceSize"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  const cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
  cudnnConvolutionBwdFilterAlgo_t algo = in->Get<cudnnConvolutionBwdFilterAlgo_t>();
  size_t sizeInBytes;

  cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, dwDesc, algo, &sizeInBytes);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<size_t>(sizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionBackwardFilterWorkspaceSize  Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(CreateConvolutionDescriptor){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateConvolutionDescriptor"));

  cudnnConvolutionDescriptor_t convDesc;

  cudnnStatus_t cs = cudnnCreateConvolutionDescriptor(&convDesc);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnCreateConvolutionDescriptor  Executed");
    return std::make_shared<Result>(cs,out); 
}

#if CUDNN_VERSION < 8204
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithm"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   const cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnConvolutionBwdFilterPreference_t preference = in->Get<cudnnConvolutionBwdFilterPreference_t>();
   size_t memoryLimitInBytes = in->Get<size_t>();
   cudnnConvolutionBwdFilterAlgo_t algo;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, &algo);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnConvolutionBwdFilterAlgo_t>(algo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionBackwardFilterAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);   
}
#endif
 
CUDNN_ROUTINE_HANDLER(SetConvolution2dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolution2dDescriptor"));

    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int pad_h = in->Get<int>();
    int pad_w = in->Get<int>();
    int u = in->Get<int>();
    int v = in->Get<int>();
    int dilation_h = in->Get<int>();
    int dilation_w = in->Get<int>();
    cudnnConvolutionMode_t mode = in->Get<cudnnConvolutionMode_t>();
    cudnnDataType_t computeType = in->Get<cudnnDataType_t>();
   
    cudnnStatus_t cs = cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnSetConvolution2dDescriptor  Executed");
    return std::make_shared<Result>(cs,out);
}

#if CUDNN_VERSION < 8204
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionFwdPreference_t preference = in->Get<cudnnConvolutionFwdPreference_t>();
   size_t memoryLimitInBytes = in->Get<size_t>();
   cudnnConvolutionFwdAlgo_t algo;  
 
   cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, &algo);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnConvolutionFwdAlgo_t>(algo);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(GetConvolutionForwardWorkspaceSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardWorkspaceSize"));
   
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    const cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    const cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
    size_t sizeInBytes;

    cudnnStatus_t cs = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &sizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<size_t>(sizeInBytes);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardWorkspaceSize  Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetVersion){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));

    size_t version = cudnnGetVersion();
    LOG4CPLUS_DEBUG(logger,"cudnnGetVersion Executed");
    return std::make_shared<Result>(version);
}

CUDNN_ROUTINE_HANDLER(GetErrorString){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    cudnnStatus_t cs = in->Get<cudnnStatus_t>();
    const char * s = cudnnGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add((char *)s);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetErrorString Executed");
    return std::make_shared<Result>(CUDNN_STATUS_SUCCESS,out);
}

CUDNN_ROUTINE_HANDLER(Create){

    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    cudnnHandle_t handle;
    cudnnStatus_t cs = cudnnCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<cudnnHandle_t>(handle);
    } catch (string e){
                        LOG4CPLUS_DEBUG(logger,e);
                        return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnCreate Executed");
    return std::make_shared<Result>(cs,out);

}

CUDNN_ROUTINE_HANDLER(Destroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Destroy"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroy(handle);
    
    //LOG4CPLUS_DEBUG(logger,"cudnnDestroy Executed");
    //cout << "DEBUG - cudnnDestroy Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudaStream_t streamId = (cudaStream_t) in->Get<long long int>();

    cudnnStatus_t cs = cudnnSetStream(handle,streamId);
    
     LOG4CPLUS_DEBUG(logger," cudnnSetStream Executed");
   //cout << "DEBUG - cudnnSetStream Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetStream"));
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudaStream_t *streamId;
    cudnnStatus_t cs = cudnnGetStream(handle,streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<long long int>((long long int)*streamId);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetStream Executed");
    //cout << "DEBUG - cudnnGetStream Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorDescriptor"));
    cudnnTensorDescriptor_t tensorDesc;
    cudnnStatus_t cs = cudnnCreateTensorDescriptor(&tensorDesc);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnCreateTensorDescriptor Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();                                                                                          
    int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    //printf("[BACKEND SetTensor4dDescriptor] N, C, H, W: %d %d %d %d\n", n, c, h, w);

    cudnnStatus_t cs = cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }                      
    //LOG4CPLUS_DEBUG(logger,"cudnnSetTensor4dDescriptor Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptorEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();

    int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    int nStride = in->Get<int>();
    int cStride = in->Get<int>();
    int hStride = in->Get<int>();
    int wStride = in->Get<int>();

    cudnnStatus_t cs = cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetTensor4dDescriptor Executed");
    //cout << "DEBUG - cudnnSetTensor4dDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();

    cudnnDataType_t dataType;
    int n,c,h,w;
    int nStride,cStride,hStride,wStride;

    cudnnStatus_t cs = cudnnGetTensor4dDescriptor(tensorDesc,&dataType,&n,&c,&h,&w,&nStride,&cStride,&hStride,&wStride);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(n);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
        out->Add<int>(nStride);
        out->Add<int>(cStride);
        out->Add<int>(hStride);
        out->Add<int>(wStride);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetTensor4dDescriptor Executed");
    //cout << "DEBUG - cudnnGetTensor4dDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();
    int *strideA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetTensorNdDescriptor Executed");
    //cout << "DEBUG - cudnnSetTensorNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptorEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptorEx"));

    cudnnTensorDescriptor_t tensorDesc;
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetTensorNdDescriptorEx Executed");
    //cout << "DEBUG - cudnnSetTensorNdDescriptorEx Executed"<<endl;
    return std::make_shared<Result>(cs);  
}

CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorNdDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t dataType;
    int *nbDims;
    int *dimA;
    int *strideA = in->Assign<int>();

    cudnnStatus_t cs = cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,&dataType,nbDims,dimA,strideA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(nbDims);
        out->Add<int>(dimA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetTensorNdDescriptor Executed");
    //cout << "DEBUG - cudnnGetTensorNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetTensorSizeInBytes){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorSizeInBytes"));

   cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   size_t size = in->Get<size_t>();

   cudnnStatus_t cs = cudnnGetTensorSizeInBytes(tensorDesc, &size);
  
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->Add<size_t>(size);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetTensorSizeInBytes Executed");
   //cout << "DEBUG - cudnnGetTensorSizeInBytes Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyTensorDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroyTensorDescriptor(tensorDesc);
    
    //LOG4CPLUS_DEBUG(logger, "DestroyTensorDescriptor Executed");
    //cout << "DEBUG - DestroyTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(InitTransformDest){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("InitTransformDest"));
    
    cudnnTensorTransformDescriptor_t transformDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int >();
    cudnnTensorDescriptor_t srcDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t destDesc;
    size_t destSizeInBytes;
  
    cudnnStatus_t cs = cudnnInitTransformDest(transformDesc, srcDesc, destDesc, &destSizeInBytes);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnTensorDescriptor_t>(destDesc);
        out->Add<size_t>(destSizeInBytes);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnInitTransformDest Executed");
    //cout << " DEBUG - cudnnInitTransformDest Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorTransformDescriptor"));

    cudnnTensorTransformDescriptor_t transformDesc;
    
    cudnnStatus_t cs = cudnnCreateTensorTransformDescriptor(&transformDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnTensorTransformDescriptor_t>(transformDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, " cudnnCreateTensorTransformDescriptor Execute");
    //cout << " DEBUG - cudnnCreateTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorTransformDescriptor"));
    
    cudnnTensorTransformDescriptor_t transformDesc;
    uint32_t nbDims = in->Get<uint32_t>();
    cudnnTensorFormat_t destFormat = in->Get<cudnnTensorFormat_t>();
    int32_t *padBeforeA = in->Assign<int32_t>();
    int32_t *padAfterA = in->Assign<int32_t>();
    uint32_t *foldA = in->Assign<uint32_t>();
    cudnnFoldingDirection_t direction = in->Get<cudnnFoldingDirection_t>();

    cudnnStatus_t cs = cudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction);
     
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnTensorTransformDescriptor_t>(transformDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger,  "cudnnSetTensorTransformDescriptor Executed");
    //cout << "DEBUG - cudnnSetTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorTransformDescriptor"));

    cudnnTensorTransformDescriptor_t transformDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int >();
    uint32_t nbDimsRequested = in->Get<uint32_t>();
    cudnnTensorFormat_t destFormat;
    int32_t padBeforeA;
    int32_t padAfterA;
    uint32_t foldA;
    cudnnFoldingDirection_t direction;
   
    cudnnStatus_t cs = cudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested, &destFormat, &padBeforeA, &padAfterA, &foldA, &direction);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnTensorFormat_t>(destFormat);
        out->Add<int32_t>(padBeforeA);
        out->Add<int32_t>(padAfterA);
        out->Add<uint32_t>(foldA);
        out->Add<cudnnFoldingDirection_t>(direction);   
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetTensorTransformDescriptor Executed"); 
    //cout << "DEBUG - cudnnGetTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyTensorTransformDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyTensorTransformDescriptor"));
    
    cudnnTensorTransformDescriptor_t transformDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();
    
    cudnnStatus_t cs = cudnnDestroyTensorTransformDescriptor(transformDesc);
    
    //LOG4CPLUS_DEBUG(logger, " cudnnDestroyTensorTransformDescriptor Execute");
    //cout << " DEBUG - cudnnDestroyTensorTransformDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    void * alpha = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * x = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();

    cudnnStatus_t cs = cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<void>(y);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnTransformTensor Executed");
    //cout << "DEBUG - cudnnTransformTensor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformTensorEx){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformTensorEx"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnTensorTransformDescriptor_t transDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>();
   void *alpha = in->Assign<void>();
   cudnnTensorDescriptor_t srcDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *srcData = in->Assign<void>();
   void *beta = in->Assign<void>();
   cudnnTensorDescriptor_t destDesc = (cudnnTensorDescriptor_t)in->Get<cudnnTensorDescriptor_t>();
   void *destData = in->Assign<void>();
  
   cudnnStatus_t cs = cudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
   
   
   LOG4CPLUS_DEBUG(logger, "cuddTransformTensorEx Executed");
   //cout << "DEBUG - cuddTransformTensorEx Executed"<<endl;
   return std::make_shared<Result>(cs);
}

// NON SONO SICURO DI QUESTA FUNZIONE DA FAR VEDERE A MONTELLA!!! 
CUDNN_ROUTINE_HANDLER(GetFoldedConvBackwardDataDescriptors){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFoldedConvBackwardDataDescriptors"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnFilterDescriptor_t filterDesc;
   cudnnTensorDescriptor_t diffDesc;
   cudnnConvolutionDescriptor_t convDesc;
   cudnnTensorDescriptor_t gradDesc;
   cudnnTensorFormat_t transformFormat;
   cudnnFilterDescriptor_t foldedFilterDesc;
   cudnnTensorDescriptor_t paddedDiffDesc;
   cudnnConvolutionDescriptor_t foldedConvDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t foldedGradDesc;
   cudnnTensorTransformDescriptor_t filterFoldTransDesc;
   cudnnTensorTransformDescriptor_t diffPadTransDesc;
   cudnnTensorTransformDescriptor_t gradFoldTransDesc;
   cudnnTensorTransformDescriptor_t gradUnfoldTransDesc;


   cudnnStatus_t cs = cudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnFilterDescriptor_t>(filterDesc);
        out->Add<cudnnTensorDescriptor_t>(diffDesc);
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
        out->Add<cudnnTensorDescriptor_t>(gradDesc);
        out->Add<cudnnTensorFormat_t>(transformFormat);
        out->Add<cudnnFilterDescriptor_t>(foldedFilterDesc);
        out->Add<cudnnTensorDescriptor_t>(paddedDiffDesc);
        out->Add<cudnnConvolutionDescriptor_t>(foldedConvDesc);
        out->Add<cudnnTensorDescriptor_t>(foldedGradDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(filterFoldTransDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(diffPadTransDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(gradFoldTransDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(gradUnfoldTransDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFoldedConvBackwardDataDescriptors Executed");
    //cout << " DEBUG - cudnnGetFoldedConvBackwardDataDescriptors Executed"<<endl;
    return std::make_shared<Result>(cs, out);
    
}

CUDNN_ROUTINE_HANDLER(AddTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("AddTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    float alpha = in->Get<float>();
    const cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    const void * A = in->GetFromMarshal<void *>();
    float beta = in->Get<float>();
    const cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * C = in->GetFromMarshal<void *>();

    cudnnStatus_t cs = cudnnAddTensor(handle,&alpha,aDesc,A,&beta,cDesc,C);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->AddMarshal<void *>(C);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger, "cudnnAddTensor Executed");
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateOpTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateOpTensorDescriptor"));
   
    cudnnOpTensorDescriptor_t opTensorDesc;
    
    cudnnStatus_t cs = cudnnCreateOpTensorDescriptor(&opTensorDesc);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnOpTensorDescriptor_t>(opTensorDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateOpTensorDescriptor Executed");
    //cout << " DEBUG - cudnnCreateOpTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetOpTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetOpTensorDescriptor"));

    cudnnOpTensorDescriptor_t opTensorDesc;
    cudnnOpTensorOp_t opTensorOp = in->Get<cudnnOpTensorOp_t>();
    cudnnDataType_t opTensorCompType = in->Get<cudnnDataType_t>();
    cudnnNanPropagation_t opTensorNanOpt = in->Get<cudnnNanPropagation_t>();

   cudnnStatus_t cs = cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnOpTensorDescriptor_t>(opTensorDesc);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   

   LOG4CPLUS_DEBUG(logger, "cudnnSetOpTensorDescriptor Executed");
   //cout << " DEBUG - cudnnSetOpTensorDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetOpTensorDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetOpTensorDescriptor"));
   
   cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
   cudnnOpTensorOp_t opTensorOp;
   cudnnDataType_t opTensorCompType;
   cudnnNanPropagation_t opTensorNanOpt;

   cudnnStatus_t cs = cudnnGetOpTensorDescriptor(opTensorDesc, &opTensorOp, &opTensorCompType, &opTensorNanOpt);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnOpTensorOp_t>(opTensorOp);
       out->Add<cudnnDataType_t>(opTensorCompType);
       out->Add<cudnnNanPropagation_t>(opTensorNanOpt);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetOpTensorDescriptor");
   //cout << " DEBUG - cudnnGetOpTensorDescriptor"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyOpTensorDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyOpTensorDescriptor"));

   cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
   
   cudnnStatus_t cs = cudnnDestroyOpTensorDescriptor(opTensorDesc);
   
   LOG4CPLUS_DEBUG(logger, "cudnnDestroyOpTensorDescriptor Executed");
   //cout << "DEBUG - cudnnDestroyOpTensorDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(OpTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("OpTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
    const void * alpha1 = in->Assign<void>();
    cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * A = in->Assign<void>();
    void * alpha2 = in->Assign<void>();
    cudnnTensorDescriptor_t bDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * B = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * C = in->Assign<void>();

    cudnnStatus_t cs = cudnnOpTensor(handle,opTensorDesc,alpha1,aDesc,A,alpha2,bDesc,B,beta,cDesc,C);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<void>(C);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
   


    LOG4CPLUS_DEBUG(logger, "cudnnOpTensor Executed");
    //cout << "DEBUG - cudnnOpTensor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc;
   
    cudnnStatus_t cs = cudnnCreateReduceTensorDescriptor(& reduceTensorDesc);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnReduceTensorDescriptor_t>(reduceTensorDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateReduceTensorDescriptor Executed");
    //cout << " DEBUG - cudnnCreateReduceTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}
 
CUDNN_ROUTINE_HANDLER(SetReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
    cudnnReduceTensorOp_t reduceTensorOp = in->Get<cudnnReduceTensorOp_t>();
    cudnnDataType_t reduceTensorCompType = in->Get<cudnnDataType_t>();
    cudnnNanPropagation_t reduceTensorNanOpt = in->Get<cudnnNanPropagation_t>();
    cudnnReduceTensorIndices_t reduceTensorIndices = in->Get<cudnnReduceTensorIndices_t>();
    cudnnIndicesType_t reduceTensorIndicesType = in->Get<cudnnIndicesType_t>();

    cudnnStatus_t cs = cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnReduceTensorDescriptor_t>(reduceTensorDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetReduceTensorDescriptor");
    //cout << " DEBUG - cudnnSetReduceTensorDescriptor"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReduceTensorDescriptor"));
 
    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>(); //INPUT
    cudnnReduceTensorOp_t reduceTensorOp; //OUTPUT
    cudnnDataType_t reduceTensorCompType; //OUTPUT
    cudnnNanPropagation_t reduceTensorNanOpt = in->Get<cudnnNanPropagation_t>(); //INPUT
    cudnnReduceTensorIndices_t reduceTensorIndices; //OUTPUT
    cudnnIndicesType_t reduceTensorIndicesType; //OUTPUT
  
    cudnnStatus_t cs = cudnnGetReduceTensorDescriptor(reduceTensorDesc, &reduceTensorOp, &reduceTensorCompType, &reduceTensorNanOpt, &reduceTensorIndices, &reduceTensorIndicesType);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnReduceTensorOp_t>(reduceTensorOp);
        out->Add<cudnnDataType_t>(reduceTensorCompType);
        out->Add<cudnnReduceTensorIndices_t>(reduceTensorIndices);
        out->Add<cudnnIndicesType_t>(reduceTensorIndicesType);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
 
    LOG4CPLUS_DEBUG(logger, "cudnnGetReduceTensorDescriptor Executed");   
    //cout << " DEBUG - cudnnGetReduceTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyReduceTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroyReduceTensorDescriptor Executed");
    //cout << "DEBUG - cudnnDestroyReduceTensorDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetReductionIndicesSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReductionIndicesSize"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    size_t *sizeInBytes;
   
    cudnnStatus_t cs = cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
     
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<size_t>(sizeInBytes);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   
    LOG4CPLUS_DEBUG(logger, "cuddGetReductionIndicesSize Executed");
   //cout << " DEBUG - cuddGetReductionIndicesSize Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetReductionWorkspaceSize){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReductionWorkspaceSize"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   size_t *sizeInBytes;

   cudnnStatus_t cs = cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
    
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<size_t>(sizeInBytes);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
    LOG4CPLUS_DEBUG(logger, "cudnnGetReductionWorkspaceSize Executed");
   //cout << " DEBUG - cudnnGetReductionWorkspaceSize Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ReduceTensor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ReduceTensor"));
 
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>(); //INPUT
   cudnnReduceTensorDescriptor_t reduceTensorDesc = (cudnnReduceTensorDescriptor_t)in->Get<long long int>(); //INPUT
   void *indices = in->Assign<void>();  //OUTPUT
   size_t indicesSizeInBytes = in->Get<size_t>(); //INPUT
   void *workspace = in->Assign<void>(); //INPUT
   size_t workspaceSizeInBytes = in->Get<size_t>(); //INPUT
   void *alpha = in->Assign<void>(); //INPUT
   cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
   void *A = in->Assign<void>(); //INPUT
   void *beta = in->Assign<void>(); //INPUT
   cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
   void *C = in->Assign<void>(); //INPUT/OUTPUT

   cudnnStatus_t cs = cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha,  aDesc, A, beta, cDesc, C);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(indices);
       out->Add<void>(C);
   } catch(string e){
      LOG4CPLUS_DEBUG(logger, e);
      return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnReduceTensor Execute");
   //cout << " DEBUG - cudnnReduceTensor Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();
    void * valuePtr = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetTensor(handle,yDesc,y,valuePtr);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<void>(y);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
    
     LOG4CPLUS_DEBUG(logger, "cudnnSetTensor Executed");
    //cout << "DEBUG - cudnnSetTensor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ScaleTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ScaleTensor"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    const cudnnTensorDescriptor_t yDesc = (const cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();
    void * alpha = in->Assign<void>();

    cudnnStatus_t cs = cudnnScaleTensor(handle,yDesc,y,alpha);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<void>(y);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
    
     LOG4CPLUS_DEBUG(logger, "cudnnScaleTensor Executed");
    //cout << "DEBUG - cudnnScaleTensor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFilterDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFilterDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc;

    cudnnStatus_t cs = cudnnCreateFilterDescriptor(&filterDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnFilterDescriptor_t>(filterDesc);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger,"cudnnCreateFilterDescriptor Executed");
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor"));
   
   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
   cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
   int k = in->Get<int>();
   int c = in->Get<int>();
   int h = in->Get<int>();
   int w = in->Get<int>();

   cudnnStatus_t cs = cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnFilterDescriptor_t>(filterDesc);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }   
   //LOG4CPLUS_DEBUG(logger,"cudnnSetFilter4dDescriptor Executed");
   return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor"));

   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnDataType_t dataType;
   cudnnTensorFormat_t format;
   int k;
   int c;
   int h;
   int w;

   cudnnStatus_t cs = cudnnGetFilter4dDescriptor(filterDesc, &dataType, &format, &k, &c, &h, &w);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnDataType_t>(dataType);
       out->Add<cudnnTensorFormat_t>(format);
       out->Add<int>(k);
       out->Add<int>(c);
       out->Add<int>(h);
       out->Add<int>(w);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetFilter4dDescriptor Executed");
   //cout << " DEBUG - cudnnGetFilter4dDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

#if CUDNN_VERSION < 6000
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v3"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();

    int k = in->Get<int>();
etConvolution2dDescriptor
    int h = in->Get<int>();
    int w = in->Get<int>();

    cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v3(filterDesc,dataType,k,c,h,w);
   
     LOG4CPLUS_DEBUG(logger, "cudnnSetFilter4dDescriptor_v3 Executed");
     //cout << "DEBUG - cudnnSetFilter4dDescriptor_v3 Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor_v3"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType;

    int k,c,h,w;

    cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v3(filterDesc,&dataType,&k,&c,&h,&w);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>((long long int)dataType);
        out->Add<int>(k);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilter4dDescriptor_v3 Executed");
    //cout << "DEBUG - cudnnGetFilter4dDescriptor_v3 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v4"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();

    int k = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetFilter4dDescriptor_v4 Executed");
    //cout << "DEBUG - cudnnSetFilter4dDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor_v4"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType;
    cudnnTensorFormat_t  format;

    int k,c,h,w;

    cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v4(filterDesc,&dataType,&format,&k,&c,&h,&w);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>((long long int)dataType);
        out->Add<long long int>((long long int)format);
        out->Add<int>(k);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilter4dDescriptor_v4 Executed");
    //cout << "DEBUG - cudnnGetFilter4dDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    int nbDims = in->Get<int>();
    int *filterDimA = in->Assign<int>();
    
    cudnnStatus_t cs = cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetFilterNdDescriptor Executed");
    //cout << " DEBUG - cudnnSetFilterNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));

    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();

    cudnnTensorFormat_t  format;

    cudnnStatus_t cs = cudnnGetFilterNdDescriptor(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);

    std:shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>(format);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilterNdDescriptor Executed");
    //cout << "DEBUG - cudnnGetFilterNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

#if CUDNN_VERSION < 6000
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v3"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();

    int nbDims = in->Get<int>();
    int * filterDimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v3(filterDesc,dataType,nbDims,filterDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<long long int>((long long int)filterDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetFilterNdDescriptor_v3 Executed");
    //cout << "DEBUG - cudnnSetFilterNdDescriptor_v3 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));

    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();


    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v3(wDesc,nbDimsRequested,dataType,nbDims,filterDimA);
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilterNdDescriptor Executed");
    //cout << "DEBUG - cudnnGetFilterNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v4"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();

    int nbDims = in->Get<int>();
    int * filterDimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v4(filterDesc,dataType,format,nbDims,filterDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<long long int>((long long int)filterDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetFilterDescriptor_v4 Executed");
    //cout << "DEBUG - cudnnSetFilterDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor_v4"));

    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();

    cudnnTensorFormat_t  format;

    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v4(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<long long int>(format);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilterDescriptor_v4 Executed");
    //cout << "DEBUG - cudnnGetFilterDescriptor_v4 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(GetFilterSizeInBytes){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterSizeInBytes"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();

    size_t size = in->Get<size_t>();
    
    cudnnStatus_t cs = cudnnGetFilterSizeInBytes(filterDesc, &size);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<size_t>(size);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger,  "cudnnGetFilterSizeInBytes Executed");
    //cout << " DEBUG - cudnnGetFilterSizeInBytes Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestoryFilterDescriptor"));

    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnDestroyFilterDescriptor(filterDesc);
    
    //LOG4CPLUS_DEBUG(logger,  "cudnnDestroyFilterDescriptor Executed");
    //cout << "DEBUG - cudnnDestroyFilterDescriptor Executed"<<endl;
    return make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformFilter){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformFilter"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>(); //INPUT
   cudnnTensorTransformDescriptor_t transDesc = (cudnnTensorTransformDescriptor_t)in->Get<long long int>(); //INPUT
   void *alpha = in->Assign<void>(); //INPUT
   cudnnFilterDescriptor_t srcDesc = (cudnnFilterDescriptor_t)in->Get<long long int>(); //INPUT
   void *srcData = in->Assign<void>(); //INPUT
   void *beta = in->Assign<void>(); //INPUT
   cudnnFilterDescriptor_t destDesc = (cudnnFilterDescriptor_t)in->Get<long long int>(); //INPUT
   void *destData = in->Assign<void>(); //OUTPUT
   
   cudnnStatus_t cs = cudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(destData);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
    LOG4CPLUS_DEBUG(logger, "cudnnTransformFilter Executed");
   //cout << " DEBUG - cudnnTransformFilter Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ReorderFilterAndBias){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ReorderFilterAndBias"));
   
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnReorderType_t reorderType = in->Get<cudnnReorderType_t>();
   void *filterData = in->Assign<void>();
   void *reorderedFilterData = in->Assign<void>();
   int reorderBias = in->Get<int>();
   void *biasData =  in->GetFromMarshal<void *>();
   void *reorderedBiasData = in->GetFromMarshal<void *>();
  
   cudnnStatus_t cs = cudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData);

   
  LOG4CPLUS_DEBUG(logger, "cudnnReorderFilterAndBias Executed");
  //cout << " DEBUG - cudnnReorderFilterAndBias Executed"<<endl;
  return std::make_shared<Result>(cs);
}  



CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithmMaxCount){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetonvolutionBackwardData"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  int count;

  cudnnStatus_t cs = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &count);
  
  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
  try{
      out->Add<int>(count);
  }  catch(string e){
      LOG4CPLUS_DEBUG(logger, e);
      return std::make_shared<Result>(cs);
  }
  
   //LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount Executed");
  //cout << " DEBUG - cudnnGetConvolutionBackwardDataAlgorithmMaxCount Executed"<<endl;
  return std::make_shared<Result>(cs, out);
}  

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardDataAlgorithm){
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardDataAlgorithm"));

  cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
  cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
  cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
  cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
  int requestedAlgoCount = in->Get<int>();
  int returnedAlgoCount;
  cudnnConvolutionBwdDataAlgoPerf_t perfResults;

  cudnnStatus_t cs = cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
  try{
      out->Add<int>(returnedAlgoCount);
      out->Add<cudnnConvolutionBwdDataAlgoPerf_t>(perfResults);
  } catch(string e){
      LOG4CPLUS_DEBUG(logger, e);
      return std::make_shared<Result>(cs);
  }
 
  LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardDataAlgorithm Executed");
  //cout << " DEBUG - cudnnFindConvolutionBackwardDataAlgorithm Executed"<<endl;
  return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardDataAlgorithmEx){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardDataAlgorithmEx"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>(); //INPUT
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>(); //INPUT
   void *w = in->Assign<void>(); //INPUT
   cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
   void *dy = in->Assign<void>(); //INPUT
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>(); //INPUT
   cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //INPUT
   void *dx = in->Assign<void>(); //INPUT/OUTPUT
   int requestedAlgoCount = in->Get<int>(); //INPUT
   int returnedAlgoCount; //OUTPUT
   cudnnConvolutionBwdDataAlgoPerf_t perfResults; //OUTPUT
   void *workSpace = in->Assign<void>(); //INPUT
   size_t workSpaceSizeInBytes = in->Get<size_t>(); //INPUT

   cudnnStatus_t cs = cudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, &returnedAlgoCount, &perfResults, workSpace, workSpaceSizeInBytes);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(dx);
       out->Add<int>(returnedAlgoCount);
       out->Add<cudnnConvolutionBwdDataAlgoPerf_t>(perfResults);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardDataAlgorithmEx Executed");
   //cout << " DEBUG - cudnnFindConvolutionBackwardDataAlgorithmEx Executed"<<endl;
   return std::make_shared<Result>(cs, out);  
}

#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithm){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardDataAlgorithm"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   const cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionBwdDataPreference_t preference = in->Get<cudnnConvolutionBwdDataPreference_t>();
   size_t memoryLimitInBytes = in->Get<size_t>();
   cudnnConvolutionBwdDataAlgo_t algo;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, &algo);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnConvolutionBwdDataAlgo_t>(algo);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataAlgorithm Executed");
   return std::make_shared<Result>(cs, out);  
}
#endif

#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithm_v7){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardDataAlgorithm_v7"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t diffDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t gradDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   int requestedAlgoCount = in->Get<int>();
   int returnedAlgoCount;
   cudnnConvolutionBwdDataAlgoPerf_t perfResults;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<int>(returnedAlgoCount);
       out->Add<cudnnConvolutionBwdDataAlgoPerf_t>(perfResults);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataAlgorithm_v7 Executed");
   //cout << " DEBUG - cudnnGetConvolutionBackwardDataAlgorithm_v7 Executed"<<endl;
   return std::make_shared<Result>(cs, out);   
}
#endif

CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataWorkspaceSize){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardDataWorkspaceSize"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   const cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   const cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnConvolutionBwdDataAlgo_t algo = in->Get<cudnnConvolutionBwdDataAlgo_t>();
   size_t sizeInBytes;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, &sizeInBytes);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<size_t>(sizeInBytes);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataWorkspaceSize Executed");
   return std::make_shared<Result>(cs, out);             
}

CUDNN_ROUTINE_HANDLER(ConvolutionBackwardData){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardData"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   float alpha = in->Get<float>();
   //const void *alpha = in->Assign<void>();
   const cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   const void *w = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const void *dy = in->GetFromMarshal<void *>();
   const cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   cudnnConvolutionBwdDataAlgo_t algo = in->Get<cudnnConvolutionBwdDataAlgo_t>();
   void *workSpace = in->GetFromMarshal<void *>();
   size_t workSpaceSizeInBytes = in->Get<size_t>();
   float beta = in->Get<float>();
   //const void *beta = in->Assign<void>();
   const cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *dx = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnConvolutionBackwardData(handle, &alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, &beta, dxDesc, dx);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->AddMarshal<void *>(dx);
   } catch(string e){
      LOG4CPLUS_DEBUG(logger, e);
      return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnConvolutionBackwardData Executed");
   return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(Im2Col){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Im2Col"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->Assign<void>();
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
   void *colBuffer;

   cudnnStatus_t cs = cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->Add<void>(colBuffer);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnIm2Col Executed");
   //cout << " DEBUG - cudnnIm2Col Executed"<<endl;
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(SoftmaxForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SoftmaxForward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnSoftmaxAlgorithm_t algo = in->Get<cudnnSoftmaxAlgorithm_t>();
   cudnnSoftmaxMode_t mode = in->Get<cudnnSoftmaxMode_t>();
   //const void *alpha = in->Assign<void>();
   float alpha = in->Get<float>();
   const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const void *x = in->GetFromMarshal<void *>();
   float beta = in->Get<float>();
   //const void *beta = in->Assign<void>();
   const cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnSoftmaxForward(handle, algo, mode, &alpha, xDesc, x, &beta, yDesc, y);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->AddMarshal<void *>(y);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   
   //LOG4CPLUS_DEBUG(logger, "cudnnSoftmaxForward Executed");
   //cout << " DEBUG - cudnnSoftmaxForward Executed"<<endl;
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(SoftmaxBackward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SoftmaxBackward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnSoftmaxAlgorithm_t algo = in->Get<cudnnSoftmaxAlgorithm_t>();
   cudnnSoftmaxMode_t mode = in->Get<cudnnSoftmaxMode_t>();
   void *alpha = in->Assign<void>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->Assign<void>();
   cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *dy = in->Assign<void>();
   void *beta = in->Assign<void>();
   cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *dx = in->Assign<void>();

   cudnnStatus_t cs = cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
 
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<void>(dx);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnSoftmaxBackward Executed");
   //cout << " DEBUG - cudnnSoftmaxBackward Executed"<<endl;
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(CreatePoolingDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreatePoolingDescriptor"));

   cudnnPoolingDescriptor_t poolingDesc;

   cudnnStatus_t cs = cudnnCreatePoolingDescriptor(&poolingDesc);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->Add<cudnnPoolingDescriptor_t>(poolingDesc);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnCreatePoolingDescriptor Executed");
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetPooling2dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPooling2dDescriptor"));

   cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
   cudnnPoolingMode_t mode = in->Get<cudnnPoolingMode_t>();
   cudnnNanPropagation_t maxpoolingNanOpt = in->Get<cudnnNanPropagation_t>();
   int windowHeight = in->Get<int>();
   int windowWidth = in->Get<int>();
   int verticalPadding = in->Get<int>();
   int horizontalPadding = in->Get<int>();
   int verticalStride = in->Get<int>();
   int horizontalStride = in->Get<int>();

   cudnnStatus_t cs = cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->Add<cudnnPoolingDescriptor_t>(poolingDesc);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnSetPooling2dDescriptor Executed");
   return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetPooling2dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPooling2dDescriptor"));

   cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
   cudnnPoolingMode_t mode;
   cudnnNanPropagation_t maxpoolingNanOpt;
   int windowHeight;
   int windowWidth;
   int verticalPadding;
   int horizontalPadding;
   int verticalStride;
   int horizontalStride;

   cudnnStatus_t cs = cudnnGetPooling2dDescriptor(poolingDesc, &mode, &maxpoolingNanOpt, &windowHeight, &windowWidth, &verticalPadding, &horizontalPadding, &verticalStride, &horizontalStride);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnPoolingMode_t>(mode);
       out->Add<cudnnNanPropagation_t>(maxpoolingNanOpt);
       out->Add<int>(windowHeight);
       out->Add<int>(windowWidth);
       out->Add<int>(verticalPadding);
       out->Add<int>(horizontalPadding);
       out->Add<int>(verticalStride);
       out->Add<int>(horizontalStride);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
    LOG4CPLUS_DEBUG(logger, "cudnnGetPooling2dDescriptor Executed");
   //cout << " DEBUG - cudnnGetPooling2dDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(SetPoolingNdDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPoolingNdDescriptor"));
   
   cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
   cudnnPoolingMode_t mode = in->Get<cudnnPoolingMode_t>();
   cudnnNanPropagation_t maxpoolingNanOpt = in->Get<cudnnNanPropagation_t>();
   int nbDims = in->Get<int>();
   int *windowDimA = in->Assign<int>();
   int *paddingA = in->Assign<int>();
   int *strideA = in->Assign<int>();

   cudnnStatus_t cs = cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

   
   LOG4CPLUS_DEBUG(logger, "cudnnSetPoolingNdDescriptor Executed");
   //cout << " DEBUG - cudnnSetPoolingNdDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetPoolingNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPoolingNdDescriptor"));

    cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt = in->Get<cudnnNanPropagation_t>();
    int nbDims;
    int *windowDimA = in->Assign<int>();
    int *paddingA = in->Assign<int>();
    int *strideA = in->Assign<int>();

    cudnnStatus_t cs = cudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, &mode, &maxpoolingNanOpt, &nbDims, windowDimA, paddingA, strideA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnPoolingMode_t>(mode);
        out->Add<int>(nbDims);
        out->Add<int>(windowDimA);
        out->Add<int>(paddingA);
        out->Add<int>(strideA);
    } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetPoolingNdDescriptor Executed");
    //cout << " DEBUG - cudnnGetPoolingNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(GetPoolingNdForwardOutputDim){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPoolingNdForwardOutputDim"));
    
    cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t inputTensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    int nbDims = in->Get<int>();
    int *outputTensorDimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(outputTensorDimA);
    }catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetPoolingNdForwardOutputDim Executed");
    //cout << " DEBUG - cudnnGetPoolingNdForwardOutputDim Executed"<<endl;
    return std::make_shared<Result>(cs, out);           
}

CUDNN_ROUTINE_HANDLER(GetPooling2dForwardOutputDim){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPooling2dForwardOutputDim"));

   cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t inputTensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   int n;
   int c;
   int h;
   int w;
 
   cudnnStatus_t cs = cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, &n, &c, &h, &w);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<int>(n);
       out->Add<int>(c);
       out->Add<int>(h);
       out->Add<int>(w);
   } catch(string e){  
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetPooling2dForwardOutputDim Executed");
   //cout << " DEBUG - cudnnGetPooling2dForwardOutputDim Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyPoolingDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyPoolingDescriptor"));
   
   cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();

   cudnnStatus_t cs = cudnnDestroyPoolingDescriptor(poolingDesc);

   //LOG4CPLUS_DEBUG(logger, "cudnnDestroyPoolingDescriptor Executed");
   //cout << " DEBUG - cudnnDestroyPoolingDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);   
}

CUDNN_ROUTINE_HANDLER(PoolingForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("PoolingForward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   const cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
   //const void *alpha = in->Assign<void>();
   float alpha = in->Get<float>();
   const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const void *x = in->GetFromMarshal<void *>();
   //const void *beta = in->Assign<void>();
   float beta = in->Get<float>();
   const cudnnTensorDescriptor_t yDesc  = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnPoolingForward(handle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->AddMarshal<void *>(y);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }  
   //LOG4CPLUS_DEBUG(logger, "cudnnPoolingForward Executed");
   return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(PoolingBackward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("PoolingBackward"));   

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   const cudnnPoolingDescriptor_t poolingDesc = (cudnnPoolingDescriptor_t)in->Get<long long int>();
   //const void *alpha = in->Assign<void>();
   float alpha = in->Get<float>();
   const cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const void *y = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const void *dy = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   const void *x = in->GetFromMarshal<void *>();
   float beta = in->Get<float>();
   //const void *beta = in->Assign<void>();
   const cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *dx = in->GetFromMarshal<void *>();


   cudnnStatus_t cs = cudnnPoolingBackward(handle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->AddMarshal<void *>(dx);  
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnPoolingBackward Executed");
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateActivationDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateActivationDescriptor"));

   cudnnActivationDescriptor_t activationDesc;
 
   cudnnStatus_t cs = cudnnCreateActivationDescriptor(&activationDesc);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
       out->Add<cudnnActivationDescriptor_t>(activationDesc);
   } catch(string e){
       LOG4CPLUS_DEBUG(logger, e);
       return std::make_shared<Result>(cs);
   }
    //LOG4CPLUS_DEBUG(logger, "cudnnCreateActivationDescriptor Executed");
   return std::make_shared<Result>(cs, out);

}

CUDNN_ROUTINE_HANDLER(SetActivationDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetActivationDescriptor"));

   cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
   cudnnActivationMode_t mode = in->Get<cudnnActivationMode_t>();
   cudnnNanPropagation_t reluNanOpt = in->Get<cudnnNanPropagation_t>();
   double coef = in->Get<double>();

   cudnnStatus_t cs = cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->Add<cudnnActivationDescriptor_t>(activationDesc);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnSetActivationDescriptor Executed");
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetActivationDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetActivationDescriptor"));

   cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
   cudnnActivationMode_t mode;
   cudnnNanPropagation_t reluNanOpt;
   double coef;

   cudnnStatus_t cs = cudnnGetActivationDescriptor(activationDesc, &mode, &reluNanOpt, &coef);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try{
        out->Add<cudnnActivationMode_t>(mode);
        out->Add<cudnnNanPropagation_t>(reluNanOpt);
        out->Add<double>(coef);
   } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetActivationDescriptor Executed");
   //cout << " DEBUG - cudnnGetActivationDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyActivationDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyActivationDescriptor"));
   
   cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
   
   cudnnStatus_t cs = cudnnDestroyActivationDescriptor(activationDesc);

   //LOG4CPLUS_DEBUG(logger, "cudnnDestroyActivationDescriptor Executed");
   //cout << " DEBUG - cudnnDestroyActivationDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ActivationForward){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ActivationForward"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
    //const void *alpha = in->Assign<void>();
    float alpha = in->Get<float>();
    const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    const void *x = in->GetFromMarshal<void *>();
    //const void *beta = in->Assign<void>();
    float beta = in->Get<float>();
    const cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *y = in->GetFromMarshal<void *>();
    cudnnStatus_t cs = cudnnActivationForward(handle, activationDesc, &alpha, xDesc, x, &beta, yDesc, y);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->AddMarshal<void *>(y);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger, "cudnnActivationForward Executed");
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ActivationBackward) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ActivationBackward"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
     //const void *alpha = in->Assign<void>();
     float alpha = in->Get<float>();
     const cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     const void *y = in->GetFromMarshal<void *>();
     const cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     const void *dy = in->GetFromMarshal<void *>();
     const cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     const void *x = in->GetFromMarshal<void *>();
     float beta = in->Get<float>();
     //const void *beta = in->Assign<void>();
     cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dx = in->GetFromMarshal<void *>();

     cudnnStatus_t cs = cudnnActivationBackward(handle, activationDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
     try{
         out->AddMarshal<void *>(dx);
     } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
     }
     //LOG4CPLUS_DEBUG(logger, "cudnnActivationBackward Executed"); 
     //cout << " DEBUG - cudnnActivationBackward Executed"<<endl;
     return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateLRNDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateLRNDescriptor"));

     cudnnLRNDescriptor_t normDesc;
     
     cudnnStatus_t cs = cudnnCreateLRNDescriptor(&normDesc);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
     try{
          out->Add<cudnnLRNDescriptor_t>(normDesc);
     } catch(string e){
          LOG4CPLUS_DEBUG(logger, e);
          return std::make_shared<Result>(cs);
     }
     
     LOG4CPLUS_DEBUG(logger, "cudnnCreateLRNDescriptor Executed");
     //cout << " DEBUG - cudnnCreateLRNDescriptor Executed"<<endl;
     return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetLRNDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetLRNDescriptor"));

    cudnnLRNDescriptor_t normDesc = (cudnnLRNDescriptor_t)in->Get<long long int>();
    unsigned lrnN = in->Get<unsigned>();
    double lrnAlpha = in->Get<double>();
    double lrnBeta = in->Get<double>();
    double lrnK = in->Get<double>();

    cudnnStatus_t cs = cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetLRNDescriptor Executed");
    //cout << " DEBUG - cudnnSetLRNDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetLRNDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetLRNDescriptor"));

    cudnnLRNDescriptor_t normDesc = (cudnnLRNDescriptor_t)in->Get<long long int>();
    unsigned lrnN;
    double lrnAlpha;
    double lrnBeta;
    double lrnK;

    cudnnStatus_t cs = cudnnGetLRNDescriptor(normDesc, &lrnN, &lrnAlpha, &lrnBeta, &lrnK);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<unsigned>(lrnN);
         out->Add<double>(lrnAlpha);  
         out->Add<double>(lrnBeta);
         out->Add<double>(lrnK);
    }catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetLRNDescriptor Executed");
    //cout << " DEBUG - cudnnGetLRNDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyLRNDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyLRNDescriptor"));

    cudnnLRNDescriptor_t lrnDesc = (cudnnLRNDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnDestroyLRNDescriptor(lrnDesc);

    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroyLRNDescriptor Executed");
    //cout << " DEBUG - cudnnDestroyLRNDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(LRNCrossChannelForward){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("LRNCrossChannelForward"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnLRNDescriptor_t normDesc = (cudnnLRNDescriptor_t)in->Get<long long int>();
    cudnnLRNMode_t lrnMode = in->Get<cudnnLRNMode_t>();
    void *alpha = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    void *beta = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();

    cudnnStatus_t cs = cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<void>(y);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
 
    LOG4CPLUS_DEBUG(logger, "cudnnLRNCrossChannelForward Executed");   
    //cout << " DEBUG - cudnnLRNCrossChannelForward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(LRNCrossChannelBackward){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("LRNCrossChannelBackward"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnLRNDescriptor_t normDesc = (cudnnLRNDescriptor_t)in->Get<long long int>();
    cudnnLRNMode_t lrnMode = in->Get<cudnnLRNMode_t>();
    void *alpha = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();
    cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dy = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    void *beta = in->Assign<void>();
    cudnnTensorDescriptor_t dxDesc;
    void *dx = in->Assign<void>();

    cudnnStatus_t cs = cudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
         out->Add<cudnnTensorDescriptor_t>(dxDesc);
         out->Add<void>(dx);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnLRNCrossChannelBackward Executed");
    //cout << " DEBUG - cudnnLRNCrossChannelBackward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DivisiveNormalizationForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DivisiveNormalizationForward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnLRNDescriptor_t normDesc = (cudnnLRNDescriptor_t)in->Get<long long int>();
   cudnnDivNormMode_t mode = in->Get<cudnnDivNormMode_t>();
   void *alpha = in->Assign<void>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->Assign<void>();
   void *means = in->Assign<void>();
   void *temp = in->Assign<void>();
   void *temp2 = in->Assign<void>();
   void *beta = in->Assign<void>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->Assign<void>();

   cudnnStatus_t cs = cudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
         out->Add<void>(y);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnDivisiveNormalizationForward Executed");
    //cout << " DEBUG - cudnnDivisiveNormalizationForward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DivisiveNormalizationBackward){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DivisiveNormalizationBackward"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnLRNDescriptor_t normDesc = (cudnnLRNDescriptor_t)in->Get<long long int>();
    cudnnDivNormMode_t mode = in->Get<cudnnDivNormMode_t>();
    void *alpha = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    void *means = in->Assign<void>();
    void *dy = in->Assign<void>();
    void *temp = in->Assign<void>();
    void *temp2 = in->Assign<void>();
    void *beta = in->Assign<void>();
    cudnnTensorDescriptor_t dXdMeansDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dx = in->Assign<void>();
    void *dMeans = in->Assign<void>();

    cudnnStatus_t cs = cudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans);
     
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
         out->Add<void>(dx);
         out->Add<void>(dMeans);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
      
    LOG4CPLUS_DEBUG(logger, "cudnnDivisiveNormalizationBackward Executed");
    //cout << " DEBUG - cudnnDivisiveNormalizationBackward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DeriveBNTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeriveBNTensorDescriptor"));

    cudnnTensorDescriptor_t derivedBnDesc;
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();

    cudnnStatus_t cs = cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
  
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
         out->Add<cudnnTensorDescriptor_t>(derivedBnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnDeriveBNTensorDescriptor");
    //cout << " DEBUG - cudnnDeriveBNTensorDescriptor"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(GetBatchNormalizationForwardTrainingExWorkspaceSize){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetBatchNormalizationForwardTrainingExWorkspaceSize"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
   cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t zDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
   size_t sizeInBytes;

   cudnnStatus_t cs = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, &sizeInBytes);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
         out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnDeriveBNTensorDescriptor");
    //cout << " DEBUG - cudnnDeriveBNTensorDescriptor"<<endl;
    return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetBatchNormalizationBackwardExWorkspaceSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetBatchNormalizationBackwardExWorkspaceSize"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t dzDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t dBnScaleBiasDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
    size_t sizeInBytes;

    cudnnStatus_t cs = cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, &sizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
         out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
   
    LOG4CPLUS_DEBUG(logger, "cudnnDeriveBNTensorDescriptor");
    //cout << " DEBUG - cudnnDeriveBNTensorDescriptor"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetBatchNormalizationTrainingExReserveSpaceSize){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetBatchNormalizationTrainingExReserveSpaceSize"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
     cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
     cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     size_t sizeInBytes;

     cudnnStatus_t cs = cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps, activationDesc, xDesc, &sizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
         out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
    //cout << " DEBUG - cudnnGetBatchNormalizationTrainingExReserveSpaceSize"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardTraining){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationForwardTraining"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    void *alpha = in->Assign<void>();
    void *beta = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *bnScale = in->Assign<void>();
    void *bnBias = in->Assign<void>();
    double exponentialAverageFactor = in->Get<double>();
    
    void *resultRunningMean = in->Assign<void>(); //INPUT/OUTPUT
    void *resultRunningVariance = in->Assign<void>(); //INPUT/OUTPUT    
    double epsilon = in->Get<double>();
    void *resultSaveMean = in->Assign<void>(); //OUTPUT
    void *resultSaveInvVariance = in->Assign<void>(); //OUTPUT

    cudnnStatus_t cs = cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(resultRunningMean);
          out->Add<void>(resultRunningVariance);
          out->Add<void>(resultSaveMean);
          out->Add<void>(resultSaveInvVariance);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
    //cout << " DEBUG - cudnnGetBatchNormalizationTrainingExReserveSpaceSize"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardTrainingEx){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationForwardTrainingEx"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
     cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
     void *alpha = in->Assign<void>();
     void *beta = in->Assign<void>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *xData = in->Assign<void>();
     cudnnTensorDescriptor_t zDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *zData = in->Assign<void>();
     cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *yData; //OUTPUT
     cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *bnScale = in->Assign<void>();
     void *bnBias = in->Assign<void>();
     double exponentialAverageFactor = in->Get<double>();
     void *resultRunningMean = in->Assign<void>(); //INPUT/OUTPUT
     void *resultRunningVariance = in->Assign<void>(); //INPUT/OUTPUT
     double epsilon = in->Get<double>();
     void *resultSaveMean = in->Assign<void>(); //OUTPUT
     void *resultSaveInvVariance = in->Assign<void>(); //OUTPUT
     cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
     void *workspace = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();
     void *reserveSpace = in->Assign<void>();
     size_t reserveSpaceSizeInBytes = in->Get<size_t>();

     cudnnStatus_t cs = cudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
  
      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
	  out->Add<void>(yData);
          out->Add<void>(resultRunningMean);
          out->Add<void>(resultRunningVariance);
          out->Add<void>(resultSaveMean);
          out->Add<void>(resultSaveInvVariance);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationForwardTrainingEx");
    //cout << " DEBUG - cudnnBatchNormalizationForwardTrainingEx"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardInference){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationForwardInference"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    void *alpha = in->Assign<void>();
    void *beta = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *bnScale = in->Assign<void>();
    void *bnBias = in->Assign<void>();
    void *estimatedMean = in->Assign<void>();
    void *estimatedVariance = in->Assign<void>();
    double epsilon = in->Get<double>();

    cudnnStatus_t cs = cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
    
    
    LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationForwardInference Executed");
    //cout << " DEBUG - cudnnBatchNormalizationForwardInference Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationBackward){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationBackward"));
     
     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
     void *alphaDataDiff = in->Assign<void>();
     void *betaDataDiff  = in->Assign<void>();
     void *alphaParamDiff = in->Assign<void>();
     void *betaParamDiff  = in->Assign<void>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *x = in->Assign<void>();
     cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dy = in->Assign<void>();
     cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dx = in->Assign<void>();
     cudnnTensorDescriptor_t dBnScaleBiasDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *bnScale = in->Assign<void>();
     void *dBnScaleResult = in->Assign<void>(); //OUTPUT
     void *dBnBiasResult = in->Assign<void>(); //OUTPUT
     double epsilon = in->Get<double>();
     void *savedMean = in->Assign<void>();
     void *savedInvVariance = in->Assign<void>();

    cudnnStatus_t cs = cudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dBnScaleResult);
          out->Add<void>(dBnBiasResult);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    

    LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationBackward Executed");
    //cout << " DEBUG - cudnnBatchNormalizationBackward Executed"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationBackwardEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationBackwardEx"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
    void *alphaDataDiff = in->Assign<void>();
    void *betaDataDiff  = in->Assign<void>();
    void *alphaParamDiff = in->Assign<void>();
    void *betaParamDiff  = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *xData = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *yData = in->Assign<void>();
    cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dyData = in->Assign<void>();
    cudnnTensorDescriptor_t dzDesc; //OUTPUT
    void *dzData = in->Assign<void>(); //OUTPUT
    cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>(); //OUTPUT
    void *dxData = in->Assign<void>(); //OUTPUT  
    cudnnTensorDescriptor_t dBnScaleBiasDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *bnScaleData = in->Assign<void>();
    void *bnBiasData = in->Assign<void>();
    void *dBnScaleData = in->Assign<void>();
    void *dBnBiasData = in->Assign<void>();
    double epsilon = in->Get<double>();
    void *savedMean = in->Assign<void>();
    void *savedInvVariance = in->Assign<void>();
    cudnnActivationDescriptor_t activationDesc = (cudnnActivationDescriptor_t)in->Get<long long int>();
    void *workSpace = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>();
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();
 
     cudnnStatus_t cs = cudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnTensorDescriptor_t>(dzDesc);
          out->Add<void>(dzData);
          out->Add<cudnnTensorDescriptor_t>(dxDesc);
          out->Add<void>(dxData);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationBackwardEx Executed");
    //cout << " DEBUG - cudnnBatchNormalizationBackwardEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateSpatialTransformerDescriptor){
      Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateSpatialTransformerDescriptor"));

      cudnnSpatialTransformerDescriptor_t stDesc;
    
      cudnnStatus_t cs = cudnnCreateSpatialTransformerDescriptor(&stDesc);

       std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnSpatialTransformerDescriptor_t>(stDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateSpatialTransformerDescriptor Executed");
    //cout << " DEBUG - cudnnCreateSpatialTransformerDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetSpatialTransformerNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetSpatialTransformerNdDescriptor"));
    
    cudnnSpatialTransformerDescriptor_t stDesc = (cudnnSpatialTransformerDescriptor_t)in->Get<long long int>();
    cudnnSamplerType_t samplerType = in->Get<cudnnSamplerType_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();


    cudnnStatus_t cs = cudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType, nbDims, dimA);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnSpatialTransformerDescriptor_t>(stDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    

     LOG4CPLUS_DEBUG(logger, "cudnnSetSpatialTransformerNdDescriptor Executed");
     //cout << " DEBUG - cudnnSetSpatialTransformerNdDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(DestroySpatialTransformerDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroySpatialTransformerDescriptor"));

    cudnnSpatialTransformerDescriptor_t stDesc = (cudnnSpatialTransformerDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnDestroySpatialTransformerDescriptor(stDesc);

    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroySpatialTransformerDescriptor Executed");
    //cout << " DEBUG - cudnnDestroySpatialTransformerDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs); 
}

CUDNN_ROUTINE_HANDLER(SpatialTfGridGeneratorForward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfGridGeneratorForward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnSpatialTransformerDescriptor_t stDesc = (cudnnSpatialTransformerDescriptor_t)in->Get<long long int>();
   void *theta = in->Assign<void>();
   void *grid = in->Assign<void>();

   cudnnStatus_t cs = cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(grid);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfGridGeneratorForward Executed");
    //cout << " DEBUG - cudnnSpatialTfGridGeneratorForward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}



CUDNN_ROUTINE_HANDLER(SpatialTfGridGeneratorBackward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfGridGeneratorBackward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnSpatialTransformerDescriptor_t stDesc = (cudnnSpatialTransformerDescriptor_t)in->Get<long long int>();
   void *dgrid = in->Assign<void>();
   void *dtheta = in->Assign<void>();

   cudnnStatus_t cs = cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dtheta);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfGridGeneratorBackward Executed");
    //cout << " DEBUG - cudnnSpatialTfGridGeneratorBackward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SpatialTfSamplerForward){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfSamplerForward"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnSpatialTransformerDescriptor_t stDesc = (cudnnSpatialTransformerDescriptor_t)in->Get<long long int>();
    void *alpha = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    void *grid = in->Assign<void>();
    void *beta = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();

    cudnnStatus_t cs = cudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfSamplerForward Executed");
    //cout << " DEBUG - cudnnSpatialTfSamplerForward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SpatialTfSamplerBackward){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfSamplerBackward"));
  
     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnSpatialTransformerDescriptor_t stDesc = (cudnnSpatialTransformerDescriptor_t)in->Get<long long int>();
     void *alpha = in->Assign<void>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *x = in->Assign<void>();
     void *beta = in->Assign<void>();
     cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dx = in->Assign<void>(); //OUTPUT
     void *alphaDgrid = in->Assign<void>();
     cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dy = in->Assign<void>();
     void *grid = in->Assign<void>();
     void *betaDgrid = in->Assign<void>();
     void *dgrid = in->Assign<void>(); //OUTPUT

     cudnnStatus_t cs = cudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dx);
          out->Add<void>(dgrid);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfSamplerBackward Executed");
    //cout << " DEBUG - cudnnSpatialTfSamplerBackward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateDropoutDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateDropoutDescriptor"));

    cudnnDropoutDescriptor_t dropoutDesc;

    cudnnStatus_t cs = cudnnCreateDropoutDescriptor(&dropoutDesc);
    
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateDropoutDescriptor Executed");
    //cout << " DEBUG - cudnnCreateDropoutDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyDropoutDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyDropoutDescriptor"));

   cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
   
   cudnnStatus_t cs = cudnnDestroyDropoutDescriptor(dropoutDesc);

    LOG4CPLUS_DEBUG(logger, "cudnnDestroyDropoutDescriptor Executed");
    //cout << " DEBUG - cudnnDestroyDropoutDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
  
}

CUDNN_ROUTINE_HANDLER(DropoutGetStatesSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutGetStatesSize"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    size_t sizeInBytes;

    cudnnStatus_t cs = cudnnDropoutGetStatesSize(handle, &sizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnDropoutGetStatesSize Executed");
    //cout << " DEBUG - cudnnDropoutGetStatesSize Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(DropoutGetReserveSpaceSize){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutGetReserveSpaceSize"));

     cudnnTensorDescriptor_t xdesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     size_t sizeInBytes;
    
     cudnnStatus_t cs = cudnnDropoutGetReserveSpaceSize(xdesc, &sizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnDropoutGetReserveSpaceSize Executed");
    //cout << " DEBUG - cudnnDropoutGetReserveSpaceSize Executed"<<endl;
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(SetDropoutDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetDropoutDescriptor"));

   cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>(); //INPUT/OUTPUT
   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   float dropout = in->Get<float>();
   void *states = in->Assign<void>(); //OUTPUT
   size_t stateSizeInBytes = in->Get<size_t>();
   unsigned long long seed = in->Get<unsigned long long>();

   cudnnStatus_t cs = cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
          out->Add<void>(states);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetDropoutDescriptor Executed");
    //cout << " DEBUG - cudnnSetDropoutDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(RestoreDropoutDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RestoreDropoutDescriptor"));

    cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    float dropout = in->Get<float>();
    void *states = in->Assign<void>();
    size_t stateSizeInBytes = in->Get<size_t>();
    unsigned long long seed = in->Get<unsigned long long>();

    cudnnStatus_t cs = cudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnRestoreDropoutDescriptor Executed");
    //cout << " DEBUG - cudnnRestoreDropoutDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}


CUDNN_ROUTINE_HANDLER(GetDropoutDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetDropoutDescriptor"));

    cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    float dropout;
    void *states = in->Assign<void>();
    unsigned long long seed = in->Get<unsigned long long>();

    cudnnStatus_t cs = cudnnGetDropoutDescriptor(dropoutDesc, handle, &dropout, &states, &seed);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<float>(dropout);
          out->Add<void>(states);
          out->Add<unsigned long long>(seed);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetDropoutDescriptor Executed");
    //cout << " DEBUG - cudnnGetDropoutDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DropoutForward){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutForward"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
     cudnnTensorDescriptor_t xdesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *x = in->Assign<void>();
     cudnnTensorDescriptor_t ydesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *y = in->Assign<void>(); //OUTPUT
     void *reserveSpace = in->Assign<void>(); //OUTPUT
     size_t reserveSpaceSizeInBytes = in->Get<size_t>();

     cudnnStatus_t cs = cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnDropoutForward Executed");
    //cout << " DEBUG - cudnnDropoutForward Executed"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(DropoutBackward){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutBackward"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
   cudnnTensorDescriptor_t dydesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *dy = in->Assign<void>();
   cudnnTensorDescriptor_t dxdesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *dx = in->Assign<void>(); //OUTPUT
   void *reserveSpace = in->Assign<void>();
   size_t reserveSpaceSizeInBytes = in->Get<size_t>();

   cudnnStatus_t cs = cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dx);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnDropoutBackward Executed");
    //cout << " DEBUG - cudnnDropoutBackward Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateRNNDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateRNNDescriptor"));

   cudnnRNNDescriptor_t rnnDesc;

   cudnnStatus_t cs = cudnnCreateRNNDescriptor(&rnnDesc); 

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
   
    LOG4CPLUS_DEBUG(logger, "cudnnCreateRNNDescriptor Executed");
    //cout << " DEBUG - cudnnCreateRNNDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(DestroyRNNDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyRNNDescriptor"));

    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnDestroyRNNDescriptor(rnnDesc);

    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroyRNNDescriptor Executed");
    //cout << " DEBUG - cudnnDestroyRNNDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs);
}

#if CUDNN_VERSION < 8000
    CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v5){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDescriptor_v5"));

    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int hiddenSize = in->Get<int>();
    int numLayers = in->Get<int>();
    cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
    cudnnRNNInputMode_t inputMode = in->Get<cudnnRNNInputMode_t>();
    cudnnDirectionMode_t direction = in->Get<cudnnDirectionMode_t>();
    cudnnRNNMode_t mode = in->Get<cudnnRNNMode_t>();
    cudnnDataType_t mathPrec = in->Get<cudnnDataType_t>();

    cudnnStatus_t cs = cudnnSetRNNDescriptor_v5(rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, mathPrec);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }

    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDescriptor_v5 Executed");
    //cout << " DEBUG - cudnnSetRNNDescriptor_v5 Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}
#endif
#if CUDNN_VERSION >= 6000
CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v6){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDescriptor_v6"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int hiddenSize = in->Get<int>();
    int numLayers  = in->Get<int>();
    cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
    cudnnRNNInputMode_t inputMode = in->Get<cudnnRNNInputMode_t>();
    cudnnDirectionMode_t direction = in->Get<cudnnDirectionMode_t>();
    cudnnRNNMode_t mode = in->Get<cudnnRNNMode_t>();
    cudnnRNNAlgo_t algo = in->Get<cudnnRNNAlgo_t>();
    cudnnDataType_t mathPrec = in->Get<cudnnDataType_t>();

    cudnnStatus_t cs = cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDescriptor_v6 Executed");
    //cout << " DEBUG - cudnnSetRNNDescriptor_v6 Executed"<<endl;
    return std::make_shared<Result>(cs, out);         
}

CUDNN_ROUTINE_HANDLER(GetRNNDescriptor_v6){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNDescriptor_v6"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int hiddenSize;
    int numLayers;
    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnRNNInputMode_t inputMode;
    cudnnDirectionMode_t direction;
    cudnnRNNMode_t mode;
    cudnnRNNAlgo_t algo;
    cudnnDataType_t mathPrec;

    cudnnStatus_t cs = cudnnGetRNNDescriptor_v6(handle, rnnDesc, &hiddenSize, &numLayers, &dropoutDesc, &inputMode, &direction, &mode, &algo, &mathPrec);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<int>(hiddenSize);
        out->Add<int>(numLayers);
        out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
        out->Add<cudnnRNNInputMode_t>(inputMode);
        out->Add<cudnnDirectionMode_t>(direction);
        out->Add<cudnnRNNMode_t>(mode);
        out->Add<cudnnRNNAlgo_t>(algo);
        out->Add<cudnnDataType_t>(mathPrec);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }

    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNDescriptor_v6 Executed");
    //cout << " DEBUG - cudnnGetRNNDescriptor_v6 Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}
#endif

#if CUDNN_VERSION >= 8000
CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v8){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDescriptor_v8"));

    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();

    cudnnRNNAlgo_t algo = in->Get<cudnnRNNAlgo_t>();
    cudnnRNNMode_t cellMode= in->Get<cudnnRNNMode_t>();
    cudnnRNNBiasMode_t biasMode= in->Get<cudnnRNNBiasMode_t>();
    cudnnDirectionMode_t dirMode= in->Get<cudnnDirectionMode_t>();
    cudnnRNNInputMode_t inputMode= in->Get<cudnnRNNInputMode_t>();
    cudnnDataType_t dataType= in->Get<cudnnDataType_t>();
    cudnnDataType_t mathPrec= in->Get<cudnnDataType_t>();
    cudnnMathType_t mathType= in->Get<cudnnMathType_t>();
    int32_t inputSize= in->Get<int>();
    int32_t hiddenSize= in->Get<int>();
    int32_t projSize= in->Get<int>();
    int32_t numLayers= in->Get<int>();
    cudnnDropoutDescriptor_t dropoutDesc= (cudnnDropoutDescriptor_t)in->Get<long long int>();
    uint32_t auxFlags= in->Get<int>();

    cudnnStatus_t cs = cudnnSetRNNDescriptor_v8(rnnDesc, algo,
             cellMode,
             biasMode,
             dirMode,
             inputMode,
             dataType,
             mathPrec,
             mathType,
             inputSize,
             hiddenSize,
             projSize,
             numLayers,
             dropoutDesc,
             auxFlags);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }

    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDescriptor_v8 Executed");
    //cout << " DEBUG - cudnnSetRNNDescriptor_v8 Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNDescriptor_v8) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNDescriptor_v8"));

    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t) in->Get<long long int>();

    cudnnRNNAlgo_t algo;
    cudnnRNNMode_t cellMode;
    cudnnRNNBiasMode_t biasMode;
    cudnnDirectionMode_t dirMode;
    cudnnRNNInputMode_t inputMode;
    cudnnDataType_t dataType;
    cudnnDataType_t mathPrec;
    cudnnMathType_t mathType;
    int32_t inputSize;
    int32_t hiddenSize;
    int32_t projSize;
    int32_t numLayers;
    cudnnDropoutDescriptor_t dropoutDesc;
    uint32_t auxFlags;

    cudnnStatus_t cs = cudnnGetRNNDescriptor_v8(rnnDesc,
                                                &algo,
                                                &cellMode, &biasMode, &dirMode, &inputMode,
                                                &dataType, &mathPrec, &mathType,
                                                &inputSize, &hiddenSize, &projSize, &numLayers,
                                                &dropoutDesc, &auxFlags);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnRNNAlgo_t>(algo);
        out->Add<cudnnRNNMode_t>(cellMode);
        out->Add<cudnnRNNBiasMode_t>(biasMode);
        out->Add<cudnnDirectionMode_t>(dirMode);
        out->Add<cudnnRNNInputMode_t>(inputMode);
        out->Add<cudnnDataType_t>(dataType);
        out->Add<cudnnDataType_t>(mathPrec);
        out->Add<cudnnMathType_t>(mathType);
        out->Add<int>(inputSize);
        out->Add<int>(hiddenSize);
        out->Add<int>(projSize);
        out->Add<int>(numLayers);
        out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
        out->Add<int>(auxFlags);

    } catch (string e) {
        LOG4CPLUS_DEBUG(logger, e);
        return std::make_shared<Result>(cs);
    }

    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNDescriptor_v8 Executed");
    //cout << " DEBUG - cudnnGetRNNDescriptor_v8 Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}
#endif

CUDNN_ROUTINE_HANDLER(SetRNNMatrixMathType){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNMatrixMathType"));
   
   cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
   cudnnMathType_t mType = in->Get<cudnnMathType_t>();
   
   cudnnStatus_t cs = cudnnSetRNNMatrixMathType(rnnDesc, mType);

   
   LOG4CPLUS_DEBUG(logger, "cudnnSetRNNMatrixMathType Executed");
   //cout << " DEBUG - cudnnSetRNNMatrixMathType Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetRNNMatrixMathType){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNMatrixMathType"));

    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    cudnnMathType_t mType;

    cudnnStatus_t cs = cudnnGetRNNMatrixMathType(rnnDesc, &mType);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnMathType_t>(mType);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNMatrixMathType Executed");
    //cout << " DEBUG - cudnnGetRNNMatrixMathType Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetRNNBiasMode){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNBiasMode"));

     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>(); 
     cudnnRNNBiasMode_t biasMode = in->Get<cudnnRNNBiasMode_t>();
     
     cudnnStatus_t cs = cudnnSetRNNBiasMode(rnnDesc, biasMode);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNBiasMode Executed");
    //cout << " DEBUG - cudnnSetRNNBiasMode Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNBiasMode){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNBiasMode"));

    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    cudnnRNNBiasMode_t biasMode;

    cudnnStatus_t  cs = cudnnGetRNNBiasMode(rnnDesc, &biasMode);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNBiasMode_t>(biasMode);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetRNNBiasMode Executed");
    //cout << " DEBUG - cudnnGetRNNBiasMode Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(RNNSetClip){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNSetClip"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     cudnnRNNClipMode_t clipMode = in->Get<cudnnRNNClipMode_t>();
     cudnnNanPropagation_t clipNanOpt = in->Get<cudnnNanPropagation_t>();
     double lclip = in->Get<double>();
     double rclip = in->Get<double>();

     cudnnStatus_t cs = cudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
    
      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnRNNSetClip Executed");
    //cout << " DEBUG - cudnnRNNSetClip Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(RNNGetClip){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNGetClip"));
   
     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     cudnnRNNClipMode_t clipMode;
     cudnnNanPropagation_t clipNanOpt;
     double lclip;
     double rclip;

     cudnnStatus_t cs = cudnnRNNGetClip(handle, rnnDesc, &clipMode, &clipNanOpt, &lclip, &rclip);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNClipMode_t>(clipMode);
          out->Add<cudnnNanPropagation_t>(clipNanOpt);
          out->Add<double>(lclip);
          out->Add<double>(rclip);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnRNNGetClip Execute");
    //cout << " DEBUG - cudnnRNNGetClip Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetRNNProjectionLayers){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNProjectionLayers"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int recProjSize = in->Get<int>();
    int outProjSize = in->Get<int>();

    cudnnStatus_t cs = cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);

    

    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNProjectionLayers Executed");
    //cout << " DEBUG - cudnnSetRNNProjectionLayers Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetRNNProjectionLayers){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNProjectionLayers"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int recProjSize;
    int outProjSize;

    cudnnStatus_t cs = cudnnGetRNNProjectionLayers(handle, rnnDesc, &recProjSize, &outProjSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<int>(recProjSize);
          out->Add<int>(outProjSize);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetRNNProjectionLayers Executed");
    //cout << " DEBUG - cudnnGetRNNProjectionLayers Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreatePersistentRNNPlan){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreatePersistentRNNPlan"));

    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int minibatch = in->Get<int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    cudnnPersistentRNNPlan_t plan;

    cudnnStatus_t cs = cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, &plan);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnPersistentRNNPlan_t>(plan);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnCreatePersistentRNNPlan Executed");
    //cout << " DEBUG - cudnnCreatePersistentRNNPlan Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyPersistentRNNPlan){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyPersistentRNNPlan"));

    cudnnPersistentRNNPlan_t plan;
    
    cudnnStatus_t cs = cudnnDestroyPersistentRNNPlan(plan);

    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroyPersistentRNNPlan Executed");
    //cout << " DEBUG - cudnnDestroyPersistentRNNPlan Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetPersistentRNNPlan){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPersistentRNNPlan"));
  
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    cudnnPersistentRNNPlan_t plan = in->Get<cudnnPersistentRNNPlan_t>();
   
    cudnnStatus_t cs = cudnnSetPersistentRNNPlan(rnnDesc, plan);

    
     LOG4CPLUS_DEBUG(logger, "cudnnSetPersistentRNNPlan Executed");
    //cout << " DEBUG - cudnnSetPersistentRNNPlan Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetRNNWorkspaceSize){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNWorkspaceSize"));
     
     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int seqLength = in->Get<int>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     size_t sizeInBytes;

     cudnnStatus_t cs = cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, &xDesc, &sizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNWorkspaceSize Executed");
    //cout << " DEBUG - cudnnGetRNNWorkspaceSize Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNTrainingReserveSize){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNTrainingReserveSize"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int seqLength = in->Get<int>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     size_t sizeInBytes;

     cudnnStatus_t cs = cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, &xDesc, &sizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetRNNTrainingReserveSize Executed");
    //cout << " DEBUG - cudnnGetRNNTrainingReserveSize Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNParamsSize){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNParamsSize"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    size_t sizeInBytes;
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();

     cudnnStatus_t cs = cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, &sizeInBytes, dataType);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetRNNParamsSize Executed");
    //cout << " DEBUG - cudnnGetRNNParamsSize Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNLinLayerMatrixParams){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNLinLayerMatrixParams"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int pseudoLayer = in->Get<int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    void *w = in->Assign<void>();
    int linLayerID = in->Get<int>();
    cudnnFilterDescriptor_t linLayerMatDesc;
    void *linLayerMat = in->Assign<void>();

    cudnnStatus_t cs = cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, &linLayerMat);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnFilterDescriptor_t>(linLayerMatDesc);
          out->Add<void>(linLayerMat);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNLinLayerMatrixParams Executed");
    //cout << " DEBUG - cudnnGetRNNLinLayerMatrixParams Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNLinLayerBiasParams){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNLinLayerBiasParams")); 
     
     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int pseudoLayer = in->Get<int>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
     void *w = in->Assign<void>();
     int linLayerID = in->Get<int>();
     cudnnFilterDescriptor_t linLayerBiasDesc;
     void *linLayerBias;

     cudnnStatus_t cs = cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, &linLayerBias);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnFilterDescriptor_t>(linLayerBiasDesc);
          out->Add<void>(linLayerBias);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNLinLayerBiasParams Executed");
    //cout << " DEBUG - cudnnGetRNNLinLayerBiasParams Executed"<<endl;
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(RNNForwardInference){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardInference"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int seqLength = in->Get<int>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *x = in->Assign<void>();
     cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *hx = in->Assign<void>();
     cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *cx = in->Assign<void>();
     cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
     void* w = in->Assign<void>();
     cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *y = in->Assign<void>();
     cudnnTensorDescriptor_t hyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *hy = in->Assign<void>();
     cudnnTensorDescriptor_t cyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *cy = in->Assign<void>();
     void *workspace = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();

     cudnnStatus_t cs = cudnnRNNForwardInference(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes);
     
      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
          out->Add<void>(hy);
          out->Add<void>(cy);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, " cudnnRNNForwardInference Executed");
    //cout << " DEBUG - cudnnRNNForwardInference Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(RNNForwardTraining){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardTraining"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int seqLength = in->Get<int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hx = in->Assign<void>();
    cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *cx = in->Assign<void>();
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    void *w = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();
    cudnnTensorDescriptor_t hyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hy = in->Assign<void>();
    cudnnTensorDescriptor_t cyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *cy = in->Assign<void>();
    void *workspace = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>();
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnRNNForwardTraining(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
          out->Add<void>(hy);
          out->Add<void>(cy);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnRNNForwardTraining Executed");
    //cout << " DEBUG - cudnnRNNForwardTraining Executed"<<endl;
    return std::make_shared<Result>(cs, out);         
}

CUDNN_ROUTINE_HANDLER(RNNBackwardData){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardData"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int seqLength = in->Get<int>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();
    cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dy = in->Assign<void>();
    cudnnTensorDescriptor_t dhyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dhy = in->Assign<void>();
    cudnnTensorDescriptor_t dcyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dcy = in->Assign<void>();
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    void *w = in->Assign<void>();
    cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hx = in->Assign<void>();
    cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *cx = in->Assign<void>();
    cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dx = in->Assign<void>();
    cudnnTensorDescriptor_t dhxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dhx = in->Assign<void>();
    cudnnTensorDescriptor_t dcxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *dcx = in->Assign<void>();
    void *workspace = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>();
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnRNNBackwardData(handle, rnnDesc, seqLength, &yDesc, y, &dyDesc, dy,dhyDesc, dhy,  dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, &dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dx);
          out->Add<void>(dhx);
          out->Add<void>(dcx);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardData Executed");
    //cout << " DEBUG - cudnnRNNBackwardData Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(RNNBackwardWeights){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardWeights"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    int seqLength = in->Get<int>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hx = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>();
    void *workspace = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    void *dw = in->Assign<void>();
    void *reserveSpace = in->Assign<void>();
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, &yDesc, y, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dw);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardWeights Executed");
    //cout << " DEBUG - cudnnRNNBackwardWeights Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetRNNPaddingMode){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNPaddingMode"));

     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     cudnnRNNPaddingMode_t paddingMode = in->Get<cudnnRNNPaddingMode_t>();
     
     cudnnStatus_t cs = cudnnSetRNNPaddingMode(rnnDesc, paddingMode);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnSetRNNPaddingMode Executed");
    //cout << " DEBUG - cudnnSetRNNPaddingMode Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNPaddingMode){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNPaddingMode"));

     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     cudnnRNNPaddingMode_t paddingMode = in->Get<cudnnRNNPaddingMode_t>();

     cudnnStatus_t cs = cudnnGetRNNPaddingMode(rnnDesc, &paddingMode);
     
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNPaddingMode Executed");
    //cout << " DEBUG - cudnnGetRNNPaddingMode Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateRNNDataDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateRNNDataDescriptor"));

     cudnnRNNDataDescriptor_t rnnDataDesc;

     cudnnStatus_t cs = cudnnCreateRNNDataDescriptor(&rnnDataDesc);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDataDescriptor_t>(rnnDataDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnCreateRNNDataDescriptor Executed");
    //cout << " DEBUG - cudnnCreateRNNDataDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyRNNDataDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyRNNDataDescriptor"));

     cudnnRNNDataDescriptor_t rnnDataDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     
     cudnnStatus_t cs = cudnnDestroyRNNDataDescriptor(rnnDataDesc);
     
     
     LOG4CPLUS_DEBUG(logger, "cudnnDestroyRNNDataDescriptor Executed");
     //cout << " DEBUG - cudnnDestroyRNNDataDescriptor Executed"<<endl;
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetRNNDataDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDataDescriptor"));

     cudnnRNNDataDescriptor_t rnnDataDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
     cudnnRNNDataLayout_t layout = in->Get<cudnnRNNDataLayout_t>();
     int maxSeqLength = in->Get<int>();
     int batchSize = in->Get<int>();
     int vectorSize = in->Get<int>();
     int *seqLengthArray = in->Assign<int>();
     void *paddingFill = in->Assign<void>();
    
     cudnnStatus_t cs = cudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDataDescriptor_t>(rnnDataDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDataDescriptor Executed");
    //cout << " DEBUG - cudnnSetRNNDataDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNDataDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNDataDescriptor"));

     cudnnRNNDataDescriptor_t rnnDataDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     cudnnDataType_t dataType;
     cudnnRNNDataLayout_t layout;
     int maxSeqLength;
     int batchSize;
     int vectorSize;
     int arrayLengthRequested = in->Get<int>();
     int *seqLengthArray = in->Assign<int>();
     void *paddingFill = in->Assign<void>();

     cudnnStatus_t cs = cudnnGetRNNDataDescriptor(rnnDataDesc, &dataType, &layout, &maxSeqLength, &batchSize, &vectorSize, arrayLengthRequested, seqLengthArray, paddingFill);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnDataType_t>(dataType);
          out->Add<cudnnRNNDataLayout_t>(layout);
          out->Add<int>(maxSeqLength);
          out->Add<int>(batchSize);
          out->Add<int>(vectorSize);
          out->Add<int>(seqLengthArray);
          out->Add<void>(paddingFill);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    

     LOG4CPLUS_DEBUG(logger, "cudnnGetRNNDataDescriptor Executed");
     //cout << " DEBUG - cudnnGetRNNDataDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(RNNForwardTrainingEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardTrainingEx"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    cudnnRNNDataDescriptor_t xDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hx = in->Assign<void>();
    cudnnTensorDescriptor_t cxDesc =(cudnnTensorDescriptor_t)in->Get<long long int>();
    void *cx = in->Assign<void>();
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    void *w = in->Assign<void>();
    cudnnRNNDataDescriptor_t yDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>(); //OUTPUT
    cudnnTensorDescriptor_t hyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hy = in->Assign<void>(); //OUTPUT
    cudnnTensorDescriptor_t cyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *cy = in->Assign<void>(); //OUTPUT
    cudnnRNNDataDescriptor_t kDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *keys = in->Assign<void>();
    cudnnRNNDataDescriptor_t cDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *cAttn = in->Assign<void>();
    cudnnRNNDataDescriptor_t iDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *iAttn = in->Assign<void>();
    cudnnRNNDataDescriptor_t qDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *queries = in->Assign<void>();
    void *workSpace = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
          out->Add<void>(hy);
          out->Add<void>(cy);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, " cudnnRNNForwardTrainingEx Executed");
    //cout << " DEBUG - cudnnRNNForwardTrainingEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(RNNForwardInferenceEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardInferenceEx"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
    cudnnRNNDataDescriptor_t xDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *x = in->Assign<void>();
    cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hx = in->Assign<void>();
    cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *cx = in->Assign<void>();
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    void *w = in->Assign<void>();
    cudnnRNNDataDescriptor_t yDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *y = in->Assign<void>(); //OUTPUT
    cudnnTensorDescriptor_t hyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *hy = in->Assign<void>(); //OUTPUT
    cudnnTensorDescriptor_t cyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void *cy = in->Assign<void>(); //OUTPUT
    cudnnRNNDataDescriptor_t kDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *keys = in->Assign<void>();
    cudnnRNNDataDescriptor_t cDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *cAttn = in->Assign<void>();
    cudnnRNNDataDescriptor_t iDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *iAttn = in->Assign<void>();
    cudnnRNNDataDescriptor_t qDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
    void *queries = in->Assign<void>();
    void *workSpace = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
          out->Add<void>(hy);
          out->Add<void>(cy);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnRNNForwardInferenceEx Executed");
    //cout << " DEBUG - cudnnRNNForwardInferenceEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(RNNBackwardDataEx){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardDataEx"));
   
     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     cudnnRNNDataDescriptor_t yDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     void *y = in->Assign<void>();
     cudnnRNNDataDescriptor_t dyDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     void *dy = in->Assign<void>();
     cudnnRNNDataDescriptor_t dcDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     void *dcAttn = in->Assign<void>();
     cudnnTensorDescriptor_t dhyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void * dhy = in->Assign<void>();
     cudnnTensorDescriptor_t dcyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dcy = in->Assign<void>();
     cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
     void *w = in->Assign<void>();
     cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *hx = in->Assign<void>();
     cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *cx = in->Assign<void>();
     cudnnRNNDataDescriptor_t dxDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     void *dx = in->Assign<void>(); //OUTPUT
     cudnnTensorDescriptor_t dhxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dhx = in->Assign<void>(); //OUTPUT
     cudnnTensorDescriptor_t dcxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *dcx = in->Assign<void>(); //OUTPUT
     cudnnRNNDataDescriptor_t dkDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     void *dkeys = in->Assign<void>();
     void *workSpace = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();
     void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT
     size_t reserveSpaceSizeInBytes = in->Get<size_t>();
 
     cudnnStatus_t cs = cudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

       std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dx);
          out->Add<void>(dhx);
          out->Add<void>(dcx);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardDataEx Executed");
    //cout << " DEBUG - cudnnRNNBackwardDataEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}
 
CUDNN_ROUTINE_HANDLER(RNNBackwardWeightsEx){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardWeightsEx"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     cudnnRNNDataDescriptor_t xDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     void *x = in->Assign<void>();
     cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *hx = in->Assign<void>();
     cudnnRNNDataDescriptor_t yDesc = (cudnnRNNDataDescriptor_t)in->Get<long long int>();
     void *y = in->Assign<void>();
     void *workSpace = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();
     cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
     void *dw = in->Assign<void>(); //INPUT/OUTPUT
     void *reserveSpace = in->Assign<void>();
     size_t reserveSpaceSizeInBytes = in->Get<long long int>();
    
     cudnnStatus_t cs = cudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dw);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardWeightsEx Executed");
    //cout << " DEBUG - cudnnRNNBackwardWeightsEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetRNNAlgorithmDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNAlgorithmDescriptor"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();

     cudnnStatus_t cs = cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNAlgorithmDescriptor Executed");
    //cout << " DEBUG - cudnnSetRNNAlgorithmDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNForwardInferenceAlgorithmMaxCount){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNForwardInferenceAlgorithmMaxCount"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int count;

     cudnnStatus_t cs = cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, &count);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<int>(count);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNForwardInferenceAlgorithmMaxCount Executed");
    //cout << " DEBUG - cudnnGetRNNForwardInferenceAlgorithmMaxCount Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindRNNForwardInferenceAlgorithmEx){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNForwardInferenceAlgorithmEx"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int seqLength = in->Get<int>();
     cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *x = in->Assign<void>();
     cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *hx = in->Assign<void>();
     cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *cx = in->Assign<void>();
     cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
     void *w = in->Assign<void>();
     cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *y = in->Assign<void>(); //OUTPUT
     cudnnTensorDescriptor_t hyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *hy = in->Assign<void>(); //OUTPUT
     cudnnTensorDescriptor_t cyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void* cy = in->Assign<void>(); //OUTPUT
     float findIntensity =  in->Get<float>();
     int requestedAlgoCount = in->Get<int>();
     int returnedAlgoCount; //OUTPUT
     cudnnAlgorithmPerformance_t perfResults; //OUTPUT
     void *workspace = in->Assign<void>();
     size_t workSpaceSizeInBytes =in->Get<size_t>();

     cudnnStatus_t cs = cudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, &returnedAlgoCount, &perfResults, workspace, workSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
          out->Add<void>(hy);
          out->Add<void>(cy);
          out->Add<int>(returnedAlgoCount);
          out->Add<cudnnAlgorithmPerformance_t>(perfResults);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
      
     LOG4CPLUS_DEBUG(logger, "cudnnFindRNNForwardInferenceAlgorithmEx Executed");
    //cout << " DEBUG - cudnnFindRNNForwardInferenceAlgorithmEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNForwardTrainingAlgorithmMaxCount){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNForwardTrainingAlgorithmMaxCount"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int count;

     cudnnStatus_t cs = cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, &count);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<int>(count);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNForwardTrainingAlgorithmMaxCount Executed");
    //cout << " DEBUG - cudnnGetRNNForwardTrainingAlgorithmMaxCount Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindRNNForwardTrainingAlgorithmEx){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNForwardTrainingAlgorithmEx"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
   int seqLength = in->Get<int>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->Assign<void>();
   cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *hx = in->Assign<void>();
   cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *cx = in->Assign<void>();
   cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   void *w = in->Assign<void>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->Assign<void>(); //OUTPUT
   cudnnTensorDescriptor_t hyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *hy = in->Assign<void>(); //OUTPUT
   cudnnTensorDescriptor_t cyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *cy = in->Assign<void>(); //OUTPUT
   float findIntensity = in->Get<float>();
   int requestedAlgoCount = in->Get<int>();
   int returnedAlgoCount; //OUTPUT
   cudnnAlgorithmPerformance_t perfResults; //OUTPUT
   void *workspace = in->Assign<void>();
   size_t workSpaceSizeInBytes = in->Get<size_t>();
   void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT
   size_t reserveSpaceSizeInBytes = in->Get<size_t>();

   cudnnStatus_t cs = cudnnFindRNNForwardTrainingAlgorithmEx(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, &returnedAlgoCount, &perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(y);
          out->Add<void>(hy);
          out->Add<void>(cy);
          out->Add<int>(returnedAlgoCount);
          out->Add<cudnnAlgorithmPerformance_t>(perfResults);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnFindRNNForwardTrainingAlgorithmEx Executed");
    //cout << " DEBUG - cudnnFindRNNForwardTrainingAlgorithmEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(GetRNNBackwardDataAlgorithmMaxCount){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNBackwardDataAlgorithmMaxCount"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
     int count;
    
     cudnnStatus_t cs = cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, &count);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<int>(count);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetRNNBackwardDataAlgorithmMaxCount Executed");
    //cout << " DEBUG - cudnnGetRNNBackwardDataAlgorithmMaxCount Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindRNNBackwardDataAlgorithmEx){
      Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNBackwardDataAlgorithmEx"));

       cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
       cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
       int seqLength = in->Get<int>();
       cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *y = in->Assign<void>();
       cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *dy = in->Assign<void>();
       cudnnTensorDescriptor_t dhyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *dhy = in->Assign<void>();
       cudnnTensorDescriptor_t dcyDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *dcy = in->Assign<void>();
       cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
       void *w = in->Assign<void>();
       cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *hx = in->Assign<void>();
       cudnnTensorDescriptor_t cxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *cx = in->Assign<void>();
       cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *dx = in->Assign<void>(); //OUTPUT
       cudnnTensorDescriptor_t dhxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *dhx = in->Assign<void>(); //OUTPUT
       cudnnTensorDescriptor_t dcxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
       void *dcx = in->Assign<void>(); //OUTPUT
       float findIntensity = in->Get<float>();
       int requestedAlgoCount = in->Get<int>();
       int returnedAlgoCount; //OUTPUT
       cudnnAlgorithmPerformance_t perfResults; //OUTPUT
       void *workspace = in->Assign<void>();
       size_t workSpaceSizeInBytes = in->Get<size_t>();
       void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT
       size_t reserveSpaceSizeInBytes = in->Get<size_t>();

       cudnnStatus_t cs = cudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, &yDesc, y, &dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, &dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, &returnedAlgoCount, &perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dx);
          out->Add<void>(dhx);
          out->Add<void>(dcx);
          out->Add<int>(returnedAlgoCount);
          out->Add<cudnnAlgorithmPerformance_t>(perfResults);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnFindRNNBackwardDataAlgorithmEx Executed");
    //cout << " DEBUG - cudnnFindRNNBackwardDataAlgorithmEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}


CUDNN_ROUTINE_HANDLER(GetRNNBackwardWeightsAlgorithmMaxCount){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNBackwardWeightsAlgorithmMaxCount"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
   int count;

   cudnnStatus_t cs = cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, &count);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<int>(count);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNBackwardWeightsAlgorithmMaxCount Executed");
    //cout << " DEBUG - cudnnGetRNNBackwardWeightsAlgorithmMaxCount Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindRNNBackwardWeightsAlgorithmEx){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNBackwardWeightsAlgorithmEx"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>();
   int seqLength = in->Get<int>();
   cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *x = in->Assign<void>();
   cudnnTensorDescriptor_t hxDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *hx = in->Assign<void>();
   cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
   void *y = in->Assign<void>();
   float findIntensity = in->Get<float>();
   int requestedAlgoCount = in->Get<int>();
   int returnedAlgoCount; //OUTPUT
   cudnnAlgorithmPerformance_t perfResults; //OUTPUT
   void *workspace = in->Assign<void>();
   size_t workSpaceSizeInBytes = in->Get<size_t>();
   cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
   void *dw = in->Assign<void>(); //INPUT/OUTPUT
   void *reserveSpace = in->Assign<void>();
   size_t reserveSpaceSizeInBytes = in->Get<size_t>();

   cudnnStatus_t cs = cudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, &yDesc, y, findIntensity, requestedAlgoCount, &returnedAlgoCount, &perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<int>(returnedAlgoCount);
          out->Add<cudnnAlgorithmPerformance_t>(perfResults);
          out->Add<void>(dw);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnFindRNNBackwardWeightsAlgorithmEx Executed");
    //cout << " DEBUG - cudnnFindRNNBackwardWeightsAlgorithmEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(CreateSeqDataDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateSeqDataDescriptor"));

    cudnnSeqDataDescriptor_t seqDataDesc;

    cudnnStatus_t cs = cudnnCreateSeqDataDescriptor(&seqDataDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnSeqDataDescriptor_t>(seqDataDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnCreateSeqDataDescriptor Executed");
    //cout << " DEBUG - cudnnCreateSeqDataDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(DestroySeqDataDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroySeqDataDescriptor"));

   cudnnSeqDataDescriptor_t seqDataDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();

   cudnnStatus_t cs = cudnnDestroySeqDataDescriptor(seqDataDesc);

   
   LOG4CPLUS_DEBUG(logger, "cudnnDestroySeqDataDescriptor Executed");
   //cout << " DEBUG - cudnnDestroySeqDataDescriptor Executed"<<endl;
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetSeqDataDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetSeqDataDescriptor"));

    cudnnSeqDataDescriptor_t seqDataDesc;
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();
    cudnnSeqDataAxis_t *axes = in->Assign<cudnnSeqDataAxis_t>();
    size_t seqLengthArraySize = in->Get<size_t>();
    int *seqLengthArray = in->Assign<int>();
    void *paddingFill = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnSeqDataDescriptor_t>(seqDataDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetSeqDataDescriptor Executed");
    //cout << " DEBUG - cudnnSetSeqDataDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetSeqDataDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetSeqDataDescriptor"));

    cudnnSeqDataDescriptor_t seqDataDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType; //OUTPUT
    int nbDims; //OUTPUT
    int nbDimsRequested = in->Get<int>();
    int *dimA = in->Assign<int>(); //OUTPUT
    cudnnSeqDataAxis_t *axes = in->Assign<cudnnSeqDataAxis_t>(); //OUTPUT
    size_t seqLengthArraySize; //OUTPUT
    size_t seqLengthSizeRequested = in->Get<size_t>();
    int *seqLengthArray = in->Assign<int>(); //OUTPUT
    void *paddingFill = in->Assign<void>(); //OUTPUT

    cudnnStatus_t cs = cudnnGetSeqDataDescriptor(seqDataDesc, &dataType, &nbDims, nbDimsRequested, dimA, axes, &seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnDataType_t>(dataType);
          out->Add<int>(nbDims);
          out->Add<int>(dimA);
          out->Add<cudnnSeqDataAxis_t>(axes);
          out->Add<int>(seqLengthArraySize);
          out->Add<int>(seqLengthArray);
          out->Add<void>(paddingFill);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetSeqDataDescriptor Executed");
    //cout << " DEBUG - cudnnGetSeqDataDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateAttnDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateAttnDescriptor"));

     cudnnAttnDescriptor_t attnDesc;

     cudnnStatus_t cs = cudnnCreateAttnDescriptor(& attnDesc);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnAttnDescriptor_t>(attnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnCreateAttnDescriptor Executed");
    //cout << " DEBUG - cudnnCreateAttnDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(DestroyAttnDescriptor){

     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyAttnDescriptor"));
   
     cudnnAttnDescriptor_t attnDesc = (cudnnAttnDescriptor_t)in->Get<long long int>();
     
     cudnnStatus_t cs = cudnnDestroyAttnDescriptor(attnDesc); 

     
      LOG4CPLUS_DEBUG(logger, "cudnnDestroyAttnDescriptor Executed");
     //cout << " DEBUG - cudnnDestroyAttnDescriptor Executed"<<endl;
     return std::make_shared<Result>(cs); 
}

CUDNN_ROUTINE_HANDLER(SetAttnDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetAttnDescriptor"));

     cudnnAttnDescriptor_t attnDesc; //OUTPUT
     unsigned attnMode = in->Get<unsigned>();
     int nHeads = in->Get<int>();
     double smScaler = in->Get<double>();
     cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
     cudnnDataType_t computePrec = in->Get<cudnnDataType_t>();
     cudnnMathType_t mathType = in->Get<cudnnMathType_t>();
     cudnnDropoutDescriptor_t attnDropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
     cudnnDropoutDescriptor_t postDropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
     int qSize = in->Get<int>();
     int kSize = in->Get<int>();
     int vSize = in->Get<int>();
     int qProjSize = in->Get<int>();
     int kProjSize = in->Get<int>();
     int vProjSize = in->Get<int>();
     int oProjSize = in->Get<int>();
     int qoMaxSeqLength = in->Get<int>();
     int kvMaxSeqLength = in->Get<int>();
     int maxBatchSize = in->Get<int>();
     int maxBeamSize = in->Get<int>();

     cudnnStatus_t cs = cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnAttnDescriptor_t>(attnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnSetAttnDescriptor Executed");
    //cout << " DEBUG - cudnnSetAttnDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetAttnDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAttnDescriptor"));

     cudnnAttnDescriptor_t attnDesc = (cudnnAttnDescriptor_t)in->Get<long long int>();
     unsigned attnMode;
     int nHeads;
     double smScaler;
     cudnnDataType_t dataType;
     cudnnDataType_t computePrec;
     cudnnMathType_t mathType;
     cudnnDropoutDescriptor_t attnDropoutDesc;
     cudnnDropoutDescriptor_t postDropoutDesc;
     int qSize;
     int kSize;
     int vSize;
     int qProjSize;
     int kProjSize;
     int vProjSize;
     int oProjSize;
     int qoMaxSeqLength;
     int kvMaxSeqLength;
     int maxBatchSize;
     int maxBeamSize;

     cudnnStatus_t cs = cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<unsigned>(attnMode);
          out->Add<int>(nHeads);
          out->Add<double>(smScaler);
          out->Add<cudnnDataType_t>(dataType);
          out->Add<cudnnDataType_t>(computePrec);
          out->Add<cudnnMathType_t>(mathType);
          out->Add<cudnnDropoutDescriptor_t>(attnDropoutDesc);
          out->Add<cudnnDropoutDescriptor_t>(postDropoutDesc);
          out->Add<int>(qSize);
          out->Add<int>(kSize);
          out->Add<int>(vSize);
          out->Add<int>(qProjSize);
          out->Add<int>(kProjSize);
          out->Add<int>(vProjSize);
          out->Add<int>(oProjSize);
          out->Add<int>(qoMaxSeqLength);
          out->Add<int>(kvMaxSeqLength);
          out->Add<int>(maxBatchSize);
          out->Add<int>(maxBeamSize);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetAttnDescriptor Executed");
    //cout << " DEBUG - cudnnGetAttnDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetMultiHeadAttnBuffers){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMultiHeadAttnBuffers"));
     
     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnAttnDescriptor_t attnDesc = (cudnnAttnDescriptor_t)in->Get<long long int>();
     size_t weightSizeInBytes;
     size_t workSpaceSizeInBytes;
     size_t reserveSpaceSizeInBytes;

     cudnnStatus_t cs = cudnnGetMultiHeadAttnBuffers(handle, attnDesc, &weightSizeInBytes, &workSpaceSizeInBytes, &reserveSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(weightSizeInBytes);
          out->Add<size_t>(workSpaceSizeInBytes);
          out->Add<size_t>(reserveSpaceSizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetMultiHeadAttnBuffers Executed");
    //cout << " DEBUG - cudnnGetMultiHeadAttnBuffers Executed"<<endl;
    return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetMultiHeadAttnWeights){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMultiHeadAttnWeights"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnAttnDescriptor_t attnDesc = (cudnnAttnDescriptor_t)in->Get<long long int>();
    cudnnMultiHeadAttnWeightKind_t wKind = (cudnnMultiHeadAttnWeightKind_t)in->Get<long long int>();
    size_t weightSizeInBytes = in->Get<size_t>();
    void *weights = in->Assign<void>();
    cudnnTensorDescriptor_t wDesc;
    void *wAddr = in->Assign<void>();

    cudnnStatus_t cs = cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, &wAddr);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnTensorDescriptor_t>(wDesc);
          out->Add<void>(wAddr);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetMultiHeadAttnWeights Executed");
    //cout << " DEBUG - cudnnGetMultiHeadAttnWeights Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(MultiHeadAttnForward){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MultiHeadAttnForward"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnAttnDescriptor_t attnDesc = (cudnnAttnDescriptor_t)in->Get<long long int>();
     int currIdx = in->Get<int>();
     int *loWinIdx = in->Assign<int>();
     int *hiWinIdx = in->Assign<int>();
     int *seqLengthArrayQRO = in->Assign<int>();
     int *seqLengthArrayKV = in->Assign<int>();
     cudnnSeqDataDescriptor_t qDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
     void *queries = in->Assign<void>();
     void *residuals = in->Assign<void>();
     cudnnSeqDataDescriptor_t kDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
     void *keys = in->Assign<void>();
     cudnnSeqDataDescriptor_t vDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
     void *values = in->Assign<void>();
     cudnnSeqDataDescriptor_t oDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
     void *output = in->Assign<void>(); //OUTPUT
     size_t weightSizeInBytes = in->Get<size_t>();
     void *weights = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();
     void *workSpace = in->Assign<void>(); //INPUT/OUTPUT
     size_t reserveSpaceSizeInBytes = in->Get<size_t>();
     void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT

     cudnnStatus_t cs = cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, seqLengthArrayQRO, seqLengthArrayKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, output, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(output);
          out->Add<void>(workSpace);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnMultiHeadAttnForward Executed");
    //cout << " DEBUG - cudnnMultiHeadAttnForward Executed"<<endl;
    return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(MultiHeadAttnBackwardData){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MultiHeadAttnBackwardData"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnAttnDescriptor_t attnDesc = (cudnnAttnDescriptor_t)in->Get<long long int>();
    int *loWinIdx = in->Assign<int>();
    int *hiWinIdx = in->Assign<int>();
    int *seqLengthArrayDQDO = in->Assign<int>();
    int *seqLengthArrayDKDV = in->Assign<int>();
    cudnnSeqDataDescriptor_t doDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *dout = in->Assign<void>();
    cudnnSeqDataDescriptor_t dqDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *dqueries = in->Assign<void>(); //OUTPUT
    void *queries = in->Assign<void>();
    cudnnSeqDataDescriptor_t dkDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *dkeys = in->Assign<void>(); //OUTPUT
    void *keys = in->Assign<void>();
    cudnnSeqDataDescriptor_t dvDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *dvalues = in->Assign<void>(); //OUTPUT
    void *values = in->Assign<void>();
    size_t weightSizeInBytes = in->Get<size_t>();
    void *weights = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *workSpace = in->Assign<void>(); //INPUT/OUTPUT
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT

    cudnnStatus_t cs = cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, seqLengthArrayDQDO, seqLengthArrayDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dqueries);
          out->Add<void>(dkeys);
          out->Add<void>(dvalues);
          out->Add<void>(workSpace);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnMultiHeadAttnBackwardData Executed");
    //cout << " DEBUG - cudnnMultiHeadAttnBackwardData Executed"<<endl;
    return std::make_shared<Result>(cs, out);        
}

CUDNN_ROUTINE_HANDLER(MultiHeadAttnBackwardWeights){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MultiHeadAttnBackwardWeights"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnAttnDescriptor_t attnDesc = (cudnnAttnDescriptor_t)in->Get<long long int>();
    cudnnWgradMode_t addGrad = in->Get<cudnnWgradMode_t>();
    cudnnSeqDataDescriptor_t qDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *queries = in->Assign<void>();
    cudnnSeqDataDescriptor_t kDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *keys = in->Assign<void>();
    cudnnSeqDataDescriptor_t vDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *values = in->Assign<void>();
    cudnnSeqDataDescriptor_t doDesc = (cudnnSeqDataDescriptor_t)in->Get<long long int>();
    void *dout = in->Assign<void>();
    size_t weightSizeInBytes = in->Get<size_t>();
    void *weights = in->Assign<void>();
    void *dweights = in->Assign<void>(); //OUTPUT
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *workSpace = in->Assign<void>(); //INPUT/OUTPUT
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT

    cudnnStatus_t cs = cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(dweights);
          out->Add<void>(workSpace);
          out->Add<void>(reserveSpace);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
      
    LOG4CPLUS_DEBUG(logger, "cudnnMultiHeadAttnBackwardWeights Executed");
    //cout << " DEBUG - cudnnMultiHeadAttnBackwardWeights Executed"<<endl;
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(CreateCTCLossDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCTCLossDescriptor"));

    cudnnCTCLossDescriptor_t ctcLossDesc;
    
    cudnnStatus_t cs = cudnnCreateCTCLossDescriptor(&ctcLossDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnCTCLossDescriptor_t>(ctcLossDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateCTCLossDescriptor Executed");
    //cout << " DEBUG - cudnnCreateCTCLossDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetCTCLossDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetCTCLossDescriptor"));

    cudnnCTCLossDescriptor_t ctcLossDesc;
    cudnnDataType_t compType = (cudnnDataType_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnSetCTCLossDescriptor(ctcLossDesc, compType);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnCTCLossDescriptor_t>(ctcLossDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, " cudnnSetCTCLossDescriptor Executed");
    //cout << " DEBUG - cudnnSetCTCLossDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetCTCLossDescriptorEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetCTCLossDescriptorEx"));

    cudnnCTCLossDescriptor_t ctcLossDesc;
    cudnnDataType_t compType = in->Get<cudnnDataType_t>();
    cudnnLossNormalizationMode_t normMode = in->Get<cudnnLossNormalizationMode_t>();
    cudnnNanPropagation_t gradMode = in->Get<cudnnNanPropagation_t>();

    cudnnStatus_t cs = cudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnCTCLossDescriptor_t>(ctcLossDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetCTCLossDescriptorEx Executed");
    //cout << " DEBUG - cudnnSetCTCLossDescriptorEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetCTCLossDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCTCLossDescriptor"));

     cudnnCTCLossDescriptor_t ctcLossDesc = (cudnnCTCLossDescriptor_t)in->Get<long long int>();
     cudnnDataType_t compType;

     cudnnStatus_t cs = cudnnGetCTCLossDescriptor(ctcLossDesc, &compType);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnDataType_t>(compType);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossDescriptor Executed");
    //cout << " DEBUG - cudnnGetCTCLossDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetCTCLossDescriptorEx){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCTCLossDescriptorEx"));

     cudnnCTCLossDescriptor_t ctcLossDesc = (cudnnCTCLossDescriptor_t)in->Get<long long int>();
     cudnnDataType_t compType;
     cudnnLossNormalizationMode_t normMode;
     cudnnNanPropagation_t gradMode;
    
     cudnnStatus_t cs = cudnnGetCTCLossDescriptorEx(ctcLossDesc, &compType, &normMode, &gradMode);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnDataType_t>(compType);
          out->Add<cudnnLossNormalizationMode_t>(normMode);
          out->Add<cudnnNanPropagation_t>(gradMode);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    

     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossDescriptorEx Executed");
     //cout << " DEBUG - cudnnGetCTCLossDescriptorEx Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyCTCLossDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyCTCLossDescriptor"));

     cudnnCTCLossDescriptor_t ctcLossDesc = in->Get<cudnnCTCLossDescriptor_t>();

     cudnnStatus_t cs = cudnnDestroyCTCLossDescriptor(ctcLossDesc);

     
     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossDescriptorEx Executed");
     //cout << " DEBUG - cudnnDestroyCTCLossDescriptor Executed"<<endl;
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CTCLoss){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CTCLoss"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnTensorDescriptor_t probsDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *probs = in->Assign<void>();
     int labels = in->Get<int>();
     int labelLengths = in->Get<int>();
     int inputLengths = in->Get<int>();
     void *costs = in->Assign<void>(); //OUTPUT
     cudnnTensorDescriptor_t gradientsDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     void *gradients = in->Assign<void>(); //OUTPUT
     cudnnCTCLossAlgo_t algo = in->Get<cudnnCTCLossAlgo_t>();
     cudnnCTCLossDescriptor_t ctcLossDesc = (cudnnCTCLossDescriptor_t)in->Get<long long int>();
     void *workspace = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();

     cudnnStatus_t cs = cudnnCTCLoss(handle, probsDesc, probs, &labels, &labelLengths, &inputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(costs);
          out->Add<void>(gradients);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
     
    LOG4CPLUS_DEBUG(logger, "cudnnCTCLoss Executed");
    //cout << " DEBUG - cudnnCTCLoss Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetCTCLossWorkspaceSize){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCTCLossWorkspaceSize"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnTensorDescriptor_t probsDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     cudnnTensorDescriptor_t gradientsDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
     int labels = in->Get<int>();
     int labelLengths = in->Get<int>();
     int inputLengths = in->Get<int>();
     cudnnCTCLossAlgo_t algo = in->Get<cudnnCTCLossAlgo_t>();
     cudnnCTCLossDescriptor_t ctcLossDesc = (cudnnCTCLossDescriptor_t)in->Get<long long int>();
     size_t sizeInBytes;

     cudnnStatus_t cs = cudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, &labels, &labelLengths, &inputLengths, algo, ctcLossDesc, &sizeInBytes);

       std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(sizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
     
     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossWorkspaceSize Executed");
    //cout << " DEBUG - cudnnGetCTCLossWorkspaceSize Executed"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(CreateAlgorithmDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateAlgorithmDescriptor"));

    cudnnAlgorithmDescriptor_t algoDesc;

    cudnnStatus_t cs = cudnnCreateAlgorithmDescriptor(&algoDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
     
     LOG4CPLUS_DEBUG(logger, "cudnnCreateAlgorithmDescriptor Executed");
    //cout << " DEBUG - cudnnCreateAlgorithmDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(	SetAlgorithmDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetAlgorithmDescriptor"));

     cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();
     cudnnAlgorithm_t algorithm = in->Get<cudnnAlgorithm_t>();

     cudnnStatus_t cs = cudnnSetAlgorithmDescriptor(algoDesc, algorithm);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetAlgorithmDescriptor Executed"); 
    //cout << " DEBUG - cudnnSetAlgorithmDescriptor Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetAlgorithmDescriptor){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAlgorithmDescriptor"));

     cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();
     cudnnAlgorithm_t algorithm = in->Get<cudnnAlgorithm_t>();
     
     cudnnStatus_t cs = cudnnGetAlgorithmDescriptor(algoDesc, &algorithm);

     LOG4CPLUS_DEBUG(logger, "cudnnGetAlgorithmDescriptor Executed");
     //cout << " DEBUG - cudnnGetAlgorithmDescriptor Executed"<<endl;
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CopyAlgorithmDescriptor){
      Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CopyAlgorithmDescriptor"));

      cudnnAlgorithmDescriptor_t src = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();
      cudnnAlgorithmDescriptor_t dest = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();

      cudnnStatus_t cs = cudnnCopyAlgorithmDescriptor(src, dest);

      
      LOG4CPLUS_DEBUG(logger, "cudnnCopyAlgorithmDescriptor Executed");
      //cout << " DEBUG - cudnnCopyAlgorithmDescriptor Executed"<<endl;
      return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(	DestroyAlgorithmDescriptor){
      Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyAlgorithmDescriptor"));

      cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();
      
      cudnnStatus_t cs = cudnnDestroyAlgorithmDescriptor(algoDesc);

      
       LOG4CPLUS_DEBUG(logger, "cudnnDestroyAlgorithmDescriptor Executed");
      //cout << " DEBUG - cudnnDestroyAlgorithmDescriptor Executed"<<endl;
      return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateAlgorithmPerformance){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateAlgorithmPerformance"));

     cudnnAlgorithmPerformance_t algoPerf;
     int numberToCreate = in->Get<int>();

     cudnnStatus_t cs = cudnnCreateAlgorithmPerformance(&algoPerf, numberToCreate);
    
     
      LOG4CPLUS_DEBUG(logger, "cudnnCreateAlgorithmPerformance Executed");
     //cout << " DEBUG - cudnnCreateAlgorithmPerformance Executed"<<endl;
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetAlgorithmPerformance){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetAlgorithmPerformance"));

     cudnnAlgorithmPerformance_t algoPerf = (cudnnAlgorithmPerformance_t)in->Get<long long int>();
     cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();
     cudnnStatus_t status = in->Get<cudnnStatus_t>();
     float time = in->Get<float>();
     size_t memory = in->Get<size_t>();

     cudnnStatus_t cs = cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnAlgorithmPerformance_t>(algoPerf);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
     
     LOG4CPLUS_DEBUG(logger, "cudnnSetAlgorithmPerformance Executed");
    //cout << " DEBUG - cudnnSetAlgorithmPerformance Executed"<<endl;
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(GetAlgorithmPerformance){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAlgorithmPerformance"));

    cudnnAlgorithmPerformance_t algoPerf = (cudnnAlgorithmPerformance_t)in->Get<long long int>();
    cudnnAlgorithmDescriptor_t algoDesc;
    cudnnStatus_t status;
    float time;
    size_t memory;

    cudnnStatus_t cs = cudnnGetAlgorithmPerformance(algoPerf, &algoDesc, &status, &time, &memory);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnAlgorithmPerformance_t>(algoPerf);
          out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
          out->Add<cudnnStatus_t>(status);
          out->Add<float>(time);
          out->Add<size_t>(memory);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetAlgorithmPerformance Executed"); 
    //cout << " DEBUG - cudnnGetAlgorithmPerformance Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyAlgorithmPerformance){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyAlgorithmPerformance"));

     cudnnAlgorithmPerformance_t algoPerf = (cudnnAlgorithmPerformance_t)in->Get<long long int>();
     int numberToDestroy = in->Get<int>();
     
      cudnnStatus_t cs = cudnnDestroyAlgorithmPerformance(&algoPerf, numberToDestroy);

      
      LOG4CPLUS_DEBUG(logger, "cudnnDestroyAlgorithmPerformance Executed");
      //cout << " DEBUG - cudnnDestroyAlgorithmPerformance Executed"<<endl;
      return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetAlgorithmSpaceSize){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAlgorithmSpaceSize"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();
     size_t algoSpaceSizeInBytes;

     cudnnStatus_t cs = cudnnGetAlgorithmSpaceSize(handle, algoDesc, &algoSpaceSizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(algoSpaceSizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
     
    LOG4CPLUS_DEBUG(logger, "cudnnGetAlgorithmSpaceSize Executed");
    //cout << " DEBUG - cudnnGetAlgorithmSpaceSize Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SaveAlgorithm){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SaveAlgorithm"));  

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();
    void *algoSpace = in->Assign<void>();
    size_t algoSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);

     LOG4CPLUS_DEBUG(logger, "cudnnSaveAlgorithm Executed");
    //cout << " DEBUG - cudnnSaveAlgorithm Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(RestoreAlgorithm){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RestoreAlgorithm"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    void *algoSpace = in->Assign<void>();
    size_t algoSpaceSizeInBytes = in->Get<size_t>();
    cudnnAlgorithmDescriptor_t algoDesc = (cudnnAlgorithmDescriptor_t)in->Get<long long int>();

    cudnnStatus_t cs = cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);

    
     LOG4CPLUS_DEBUG(logger, "cudnnRestoreAlgorithm Executed");
    //cout << " DEBUG - cudnnRestoreAlgorithm Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetCallback){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetCallback"));

     unsigned mask = in->Get<unsigned>();
     void *udata = in->Assign<void>();
     cudnnCallback_t fptr = in->Get<cudnnCallback_t>();

     cudnnStatus_t cs = cudnnSetCallback(mask, udata, fptr);

    
    LOG4CPLUS_DEBUG(logger, "cudnnSetCallback Executed");
    //cout << " DEBUG - cudnnSetCallback Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetCallback){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCallback"));

     unsigned mask;
     void *udata;
     cudnnCallback_t fptr;

     cudnnStatus_t cs = cudnnSetCallback( mask, &udata, fptr);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<unsigned>(mask);
          out->Add<void>(udata);
          out->Add<cudnnCallback_t>(fptr);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetCallback Executed");
    //cout << " DEBUG - cudnnGetCallback Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFusedOpsConstParamPack){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFusedOpsConstParamPack"));

     cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
     cudnnFusedOps_t ops = in->Get<cudnnFusedOps_t>();

     cudnnStatus_t cs = cudnnCreateFusedOpsConstParamPack(&constPack, ops);

     
     LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsConstParamPack Executed");
     //cout << " DEBUG - cudnnCreateFusedOpsConstParamPack Executed"<<endl;
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(DestroyFusedOpsConstParamPack){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyFusedOpsConstParamPack"));

     cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
    
     cudnnStatus_t cs = cudnnDestroyFusedOpsConstParamPack(constPack);

     LOG4CPLUS_DEBUG(logger, "cudnnDestroyFusedOpsConstParamPack Executed");
     //cout << " DEBUG - cudnnDestroyFusedOpsConstParamPack Executed"<<endl;
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFusedOpsConstParamPackAttribute){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFusedOpsConstParamPackAttribute"));

    cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
    cudnnFusedOpsConstParamLabel_t paramLabel = in->Get<cudnnFusedOpsConstParamLabel_t>();
    void *param = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param);

    
     LOG4CPLUS_DEBUG(logger, "cudnnSetFusedOpsConstParamPackAttribute Executed");
    //cout << " DEBUG - cudnnSetFusedOpsConstParamPackAttribute Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFusedOpsConstParamPackAttribute){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFusedOpsConstParamPackAttribute"));

     cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
     cudnnFusedOpsConstParamLabel_t paramLabel = in->Get<cudnnFusedOpsConstParamLabel_t>();
     void *param = in->Assign<void>();
     int isNULL = in->Get<int>();

     cudnnStatus_t cs = cudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param, &isNULL);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<int>(isNULL);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFusedOpsConstParamPackAttribute Executed");
    //cout << " DEBUG - cudnnGetFusedOpsConstParamPackAttribute Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFusedOpsVariantParamPack){
      Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFusedOpsVariantParamPack"));

      cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();
      cudnnFusedOps_t ops = in->Get<cudnnFusedOps_t>();

      cudnnStatus_t cs = cudnnCreateFusedOpsVariantParamPack(&varPack, ops);

      
       LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsVariantParamPack Executed");
      //cout << " DEBUG - cudnnCreateFusedOpsVariantParamPack Executed"<<endl;
      return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(DestroyFusedOpsVariantParamPack){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyFusedOpsVariantParamPack"));

     cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();

     cudnnStatus_t cs = cudnnDestroyFusedOpsVariantParamPack(varPack);

     
     LOG4CPLUS_DEBUG(logger, "cudnnDestroyFusedOpsVariantParamPack Executed");
     //cout << " DEBUG - cudnnDestroyFusedOpsVariantParamPack Executed"<<endl;
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFusedOpsVariantParamPackAttribute){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFusedOpsVariantParamPackAttribute"));

    cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();
    cudnnFusedOpsVariantParamLabel_t paramLabel = in->Get<cudnnFusedOpsVariantParamLabel_t>();
    void *ptr = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);

     LOG4CPLUS_DEBUG(logger, "cudnnSetFusedOpsVariantParamPackAttribute Executed");   
    //cout << " DEBUG - cudnnSetFusedOpsVariantParamPackAttribute Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFusedOpsVariantParamPackAttribute){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFusedOpsVariantParamPackAttribute"));

    cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();
    cudnnFusedOpsVariantParamLabel_t paramLabel = in->Get<cudnnFusedOpsVariantParamLabel_t>();
    void *ptr;

    cudnnStatus_t cs = cudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<void>(ptr);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFusedOpsVariantParamPackAttribute Executed");   
    //cout << " DEBUG - cudnnGetFusedOpsVariantParamPackAttribute Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFusedOpsPlan){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFusedOpsPlan"));

     cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
     cudnnFusedOps_t ops = in->Get<cudnnFusedOps_t>();

     cudnnStatus_t cs = cudnnCreateFusedOpsPlan(&plan, ops);

     LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsPlan Executed");    
     //cout << " DEBUG - cudnnCreateFusedOpsPlan Executed"<<endl;
     return std::make_shared<Result>(cs);
}


CUDNN_ROUTINE_HANDLER(DestroyFusedOpsPlan){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyFusedOpsPlan"));

     cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
 
    cudnnStatus_t cs = cudnnDestroyFusedOpsPlan(plan);

    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsPlan Executed"); 
    //cout << " DEBUG - cudnnCreateFusedOpsPlan Executed"<<endl;
    return std::make_shared<Result>(cs);   
}

CUDNN_ROUTINE_HANDLER(MakeFusedOpsPlan){
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MakeFusedOpsPlan"));

     cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
     cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
     cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
     size_t workspaceSizeInBytes;

     cudnnStatus_t cs = cudnnMakeFusedOpsPlan(handle, plan, constPack, &workspaceSizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<size_t>(workspaceSizeInBytes);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnMakeFusedOpsPlan Executed");
    //cout << " DEBUG - cudnnMakeFusedOpsPlan Executed"<<endl;
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(FusedOpsExecute){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FusedOpsExecute"));

    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
    cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();

    cudnnStatus_t cs = cudnnFusedOpsExecute(handle, plan, varPack);

    LOG4CPLUS_DEBUG(logger, "cudnnFusedOpsExecute Executed"); 
    //cout << " DEBUG - cudnnFusedOpsExecute Executed"<<endl;
    return std::make_shared<Result>(cs);   
}
/*
CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v6){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDescriptor_v6"));

   cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
   cudnnRNNDescriptor_t rnnDesc = (cudnnRNNDescriptor_t)in->Get<long long int>(); //INPUT/OUTPUT
   int hiddenSize = in->Get<int>();
   int numLayers = in->Get<int>();
   cudnnDropoutDescriptor_t dropoutDesc = (cudnnDropoutDescriptor_t)in->Get<long long int>();
   cudnnRNNInputMode_t inputMode = in->Get<cudnnRNNInputMode_t>();
   cudnnDirectionMode_t direction = in->Get<cudnnDirectionMode_t>();
   cudnnRNNMode_t mode = in->Get<cudnnRNNMode_t>();
   cudnnRNNAlgo_t algo = in->Get<cudnnRNNAlgo_t>();
   cudnnDataType_t mathPrec = in->Get<cudnnDataType_t>();

   cudnnStatus_t cs = cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try{
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch(string e){
         LOG4CPLUS_DEBUG(logger, e);
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDescriptor_v6 Executed");
    //cout << " DEBUG - cudnnSetRNNDescriptor_v6 Executed"<<endl;
    return std::make_shared<Result>(cs, out);
}
*/
