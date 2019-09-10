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
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include "CudaRtHandler.h"

using namespace log4cplus;
/*
void setLogLevel(Logger *logger) {
  log4cplus::LogLevel logLevel=log4cplus::INFO_LOG_LEVEL;
  char * val = getenv("GVIRTUS_LOGLEVEL" );
  std::string logLevelString=(val == NULL ? std::string("") : std::string(val));
  if (logLevelString!="") {
      logLevel=std::stoi(logLevelString);
  }
  logger->setLogLevel(logLevel);
}
*/
CUDA_ROUTINE_HANDLER(DeviceSetCacheConfig) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceSetCacheConfig"));
    CudaRtHandler::setLogLevel(&logger);

    try {
        cudaFuncCache cacheConfig = input_buffer->Get<cudaFuncCache>();
        cudaError_t exit_code = cudaDeviceSetCacheConfig(cacheConfig);
        return std::make_shared<Result>(exit_code);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation); //???
    }
}

CUDA_ROUTINE_HANDLER(DeviceSetLimit) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceSetLimit"));
    CudaRtHandler::setLogLevel(&logger);
    try {
        cudaLimit limit = input_buffer->Get<cudaLimit>();
        size_t value = input_buffer->Get<size_t>();
        cudaError_t exit_code = cudaDeviceSetLimit(limit, value);
        return std::make_shared<Result>(exit_code);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation); //???
    }
}

CUDA_ROUTINE_HANDLER(IpcOpenMemHandle) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("IpcOpenMemHandle"));
    CudaRtHandler::setLogLevel(&logger);
    void *devPtr = NULL;
    try {
        cudaIpcMemHandle_t handle = input_buffer->Get<cudaIpcMemHandle_t>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        cudaError_t exit_code = cudaIpcOpenMemHandle(&devPtr, handle, flags);
            std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->AddMarshal(devPtr);
        return std::make_shared<Result>(exit_code, out);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(DeviceEnablePeerAccess) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceEnablePeerAccess"));
    CudaRtHandler::setLogLevel(&logger);
    int peerDevice = input_buffer->Get<int>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    cudaError_t exit_code = cudaDeviceEnablePeerAccess(peerDevice, flags);
    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(DeviceDisablePeerAccess) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceDisablePeerAccess"));
    CudaRtHandler::setLogLevel(&logger);
    int peerDevice = input_buffer->Get<int>();
    cudaError_t exit_code = cudaDeviceDisablePeerAccess(peerDevice);
    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(DeviceCanAccessPeer) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceCanAccessPeer"));
    CudaRtHandler::setLogLevel(&logger);
    int *canAccessPeer = input_buffer->Assign<int>();
    int device = input_buffer->Get<int>();
    int peerDevice = input_buffer->Get<int>();

    cudaError_t exit_code = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(canAccessPeer);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceGetStreamPriorityRange) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetStreamPriorityRange"));
    CudaRtHandler::setLogLevel(&logger);

    int *leastPriority = input_buffer->Assign<int>();
    int *greatestPriority = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(leastPriority);
        out->Add(greatestPriority);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceGetAttribute) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetAttribute"));
    CudaRtHandler::setLogLevel(&logger);

    int *value = input_buffer->Assign<int>();
    cudaDeviceAttr attr = input_buffer->Get<cudaDeviceAttr>();
    int device = input_buffer->Get<int>();
    cudaError_t exit_code = cudaDeviceGetAttribute(value, attr, device);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(value);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(IpcGetMemHandle) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("IpcGetMemHanlde"));
    CudaRtHandler::setLogLevel(&logger);

    cudaIpcMemHandle_t *handle = input_buffer->Assign<cudaIpcMemHandle_t>();
    void *devPtr = input_buffer->GetFromMarshal<void *>();

    cudaError_t exit_code = cudaIpcGetMemHandle(handle, devPtr);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(handle);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(IpcGetEventHandle) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("IpcGetEventHandle"));
    CudaRtHandler::setLogLevel(&logger);

    cudaIpcEventHandle_t *handle = input_buffer->Assign<cudaIpcEventHandle_t>();
    cudaEvent_t event = input_buffer->Get<cudaEvent_t>();

    cudaError_t exit_code = cudaIpcGetEventHandle(handle, event);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(handle);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(ChooseDevice) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ChooseDevice"));
    CudaRtHandler::setLogLevel(&logger);

    int *device = input_buffer->Assign<int>();
    const cudaDeviceProp *prop = input_buffer->Assign<cudaDeviceProp>();
    cudaError_t exit_code = cudaChooseDevice(device, prop);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(device);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDevice) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetDevice"));
    CudaRtHandler::setLogLevel(&logger);

    try {
        int *device = input_buffer->Assign<int>();
        cudaError_t exit_code = cudaGetDevice(device);
            std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(device);
        return std::make_shared<Result>(exit_code, out);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(DeviceReset) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceReset"));
    CudaRtHandler::setLogLevel(&logger);

    cudaError_t exit_code = cudaDeviceReset();
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceSynchronize) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("DeviceSynchronize"));
    CudaRtHandler::setLogLevel(&logger);

    cudaError_t exit_code = cudaDeviceSynchronize();

    try {
            std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        return std::make_shared<Result>(exit_code, out);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(GetDeviceCount) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetDeviceCount"));
    CudaRtHandler::setLogLevel(&logger);

    try {
        int *count = input_buffer->Assign<int>();
        cudaError_t exit_code = cudaGetDeviceCount(count);
            std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(count);
        return std::make_shared<Result>(exit_code, out);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(GetDeviceProperties) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetDeviceProperties"));
    CudaRtHandler::setLogLevel(&logger);

    try {
        struct cudaDeviceProp *prop = input_buffer->Assign<struct cudaDeviceProp>();
        int device = input_buffer->Get<int>();
        cudaError_t exit_code = cudaGetDeviceProperties(prop, device);
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
        prop->canMapHostMemory = 0;
#endif
            std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(prop, 1);
        return std::make_shared<Result>(exit_code, out);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(SetDevice) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetDevice"));
    CudaRtHandler::setLogLevel(&logger);

    try {
        int device = input_buffer->Get<int>();
        cudaError_t exit_code = cudaSetDevice(device);
        return std::make_shared<Result>(exit_code);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030

CUDA_ROUTINE_HANDLER(SetDeviceFlags) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetDeviceFlags"));
    CudaRtHandler::setLogLevel(&logger);

    try {
        int flags = input_buffer->Get<int>();
        cudaError_t exit_code = cudaSetDeviceFlags(flags);
        return std::make_shared<Result>(exit_code);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(IpcOpenEventHandle) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("IpcOpenEventHandler"));
    CudaRtHandler::setLogLevel(&logger);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        cudaEvent_t *event = input_buffer->Assign<cudaEvent_t>();
        cudaIpcEventHandle_t handle = input_buffer->Get<cudaIpcEventHandle_t>();
        cudaError_t exit_code = cudaIpcOpenEventHandle(event, handle);
        out->Add(event);
    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(SetValidDevices) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetValidDevice"));
    CudaRtHandler::setLogLevel(&logger);

    try {
        int len = input_buffer->BackGet<int>();
        int *device_arr = input_buffer->Assign<int>(len);
        cudaError_t exit_code = cudaSetValidDevices(device_arr, len);
            std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(device_arr, len);
        return std::make_shared<Result>(exit_code, out);

    } catch (string e) {
        //cerr << e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

}
#endif

