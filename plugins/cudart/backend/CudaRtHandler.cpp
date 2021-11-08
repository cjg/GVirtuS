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

/**
 * @file   Backend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 *
 * @brief
 *
 *
 */
//#define DEBUG
#include "CudaRtHandler.h"

#include <cstring>

#include <cuda_runtime_api.h>

#include "CudaUtil.h"

#include <dlfcn.h>

using namespace std;
using namespace log4cplus;

map<string, CudaRtHandler::CudaRoutineHandler> *CudaRtHandler::mspHandlers =
    NULL;

extern "C" std::shared_ptr<CudaRtHandler> create_t() {
  return std::make_shared<CudaRtHandler>();
}

CudaRtHandler::CudaRtHandler() {
  logger = Logger::getInstance(LOG4CPLUS_TEXT("CudaRtHandler"));
  setLogLevel(&logger);
  mpFatBinary = new map<string, void **>();
  mpDeviceFunction = new map<string, string>();
  mpVar = new map<string, string>();
  mpTexture = new map<string, textureReference *>();
  mpSurface = new map<string, surfaceReference *>();

  mapHost2DeviceFunc = new map<const void*, std::string>();
  mapDeviceFunc2InfoFunc = new map<std::string, NvInfoFunction>();
  Initialize();
}

CudaRtHandler::~CudaRtHandler() {}

void CudaRtHandler::setLogLevel(Logger *logger) {
  log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
  char *val = getenv("GVIRTUS_LOGLEVEL");
  std::string logLevelString =
      (val == NULL ? std::string("") : std::string(val));
  if (logLevelString != "") {
    logLevel = std::stoi(logLevelString);
  }
  logger->setLogLevel(logLevel);
}

bool CudaRtHandler::CanExecute(std::string routine) {
  map<string, CudaRtHandler::CudaRoutineHandler>::iterator it;
  it = mspHandlers->find(routine);
  if (it == mspHandlers->end()) return false;
  return true;
}

std::shared_ptr<Result> CudaRtHandler::Execute(
    std::string routine, std::shared_ptr<Buffer> input_buffer) {
  map<string, CudaRtHandler::CudaRoutineHandler>::iterator it;
  it = mspHandlers->find(routine);
  //#ifdef DEBUG
  //    cerr << "Requested: " << routine << endl;
  //#endif
  LOG4CPLUS_DEBUG(logger, "Called: " << routine);
  if (it == mspHandlers->end()) throw "No handler for '" + routine + "' found!";
  return it->second(this, input_buffer);
}

void CudaRtHandler::RegisterFatBinary(std::string &handler,
                                      void **fatCubinHandle) {
  map<string, void **>::iterator it = mpFatBinary->find(handler);
  if (it != mpFatBinary->end()) {
    mpFatBinary->erase(it);
  }
  mpFatBinary->insert(make_pair(handler, fatCubinHandle));
  //#ifdef DEBUG
  //    cout << "Registered FatBinary " << fatCubinHandle << " with handler " <<
  //    handler << endl;
  //#endif
  LOG4CPLUS_DEBUG(logger, "Registered FatBinary "
                              << fatCubinHandle << " with handler " << handler);
}

void CudaRtHandler::RegisterFatBinary(const char *handler,
                                      void **fatCubinHandle) {
  string tmp(handler);
  RegisterFatBinary(tmp, fatCubinHandle);
}

void **CudaRtHandler::GetFatBinary(string &handler) {
  map<string, void **>::iterator it = mpFatBinary->find(handler);
  if (it == mpFatBinary->end()) throw "Fat Binary '" + handler + "' not found";
  return it->second;
}

void **CudaRtHandler::GetFatBinary(const char *handler) {
  string tmp(handler);
  return GetFatBinary(tmp);
}

void CudaRtHandler::UnregisterFatBinary(std::string &handler) {
  map<string, void **>::iterator it = mpFatBinary->find(handler);
  if (it == mpFatBinary->end()) return;
  /* FIXME: think about freeing memory */
  //#ifdef DEBUG
  //    cout << "Unregistered FatBinary " << it->second << " with handler "<<
  //    handler << endl;
  //#endif
  LOG4CPLUS_DEBUG(logger, "Unregistered FatBinary "
                              << it->second << " with handler " << handler);
  mpFatBinary->erase(it);
}

void CudaRtHandler::UnregisterFatBinary(const char *handler) {
  string tmp(handler);
  UnregisterFatBinary(tmp);
}

void CudaRtHandler::RegisterDeviceFunction(std::string &handler,
                                           std::string &function) {
  map<string, string>::iterator it = mpDeviceFunction->find(handler);
  if (it != mpDeviceFunction->end()) mpDeviceFunction->erase(it);
  mpDeviceFunction->insert(make_pair(handler, function));
  //#ifdef DEBUG
  //    cout << "Registered DeviceFunction " << function << " with handler " <<
  //    handler << endl;
  //#endif
  LOG4CPLUS_DEBUG(logger, "Registered DeviceFunction "
                              << function << " with handler " << handler);
}

void CudaRtHandler::RegisterDeviceFunction(const char *handler,
                                           const char *function) {
  string tmp1(handler);
  string tmp2(function);
  RegisterDeviceFunction(tmp1, tmp2);
}

const char *CudaRtHandler::GetDeviceFunction(std::string &handler) {
  map<string, string>::iterator it = mpDeviceFunction->find(handler);
  if (it == mpDeviceFunction->end())
    throw "Device Function '" + handler + "' not found";
  return it->second.c_str();
}

const char *CudaRtHandler::GetDeviceFunction(const char *handler) {
  string tmp(handler);
  return GetDeviceFunction(tmp);
}

void CudaRtHandler::RegisterVar(string &handler, string &symbol) {
  mpVar->insert(make_pair(handler, symbol));
  //#ifdef DEBUG
  //    cout << "Registered Var " << symbol << " with handler " << handler <<
  //    endl;
  //#endif
  LOG4CPLUS_DEBUG(logger,
                  "Registered Var " << symbol << " with handler " << handler);
}

void CudaRtHandler::RegisterVar(const char *handler, const char *symbol) {
  string tmp1(handler);
  string tmp2(symbol);
  RegisterVar(tmp1, tmp2);
}

const char *CudaRtHandler::GetVar(string &handler) {
  map<string, string>::iterator it = mpVar->find(handler);
  if (it == mpVar->end()) return NULL;
  return it->second.c_str();
}

const char *CudaRtHandler::GetVar(const char *handler) {
  string tmp(handler);
  return GetVar(tmp);
}

void CudaRtHandler::RegisterTexture(string &handler, textureReference *texref) {
  mpTexture->insert(make_pair(handler, texref));
  //#ifdef DEBUG
  //    cout << "Registered Texture " << texref << " with handler " << handler<<
  //    endl;
  //#endif
  LOG4CPLUS_DEBUG(
      logger, "Registered Texture " << texref << " with handler " << handler);
}

void CudaRtHandler::RegisterTexture(const char *handler,
                                    textureReference *texref) {
  string tmp(handler);
  RegisterTexture(tmp, texref);
}

void CudaRtHandler::RegisterSurface(string &handler,
                                    surfaceReference *surfref) {
  mpSurface->insert(make_pair(handler, surfref));
  //#ifdef DEBUG
  //    cout << "Registered Surface " << surfref << " with handler " <<
  //    handler<< endl;
  //#endif
  LOG4CPLUS_DEBUG(
      logger, "Registered Surface " << surfref << " with handler " << handler);
}

void CudaRtHandler::RegisterSurface(const char *handler,
                                    surfaceReference *surfref) {
  string tmp(handler);
  RegisterSurface(tmp, surfref);
}

textureReference *CudaRtHandler::GetTexture(string &handler) {
  map<string, textureReference *>::iterator it = mpTexture->find(handler);
  if (it == mpTexture->end()) return NULL;
  return it->second;
}

textureReference *CudaRtHandler::GetTexture(const char *handler) {
  string tmp(handler);
  return GetTexture(tmp);
}

const char *CudaRtHandler::GetTextureHandler(textureReference *texref) {
  for (map<string, textureReference *>::iterator it = mpTexture->begin();
       it != mpTexture->end(); it++)
    if (it->second == texref) return it->first.c_str();
  return NULL;
}

surfaceReference *CudaRtHandler::GetSurface(string &handler) {
  map<string, surfaceReference *>::iterator it = mpSurface->find(handler);
  if (it == mpSurface->end()) return NULL;
  return it->second;
}

surfaceReference *CudaRtHandler::GetSurface(const char *handler) {
  string tmp(handler);
  return GetSurface(tmp);
}

const char *CudaRtHandler::GetSurfaceHandler(surfaceReference *surfref) {
  for (map<string, surfaceReference *>::iterator it = mpSurface->begin();
       it != mpSurface->end(); it++)
    if (it->second == surfref) return it->first.c_str();
  return NULL;
}

const char *CudaRtHandler::GetSymbol(std::shared_ptr<Buffer> in) {
  char *symbol_handler = in->AssignString();
  char *symbol = in->AssignString();
  char *our_symbol = const_cast<char *>(GetVar(symbol_handler));
  if (our_symbol != NULL) symbol = const_cast<char *>(our_symbol);
  return symbol;
}

void CudaRtHandler::Initialize() {
  if (mspHandlers != NULL) return;
  mspHandlers = new map<string, CudaRtHandler::CudaRoutineHandler>();

  /* CudaRtHandler_device */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ChooseDevice));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDevice));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceCount));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceProperties));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDevice));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceReset));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceSynchronize));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceSetCacheConfig));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceSetLimit));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceCanAccessPeer));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceDisablePeerAccess));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceEnablePeerAccess));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcGetMemHandle));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcGetEventHandle));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcOpenEventHandle));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcOpenMemHandle));
  mspHandlers->insert(
      CUDA_ROUTINE_HANDLER_PAIR(OccupancyMaxActiveBlocksPerMultiprocessor));
#if (CUDART_VERSION >= 7000)
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(
      OccupancyMaxActiveBlocksPerMultiprocessorWithFlags));
#endif
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceGetAttribute));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceGetStreamPriorityRange));

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDeviceFlags));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetValidDevices));
#endif

  /* CudaRtHandler_error */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetErrorString));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetLastError));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(PeekAtLastError));

  /* CudaRtHandler_event */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventCreate));
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventCreateWithFlags));
#endif
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventDestroy));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventElapsedTime));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventQuery));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventRecord));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventSynchronize));

  /* CudaRtHandler_execution */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ConfigureCall));
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FuncGetAttributes));
#endif
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Launch));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDoubleForDevice));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDoubleForHost));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetupArgument));
#if CUDART_VERSION >= 9020
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(PushCallConfiguration));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(PopCallConfiguration));
#endif
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(LaunchKernel));

  /* CudaRtHandler_internal */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFatBinary));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFatBinaryEnd));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(UnregisterFatBinary));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFunction));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterVar));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterSharedVar));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterShared));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterTexture));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterSurface));

  /* CudaRtHandler_memory */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Free));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FreeArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetSymbolAddress));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetSymbolSize));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Malloc));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MallocArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MallocManaged));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MallocPitch));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy2D));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy3D));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyAsync));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyFromSymbol));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyToArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyToSymbol));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memset));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memset2D));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyFromArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyArrayToArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy2DFromArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy2DToArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Malloc3DArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyPeerAsync));

  /* CudaRtHandler_opengl */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GLSetGLDevice));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsGLRegisterBuffer));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsMapResources));
  mspHandlers->insert(
      CUDA_ROUTINE_HANDLER_PAIR(GraphicsResourceGetMappedPointer));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsUnmapResources));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsUnregisterResource));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsResourceSetMapFlags));

  /* CudaRtHandler_stream */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamCreate));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamDestroy));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamQuery));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamSynchronize));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamCreateWithFlags));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamWaitEvent));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamCreateWithPriority));

  /* CudaRtHandler_surface */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(BindSurfaceToArray));

  /* CudaRtHandler_texture */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(BindTexture));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(BindTexture2D));
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(BindTextureToArray));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(CreateTextureObject));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetChannelDesc));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetTextureAlignmentOffset));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetTextureReference));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(UnbindTexture));

  /* CudaRtHandler_thread */
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ThreadExit));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ThreadSynchronize));

  /* CudaRtHandler_version */
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DriverGetVersion));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RuntimeGetVersion));
  mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FuncSetCacheConfig));
#endif
}
