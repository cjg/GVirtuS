//
// Created by Raffaele Montella on 06/11/21.
//

#ifndef GVIRTUS_CUDART_INTERNAL_H
#define GVIRTUS_CUDART_INTERNAL_H

#include <vector>

#define EIFMT_NVAL 0x01
#define EIFMT_HVAL 0x03
#define EIFMT_SVAL 0x04

#define EIATTR_ERROR 0x00
#define EIATTR_PAD 0x01
#define EIATTR_IMAGE_SLOT 0x02
#define EIATTR_JUMPTABLE_RELOCS 0x03
#define EIATTR_CTAIDZ_USED 0x04
#define EIATTR_MAX_THREADS 0x05
#define EIATTR_IMAGE_OFFSET 0x06
#define EIATTR_IMAGE_SIZE 0x07
#define EIATTR_TEXTURE_NORMALIZED 0x08
#define EIATTR_SAMPLER_INIT 0x09
#define EIATTR_PARAM_CBANK 0x0a
#define EIATTR_SMEM_PARAM_OFFSETS 0x0b
#define EIATTR_CBANK_PARAM_OFFSETS 0x0c
#define EIATTR_SYNC_STACK 0x0d
#define EIATTR_TEXID_SAMPID_MAP 0x0e
#define EIATTR_EXTERNS 0x0f
#define EIATTR_REQNTID 0x10
#define EIATTR_FRAME_SIZE 0x11
#define EIATTR_MIN_STACK_SIZE 0x12
#define EIATTR_SAMPLER_FORCE_UNNORMALIZED 0x13
#define EIATTR_BINDLESS_IMAGE_OFFSETS 0x14
#define EIATTR_BINDLESS_TEXTURE_BANK 0x15
#define EIATTR_BINDLESS_SURFACE_BANK 0x16
#define EIATTR_KPARAM_INFO 0x17
#define EIATTR_SMEM_PARAM_SIZE 0x18
#define EIATTR_CBANK_PARAM_SIZE 0x19
#define EIATTR_QUERY_NUMATTRIB 0x1a
#define EIATTR_MAXREG_COUNT 0x1b
#define EIATTR_EXIT_INSTR_OFFSETS 0x1c
#define EIATTR_S2RCTAID_INSTR_OFFSETS 0x1d
#define EIATTR_CRS_STACK_SIZE 0x1e
#define EIATTR_NEED_CNP_WRAPPER 0x1f
#define EIATTR_NEED_CNP_PATCH 0x20
#define EIATTR_EXPLICIT_CACHING 0x21
#define EIATTR_ISTYPEP_USED 0x22
#define EIATTR_MAX_STACK_SIZE 0x23
#define EIATTR_SUQ_USED 0x24
#define EIATTR_LD_CACHEMOD_INSTR_OFFSETS 0x25
#define EIATTR_LOAD_CACHE_REQUEST 0x26
#define EIATTR_ATOM_SYS_INSTR_OFFSETS 0x27
#define EIATTR_COOP_GROUP_INSTR_OFFSETS 0x28
#define EIATTR_COOP_GROUP_MASK_REGIDS 0x29
#define EIATTR_SW1850030_WAR 0x2a
#define EIATTR_WMMA_USED 0x2b
#define EIATTR_SW2393858_WAR 0x30
#define EIATTR_CUDA_API_VERSION 0x37

#include <elf.h>

typedef struct {
    uint8_t dummy1[5*16];
    Elf64_Ehdr elf;
} NvFatCubin;

typedef struct {
    uint8_t fmt;
    uint8_t attr;
    uint16_t value;
} NvInfoAttribute;

typedef struct {
    NvInfoAttribute nvInfoAttribute;
    uint16_t index;
    uint16_t align;
    uint16_t ordinal;
    uint16_t offset;
    uint16_t a;
    uint8_t size;
    uint8_t b;
} NvInfoKParam;



typedef struct __infoFunction {
    std::vector<NvInfoKParam> params;
} NvInfoFunction;







#endif //GVIRTUS_CUDART_INTERNAL_H
