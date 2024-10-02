#ifndef ISA_EXTERN_H_
#define ISA_EXTERN_H_

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

#define isa_internal static
#define isa_persist  static
#define isa_global   static

typedef int_least32_t isa_errno;

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float  f32;
typedef double f64;

#define ISA_EXPAND(x)       x
#define ISA__STRINGIFY__(x) #x
#define ISA_STRINGIFY(x)    ISA__STRINGIFY__(x)

#define ISA__CONCAT2__(x, y) x##y
#define ISA_CONCAT2(x, y)    ISA__CONCAT2__(x, y)

#define ISA__CONCAT3__(x, y, z) x##y##z
#define ISA_CONCAT3(x, y, z)    ISA__CONCAT3__(x, y, z)

#define IsaKiloByte(Number) (Number * 1000ULL)
#define IsaMegaByte(Number) (IsaKiloByte(Number) * 1000ULL)
#define IsaGigaByte(Number) (IsaMegaByte(Number) * 1000ULL)
#define IsaTeraByte(Number) (IsaGigaByte(Number) * 1000ULL)

#define IsaKibiByte(Number) (Number * 1024ULL)
#define IsaMebiByte(Number) (IsaKibiByte(Number) * 1024ULL)
#define IsaGibiByte(Number) (IsaMebiByte(Number) * 1024ULL)
#define IsaTebiByte(Number) (IsaGibiByte(Number) * 1024ULL)

#define IsaArrayLen(Array) (sizeof(Array) / sizeof(Array[0]))

#define IsaMax(a, b) ((a > b) ? a : b)
#define IsaMin(a, b) ((a < b) ? a : b)

#if !defined(ISA_LOG_BUF_SIZE)
#define ISA_LOG_BUF_SIZE 1024
#endif

typedef struct isa__log_module__
{
    const char *Name;
    u64         BufferSize;
    char       *Buffer;
} isa__log_module__;

i64
Isa__WriteLog__(isa__log_module__ *Module, const char *LogLevel, va_list VaArgs)
{
    time_t    PosixTime;
    struct tm TimeInfo;

    if((time_t)(-1) == time(&PosixTime)) {
        return -errno;
    }

    if(NULL == localtime_r(&PosixTime, &TimeInfo)) {
        return -errno;
    }

    u64 CharsWritten    = 0;
    u64 BufferRemaining = Module->BufferSize;

    int FormatRet = strftime(Module->Buffer, BufferRemaining, "%T: ", &TimeInfo);
    if(0 == FormatRet) {
        assert(FormatRet);
        return -ENOMEM;
    }

    CharsWritten = FormatRet;
    BufferRemaining -= FormatRet;

    FormatRet = snprintf(Module->Buffer + CharsWritten, BufferRemaining, "%s: %s: ", Module->Name,
                         LogLevel);
    if(FormatRet < 0) {
        return -errno;
    } else if((u64)FormatRet >= BufferRemaining) {
        assert(FormatRet);
        return -ENOMEM;
    }

    CharsWritten += FormatRet;
    BufferRemaining -= FormatRet;

    const char *FormatString = va_arg(VaArgs, const char *);
    FormatRet = vsnprintf(Module->Buffer + CharsWritten, BufferRemaining, FormatString, VaArgs);

    if(FormatRet < 0) {
        return -errno;
    } else if((u64)FormatRet >= BufferRemaining) {
        (void)memset(Module->Buffer + CharsWritten, 0, BufferRemaining);
        FormatRet = snprintf(Module->Buffer + CharsWritten, BufferRemaining, "%s",
                             "Message dropped; too big");
        if(FormatRet < 0) {
            return -errno;
        } else if((u64)FormatRet >= BufferRemaining) {
            assert(FormatRet);
            return -ENOMEM;
        }
    }

    CharsWritten += FormatRet;
    Module->Buffer[CharsWritten++] = '\n';

    int OutFd;
    if('E' == LogLevel[0]) {
        OutFd = STDERR_FILENO;
    } else {
        OutFd = STDOUT_FILENO;
    }

    if((ssize_t)(-1) == write(OutFd, Module->Buffer, CharsWritten)) {
        return -errno;
    }

    return 0;
}

isa_errno
Isa__WriteLogIntermediate__(isa__log_module__ *Module, const char *LogLevel, ...)
{
    va_list VaArgs;
    va_start(VaArgs, LogLevel);
    isa_errno Ret = Isa__WriteLog__(Module, LogLevel, VaArgs);
    va_end(VaArgs);
    return Ret;
}

isa_errno
Isa__WriteLogNoModule__(const char *LogLevel, const char *FunctionName, ...)
{
    char             *Buffer = (char *)calloc(ISA_LOG_BUF_SIZE, sizeof(char));
    isa__log_module__ Module = {
        .Name       = FunctionName,
        .BufferSize = ISA_LOG_BUF_SIZE,
        .Buffer     = Buffer,
    };

    va_list VaArgs;
    va_start(VaArgs, FunctionName);
    isa_errno Ret = Isa__WriteLog__(&Module, LogLevel, VaArgs);
    va_end(VaArgs);

    free(Buffer);

    return Ret;
}

#if !defined(ISA_LOG_LEVEL)
#define ISA_LOG_LEVEL 3
#endif

#define ISA_LOG_LEVEL_NONE (0U)
#define ISA_LOG_LEVEL_ERR  (1U)
#define ISA_LOG_LEVEL_INF  (3U)
#define ISA_LOG_LEVEL_DBG  (4U)

#define ISA__LOG_LEVEL_CHECK__(level) (ISA_LOG_LEVEL >= ISA_LOG_LEVEL_##level)

#define ISA_LOG_REGISTER(module_name)                                                              \
    isa_global char ISA_CONCAT3(Isa__LogModule, module_name, Buffer__)[ISA_LOG_BUF_SIZE];          \
    isa_global isa__log_module__ ISA_CONCAT3(Isa__LogModule, module_name, __)                      \
        = { .Name       = #module_name,                                                            \
            .BufferSize = ISA_LOG_BUF_SIZE,                                                        \
            .Buffer     = ISA_CONCAT3(Isa__LogModule, module_name, Buffer__) };                        \
    isa_global isa__log_module__ *Isa__LogInstance__ = &ISA_CONCAT3(Isa__LogModule, module_name, __)

#define ISA_LOG_DECLARE_EXTERN(name)                                                               \
    extern isa__log_module__      ISA_CONCAT3(Isa__LogModule, name, __);                           \
    isa_global isa__log_module__ *Isa__LogInstance__ = &ISA_CONCAT3(Isa__LogModule, name, __)

#define ISA_LOG_DECLARE_SAME_TU extern struct isa__log_module__ *Isa__LogInsta

#define ISA__LOG__(log_level, ...)                                                                 \
    do {                                                                                           \
        if(ISA__LOG_LEVEL_CHECK__(log_level)) {                                                    \
            isa_errno LogRet = Isa__WriteLogIntermediate__(Isa__LogInstance__,                     \
                                                           ISA_STRINGIFY(log_level), __VA_ARGS__); \
            assert(LogRet >= 0);                                                                   \
        }                                                                                          \
    } while(0)

#define ISA__LOG_NO_MODULE__(log_level, ...)                                                       \
    do {                                                                                           \
        if(ISA__LOG_LEVEL_CHECK__(log_level)) {                                                    \
            isa_errno LogRet                                                                       \
                = Isa__WriteLogNoModule__(ISA_STRINGIFY(log_level), __func__, __VA_ARGS__);        \
            assert(LogRet >= 0);                                                                   \
        }                                                                                          \
    } while(0)

#define IsaLogDebug(...)   ISA__LOG__(DBG, __VA_ARGS__)
#define IsaLogInfo(...)    ISA__LOG__(INF, __VA_ARGS__)
#define IsaLogWarning(...) ISA__LOG__(WRN, __VA_ARGS__)
#define IsaLogError(...)   ISA__LOG__(ERR, __VA_ARGS__)

#endif
