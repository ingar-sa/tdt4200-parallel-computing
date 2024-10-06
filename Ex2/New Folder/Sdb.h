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

#define sdb_internal static
#define sdb_persist  static
#define sdb_global   static

// TODO(ingar): Add support for custom errno codes
/* 0 is defined as success; negative errno code otherwise. */
typedef int_least32_t sdb_errno;

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

// NOTE(ingar): DO NOT USE THESE OUTSIDE OF THIS FILE! Since we are working with embedded systems
// floats and doubles may not necessarily be 32-bit and 64-bit respectively
typedef float  f32;
typedef double f64;

#define SDB_EXPAND(x)       x
#define SDB__STRINGIFY__(x) #x
#define SDB_STRINGIFY(x)    SDB__STRINGIFY__(x)

#define SDB__CONCAT2__(x, y) x##y
#define SDB_CONCAT2(x, y)    SDB__CONCAT2__(x, y)

#define SDB__CONCAT3__(x, y, z) x##y##z
#define SDB_CONCAT3(x, y, z)    SDB__CONCAT3__(x, y, z)

#define SdbKiloByte(Number) (Number * 1000ULL)
#define SdbMegaByte(Number) (SdbKiloByte(Number) * 1000ULL)
#define SdbGigaByte(Number) (SdbMegaByte(Number) * 1000ULL)
#define SdbTeraByte(Number) (SdbGigaByte(Number) * 1000ULL)

#define SdbKibiByte(Number) (Number * 1024ULL)
#define SdbMebiByte(Number) (SdbKibiByte(Number) * 1024ULL)
#define SdbGibiByte(Number) (SdbMebiByte(Number) * 1024ULL)
#define SdbTebiByte(Number) (SdbGibiByte(Number) * 1024ULL)

#define SdbArrayLen(Array) (sizeof(Array) / sizeof(Array[0]))

#define SdbMax(a, b) ((a > b) ? a : b)
#define SdbMin(a, b) ((a < b) ? a : b)

#if !defined(SDB_LOG_BUF_SIZE)
#define SDB_LOG_BUF_SIZE 1024
#endif

typedef struct sdb__log_module__
{
    const char *Name;
    u64         BufferSize;
    char       *Buffer;
} sdb__log_module__;

i64
Sdb__WriteLog__(sdb__log_module__ *Module, const char *LogLevel, va_list VaArgs)
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
        // NOTE(ingar): Since the buffer size is at least 128, this should never happen
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
        // NOTE(ingar): If the log module name is so long that it does not fit in 128 bytes - the
        // time stamp, it should be changed
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

sdb_errno
Sdb__WriteLogIntermediate__(sdb__log_module__ *Module, const char *LogLevel, ...)
{
    va_list VaArgs;
    va_start(VaArgs, LogLevel);
    sdb_errno Ret = Sdb__WriteLog__(Module, LogLevel, VaArgs);
    va_end(VaArgs);
    return Ret;
}

sdb_errno
Sdb__WriteLogNoModule__(const char *LogLevel, const char *FunctionName, ...)
{
    char             *Buffer = (char *)calloc(SDB_LOG_BUF_SIZE, sizeof(char));
    sdb__log_module__ Module = {
        .Name       = FunctionName,
        .BufferSize = SDB_LOG_BUF_SIZE,
        .Buffer     = Buffer,
    };

    va_list VaArgs;
    va_start(VaArgs, FunctionName);
    sdb_errno Ret = Sdb__WriteLog__(&Module, LogLevel, VaArgs);
    va_end(VaArgs);

    free(Buffer);

    return Ret;
}

#if !defined(SDB_LOG_LEVEL)
#define SDB_LOG_LEVEL 3
#endif

#define SDB_LOG_LEVEL_NONE (0U)
#define SDB_LOG_LEVEL_ERR  (1U)
#define SDB_LOG_LEVEL_INF  (3U)
#define SDB_LOG_LEVEL_DBG  (4U)

#define SDB__LOG_LEVEL_CHECK__(level) (SDB_LOG_LEVEL >= SDB_LOG_LEVEL_##level)

#define SDB_LOG_REGISTER(module_name)                                                              \
    sdb_global char SDB_CONCAT3(Sdb__LogModule, module_name, Buffer__)[SDB_LOG_BUF_SIZE];          \
    sdb_global sdb__log_module__ SDB_CONCAT3(Sdb__LogModule, module_name, __)                      \
        = { .Name       = #module_name,                                                            \
            .BufferSize = SDB_LOG_BUF_SIZE,                                                        \
            .Buffer     = SDB_CONCAT3(Sdb__LogModule, module_name, Buffer__) };                        \
    sdb_global sdb__log_module__ *Sdb__LogInstance__ = &SDB_CONCAT3(Sdb__LogModule, module_name, __)

#define SDB_LOG_DECLARE_EXTERN(name)                                                               \
    extern sdb__log_module__      SDB_CONCAT3(Sdb__LogModule, name, __);                           \
    sdb_global sdb__log_module__ *Sdb__LogInstance__ = &SDB_CONCAT3(Sdb__LogModule, name, __)

#define SDB_LOG_DECLARE_SAME_TU extern struct sdb__log_module__ *Sdb__LogInsta

#define SDB__LOG__(log_level, ...)                                                                 \
    do {                                                                                           \
        if(SDB__LOG_LEVEL_CHECK__(log_level)) {                                                    \
            sdb_errno LogRet = Sdb__WriteLogIntermediate__(Sdb__LogInstance__,                     \
                                                           SDB_STRINGIFY(log_level), __VA_ARGS__); \
            assert(LogRet >= 0);                                                                   \
        }                                                                                          \
    } while(0)

#define SDB__LOG_NO_MODULE__(log_level, ...)                                                       \
    do {                                                                                           \
        if(SDB__LOG_LEVEL_CHECK__(log_level)) {                                                    \
            sdb_errno LogRet                                                                       \
                = Sdb__WriteLogNoModule__(SDB_STRINGIFY(log_level), __func__, __VA_ARGS__);        \
            assert(LogRet >= 0);                                                                   \
        }                                                                                          \
    } while(0)

#define SdbLogDebug(...)   SDB__LOG__(DBG, __VA_ARGS__)
#define SdbLogInfo(...)    SDB__LOG__(INF, __VA_ARGS__)
#define SdbLogWarning(...) SDB__LOG__(WRN, __VA_ARGS__)
#define SdbLogError(...)   SDB__LOG__(ERR, __VA_ARGS__)

// WARN: Uses calloc!
#define SdbLogDebugNoModule(...)   SDB__LOG_NO_MODULE__(DBG, __VA_ARGS__)
#define SdbLogInfoNoModule(...)    SDB__LOG_NO_MODULE__(INF, __VA_ARGS__)
#define SdbLogWarningNoModule(...) SDB__LOG_NO_MODULE__(WRN, __VA_ARGS__)
#define SdbLogErrorNoModule(...)   SDB__LOG_NO_MODULE__(ERR, __VA_ARGS__)

#define SdbAssert(condition)                                                                       \
    do {                                                                                           \
        if(!(condition)) {                                                                         \
            SdbLogError("Assertion failed: " SDB_STRINGIFY(condition));                            \
            assert(condition);                                                                     \
        }                                                                                          \
    } while(0)
