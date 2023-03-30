#include "gromacs/utility/basedefinitions.h"

#if GMX_GPU_CUDA
#    define USE_NVTX 1
#endif // GMX_GPU_CUDA

#if USE_NVTX

#    include "nvToolsExt.h"

const uint32_t rangeColors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                                 0xff00ffff, 0xffff0000, 0xffffffff };
const int      numColors     = sizeof(rangeColors) / sizeof(uint32_t);

const uint32_t subRangeColors[] = { 0x9900ff00, 0x990000ff, 0x99ffff00, 0x99ff00ff,
                                    0x9900ffff, 0x99ff0000, 0x99ffffff };
const int      numColorsSub     = sizeof(subRangeColors) / sizeof(uint32_t);

static void traceRangeStart(const char* rangeName, int rangeId)
{
    int                   colorId     = rangeId % numColors;
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = NVTX_COLOR_ARGB;
    eventAttrib.color                 = rangeColors[colorId];
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = rangeName;
    nvtxRangePushEx(&eventAttrib);
}

static void traceSubRangeStart(const char* rangeName, int rangeId)
{
    int                   colorId     = rangeId % numColors;
    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = NVTX_COLOR_ARGB;
    eventAttrib.color                 = subRangeColors[colorId];
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = rangeName;
    nvtxRangePushEx(&eventAttrib);
}

static void traceRangeEnd()
{
    nvtxRangePop();
}

static void traceSubRangeEnd()
{
    nvtxRangePop();
}

#elif USE_ROCTX

#    include "roctracer/roctx.h"

static void traceRangeStart(const char* rangeName, int /*rangeId*/)
{
    roctxRangePush(rangeName);
}

static void traceSubRangeStart(const char* rangeName, int /*rangeId*/)
{
    roctxRangePush(rangeName);
}

static void traceRangeEnd()
{
    roctxRangePop();
}

static void traceSubRangeEnd()
{
    roctxRangePop();
}

#elif USE_ITT

#    ifdef __clang__
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wold-style-cast"
#        pragma clang diagnostic ignored "-Wnewline-eof"
#    endif
#    include <ittnotify.h>
#    ifdef __clang__
#        pragma clang diagnostic pop
#    endif

// Defined in wallcycle.cpp, initialized in wallcycle_init
extern const __itt_domain*  g_ittDomain;
extern __itt_string_handle* g_ittCounterHandles[];
extern __itt_string_handle* g_ittSubCounterHandles[];

static void traceRangeStart(const char* /*rangeName*/, int rangeId)
{
    __itt_task_begin(g_ittDomain, __itt_null, __itt_null, g_ittCounterHandles[rangeId]);
}

static void traceSubRangeStart(const char* /*rangeName*/, int rangeId)
{
    __itt_task_begin(g_ittDomain, __itt_null, __itt_null, g_ittSubCounterHandles[rangeId]);
}

static void traceRangeEnd()
{
    __itt_task_end(g_ittDomain);
}

static void traceSubRangeEnd()
{
    __itt_task_end(g_ittDomain);
}

#else

gmx_unused static void traceRangeStart(gmx_unused const char* rangeName, gmx_unused int rangeId) {}
gmx_unused static void traceSubRangeStart(gmx_unused const char* rangeName, gmx_unused int rangeId)
{
}

gmx_unused static void traceRangeEnd() {}
gmx_unused static void traceSubRangeEnd() {}


#endif
