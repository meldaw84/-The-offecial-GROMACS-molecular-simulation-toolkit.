
#if GMX_GPU_CUDA
#    define USE_NVTX 1
#endif // GMX_GPU_CUDA

#if defined(USE_NVTX) && USE_NVTX

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

#elif defined(USE_ROCTX) && USE_ROCTX

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


#else

gmx_unused static void traceRangeStart(gmx_unused const char* rangeName, gmx_unused int rangeId) {}
gmx_unused static void traceSubRangeStart(gmx_unused const char* rangeName, gmx_unused int rangeId)
{
}

gmx_unused static void traceRangeEnd() {}
gmx_unused static void traceSubRangeEnd() {}


#endif
