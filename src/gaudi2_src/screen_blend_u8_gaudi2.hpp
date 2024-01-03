#ifndef _SCREEN_BLEND_U8_GAUDI2_HPP
#define _SCREEN_BLEND_U8_GAUDI2_HPP

#include "gc_interface.h"

class ScreenBlendU8Gaudi2
{
    public:
        ScreenBlendU8Gaudi2() {}
        virtual ~ScreenBlendU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        ScreenBlendU8Gaudi2(const ScreenBlendU8Gaudi2& other) = delete;
        ScreenBlendU8Gaudi2& operator=(const ScreenBlendU8Gaudi2& other) = delete;
};

#endif
