#ifndef _LIGHTEN_BLEND_U8_GAUDI2_HPP
#define _LIGHTEN_BLEND_U8_GAUDI2_HPP

#include "gc_interface.h"

class LightenBlendU8Gaudi2
{
    public:
        LightenBlendU8Gaudi2() {}
        virtual ~LightenBlendU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t* in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        LightenBlendU8Gaudi2(const LightenBlendU8Gaudi2& other) = delete;
        LightenBlendU8Gaudi2& operator=(const LightenBlendU8Gaudi2& other) = delete;
};

#endif
