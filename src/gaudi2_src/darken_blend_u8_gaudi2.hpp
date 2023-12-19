#ifndef _DARKEN_BLEND_U8_GAUDI2_HPP
#define _DARKEN_BLEND_U8_GAUDI2_HPP

#include "gc_interface.h"

class DarkenBlendU8Gaudi2
{
    public:
        DarkenBlendU8Gaudi2() {}
        virtual ~DarkenBlendU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t* in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        DarkenBlendU8Gaudi2(const DarkenBlendU8Gaudi2& other) = delete;
        DarkenBlendU8Gaudi2& operator=(const DarkenBlendU8Gaudi2& other) = delete;
};

#endif
