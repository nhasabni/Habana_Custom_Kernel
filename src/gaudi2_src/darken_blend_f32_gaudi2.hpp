#ifndef _DARKEN_BLEND_F32_GAUDI2_HPP
#define _DARKEN_BLEND_F32_GAUDI2_HPP

#include "gc_interface.h"

class DarkenBlendF32Gaudi2
{
    public:
        DarkenBlendF32Gaudi2() {}
        virtual ~DarkenBlendF32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t* in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        DarkenBlendF32Gaudi2(const DarkenBlendF32Gaudi2& other) = delete;
        DarkenBlendF32Gaudi2& operator=(const DarkenBlendF32Gaudi2& other) = delete;
};

#endif
