#ifndef _NORMAL_BLEND_F32_GAUDI2_HPP
#define _NORMAL_BLEND_F32_GAUDI2_HPP

#include "gc_interface.h"

class NormalBlendF32Gaudi2
{
    public:
        NormalBlendF32Gaudi2() {}
        virtual ~NormalBlendF32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

        struct NormalBlendParam {
            float opacity;
        };
    private:
        NormalBlendF32Gaudi2(const NormalBlendF32Gaudi2& other) = delete;
        NormalBlendF32Gaudi2& operator=(const NormalBlendF32Gaudi2& other) = delete;
};

#endif
