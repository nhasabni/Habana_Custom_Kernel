#ifndef _DISSOLVE_BLEND_F32_GAUDI2_HPP
#define _DISSOLVE_BLEND_F32_GAUDI2_HPP

#include "gc_interface.h"

class DissolveBlendF32Gaudi2
{
    public:
        DissolveBlendF32Gaudi2() {}
        virtual ~DissolveBlendF32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t* in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);

        struct DissolveBlendParam {
            float opacity;
        };                           

    private:
        DissolveBlendF32Gaudi2(const DissolveBlendF32Gaudi2& other) = delete;
        DissolveBlendF32Gaudi2& operator=(const DissolveBlendF32Gaudi2& other) = delete;
};

#endif
