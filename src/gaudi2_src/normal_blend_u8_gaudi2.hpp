#ifndef _NORMAL_BLEND_U8_GAUDI2_HPP
#define _NORMAL_BLEND_U8_GAUDI2_HPP

#include "gc_interface.h"

class NormalBlendU8Gaudi2
{
    public:
        NormalBlendU8Gaudi2() {}
        virtual ~NormalBlendU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

        struct NormalBlendParam {
            uint8_t opacity;
        };
    private:
        NormalBlendU8Gaudi2(const NormalBlendU8Gaudi2& other) = delete;
        NormalBlendU8Gaudi2& operator=(const NormalBlendU8Gaudi2& other) = delete;
};

#endif
