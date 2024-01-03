#ifndef _MULTIPLY_BLEND_U8_GAUDI2_HPP
#define _MULTIPLY_BLEND_U8_GAUDI2_HPP

#include "gc_interface.h"

class MultiplyBlendU8Gaudi2
{
    public:
        MultiplyBlendU8Gaudi2() {}
        virtual ~MultiplyBlendU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        MultiplyBlendU8Gaudi2(const MultiplyBlendU8Gaudi2& other) = delete;
        MultiplyBlendU8Gaudi2& operator=(const MultiplyBlendU8Gaudi2& other) = delete;
};

#endif
