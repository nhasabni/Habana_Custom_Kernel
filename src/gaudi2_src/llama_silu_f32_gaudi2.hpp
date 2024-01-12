#ifndef _LLAMA_SILU_F32_GAUDI2_HPP
#define _LLAMA_SILU_F32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaSiluF32Gaudi2
{
    public:
        LlamaSiluF32Gaudi2() {}
        virtual ~LlamaSiluF32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                       

    private:
        LlamaSiluF32Gaudi2(const LlamaSiluF32Gaudi2& other) = delete;
        LlamaSiluF32Gaudi2& operator=(const LlamaSiluF32Gaudi2& other) = delete;
};

#endif
