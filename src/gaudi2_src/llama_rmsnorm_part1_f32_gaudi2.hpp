#ifndef _LLAMA_RMSNORM_PART1_F32_GAUDI2_HPP
#define _LLAMA_RMSNORM_PART1_F32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaRmsnormPart1F32Gaudi2
{
    public:
        LlamaRmsnormPart1F32Gaudi2() {}
        virtual ~LlamaRmsnormPart1F32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        LlamaRmsnormPart1F32Gaudi2(const LlamaRmsnormPart1F32Gaudi2& other) = delete;
        LlamaRmsnormPart1F32Gaudi2& operator=(const LlamaRmsnormPart1F32Gaudi2& other) = delete;
};

#endif
