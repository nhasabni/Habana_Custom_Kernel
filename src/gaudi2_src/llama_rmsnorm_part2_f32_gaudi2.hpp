#ifndef _LLAMA_RMSNORM_PART2_F32_GAUDI2_HPP
#define _LLAMA_RMSNORM_PART2_F32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaRmsnormPart2F32Gaudi2
{
    public:
        LlamaRmsnormPart2F32Gaudi2() {}
        virtual ~LlamaRmsnormPart2F32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);

        struct LlamaRmsnormPart2Param {
            float ss;
        };                       

    private:
        LlamaRmsnormPart2F32Gaudi2(const LlamaRmsnormPart2F32Gaudi2& other) = delete;
        LlamaRmsnormPart2F32Gaudi2& operator=(const LlamaRmsnormPart2F32Gaudi2& other) = delete;
};

#endif
