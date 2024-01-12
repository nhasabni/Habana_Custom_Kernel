#ifndef _LLAMA_SOFTMAX_PART1_F32_GAUDI2_HPP
#define _LLAMA_SOFTMAX_PART1_F32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaSoftmaxPart1F32Gaudi2
{
    public:
        LlamaSoftmaxPart1F32Gaudi2() {}
        virtual ~LlamaSoftmaxPart1F32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        LlamaSoftmaxPart1F32Gaudi2(const LlamaSoftmaxPart1F32Gaudi2& other) = delete;
        LlamaSoftmaxPart1F32Gaudi2& operator=(const LlamaSoftmaxPart1F32Gaudi2& other) = delete;
};

#endif
