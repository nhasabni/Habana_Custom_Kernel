#ifndef _LLAMA_SOFTMAX_PART1_I32_GAUDI2_HPP
#define _LLAMA_SOFTMAX_PART1_I32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaSoftmaxPart1I32Gaudi2
{
    public:
        LlamaSoftmaxPart1I32Gaudi2() {}
        virtual ~LlamaSoftmaxPart1I32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        LlamaSoftmaxPart1I32Gaudi2(const LlamaSoftmaxPart1I32Gaudi2& other) = delete;
        LlamaSoftmaxPart1I32Gaudi2& operator=(const LlamaSoftmaxPart1I32Gaudi2& other) = delete;
};

#endif
