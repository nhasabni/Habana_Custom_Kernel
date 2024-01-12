#ifndef _LLAMA_SOFTMAX_PART2_I32_GAUDI2_HPP
#define _LLAMA_SOFTMAX_PART2_I32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaSoftmaxPart2F32Gaudi2
{
    public:
        LlamaSoftmaxPart2F32Gaudi2() {}
        virtual ~LlamaSoftmaxPart2F32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);

        struct LlamaSoftmaxPart2Param {
            float max_val;
        };                       

    private:
        LlamaSoftmaxPart2F32Gaudi2(const LlamaSoftmaxPart2F32Gaudi2& other) = delete;
        LlamaSoftmaxPart2F32Gaudi2& operator=(const LlamaSoftmaxPart2F32Gaudi2& other) = delete;
};

#endif
