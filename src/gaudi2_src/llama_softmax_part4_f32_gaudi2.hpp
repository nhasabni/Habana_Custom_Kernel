#ifndef _LLAMA_SOFTMAX_PART4_F32_GAUDI2_HPP
#define _LLAMA_SOFTMAX_PART4_F32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaSoftmaxPart4F32Gaudi2
{
    public:
        LlamaSoftmaxPart4F32Gaudi2() {}
        virtual ~LlamaSoftmaxPart4F32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);

        struct LlamaSoftmaxPart4Param {
            float sum;
        };                       

    private:
        LlamaSoftmaxPart4F32Gaudi2(const LlamaSoftmaxPart4F32Gaudi2& other) = delete;
        LlamaSoftmaxPart4F32Gaudi2& operator=(const LlamaSoftmaxPart4F32Gaudi2& other) = delete;
};

#endif
