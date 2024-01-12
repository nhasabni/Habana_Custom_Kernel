#ifndef _LLAMA_SOFTMAX_PART3_F32_GAUDI2_HPP
#define _LLAMA_SOFTMAX_PART3_F32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaSoftmaxPart3F32Gaudi2
{
    public:
        LlamaSoftmaxPart3F32Gaudi2() {}
        virtual ~LlamaSoftmaxPart3F32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        LlamaSoftmaxPart3F32Gaudi2(const LlamaSoftmaxPart3F32Gaudi2& other) = delete;
        LlamaSoftmaxPart3F32Gaudi2& operator=(const LlamaSoftmaxPart3F32Gaudi2& other) = delete;
};

#endif
