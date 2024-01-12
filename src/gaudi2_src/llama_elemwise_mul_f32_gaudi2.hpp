#ifndef _LLAMA_ELEMWISE_MUL_F32_GAUDI2_HPP
#define _LLAMA_ELEMWISE_MUL_F32_GAUDI2_HPP

#include "gc_interface.h"

class LlamaElemwiseMulF32Gaudi2
{
    public:
        LlamaElemwiseMulF32Gaudi2() {}
        virtual ~LlamaElemwiseMulF32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        LlamaElemwiseMulF32Gaudi2(const LlamaElemwiseMulF32Gaudi2& other) = delete;
        LlamaElemwiseMulF32Gaudi2& operator=(const LlamaElemwiseMulF32Gaudi2& other) = delete;
};

#endif
