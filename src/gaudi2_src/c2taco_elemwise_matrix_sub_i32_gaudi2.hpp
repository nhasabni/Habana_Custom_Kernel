#ifndef _C2TACO_ELEMWISE_MATRIX_SUB_I32_GAUDI2_HPP
#define _C2TACO_ELEMWISE_MATRIX_SUB_I32_GAUDI2_HPP

#include "gc_interface.h"

class C2TacoElemwiseMatrixSubI32Gaudi2
{
    public:
        C2TacoElemwiseMatrixSubI32Gaudi2() {}
        virtual ~C2TacoElemwiseMatrixSubI32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t* in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        C2TacoElemwiseMatrixSubI32Gaudi2(const C2TacoElemwiseMatrixSubI32Gaudi2& other) = delete;
        C2TacoElemwiseMatrixSubI32Gaudi2& operator=(const C2TacoElemwiseMatrixSubI32Gaudi2& other) = delete;
};

#endif
