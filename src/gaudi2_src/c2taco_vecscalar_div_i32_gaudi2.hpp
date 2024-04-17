#ifndef _C2TACO_VECSCALAR_DIV_I32_GAUDI2_HPP
#define _C2TACO_VECSCALAR_DIV_I32_GAUDI2_HPP

#include "gc_interface.h"

class C2TacoVecScalarDivI32Gaudi2
{
    public:
        C2TacoVecScalarDivI32Gaudi2() {}
        virtual ~C2TacoVecScalarDivI32Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

        struct Param {
            int scalar;
        };
    private:
        C2TacoVecScalarDivI32Gaudi2(const C2TacoVecScalarDivI32Gaudi2& other) = delete;
        C2TacoVecScalarDivI32Gaudi2& operator=(const C2TacoVecScalarDivI32Gaudi2& other) = delete;
};

#endif