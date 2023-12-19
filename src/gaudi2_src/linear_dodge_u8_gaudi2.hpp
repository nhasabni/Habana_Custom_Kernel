#ifndef _LINEAR_DODGE_U8_GAUDI2_HPP
#define _LINEAR_DODGE_U8_GAUDI2_HPP

#include "gc_interface.h"

class LinearDodgeU8Gaudi2
{
    public:
        LinearDodgeU8Gaudi2() {}
        virtual ~LinearDodgeU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t* in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        LinearDodgeU8Gaudi2(const LinearDodgeU8Gaudi2& other) = delete;
        LinearDodgeU8Gaudi2& operator=(const LinearDodgeU8Gaudi2& other) = delete;
};

#endif
