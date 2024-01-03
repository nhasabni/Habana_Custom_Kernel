#ifndef _COLOR_DODGE_U8_GAUDI2_HPP
#define _COLOR_DODGE_U8_GAUDI2_HPP

#include "gc_interface.h"

class ColorDodgeU8Gaudi2
{
    public:
        ColorDodgeU8Gaudi2() {}
        virtual ~ColorDodgeU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        ColorDodgeU8Gaudi2(const ColorDodgeU8Gaudi2& other) = delete;
        ColorDodgeU8Gaudi2& operator=(const ColorDodgeU8Gaudi2& other) = delete;
};

#endif
