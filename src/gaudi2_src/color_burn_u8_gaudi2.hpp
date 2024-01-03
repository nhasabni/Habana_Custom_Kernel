#ifndef _COLOR_BURN_U8_GAUDI2_HPP
#define _COLOR_BURN_U8_GAUDI2_HPP

#include "gc_interface.h"

class ColorBurnU8Gaudi2
{
    public:
        ColorBurnU8Gaudi2() {}
        virtual ~ColorBurnU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        ColorBurnU8Gaudi2(const ColorBurnU8Gaudi2& other) = delete;
        ColorBurnU8Gaudi2& operator=(const ColorBurnU8Gaudi2& other) = delete;
};

#endif
