#ifndef _OVERLAY_BLEND_U8_GAUDI2_HPP
#define _OVERLAY_BLEND_U8_GAUDI2_HPP

#include "gc_interface.h"

class OverlayBlendU8Gaudi2
{
    public:
        OverlayBlendU8Gaudi2() {}
        virtual ~OverlayBlendU8Gaudi2() {}

        virtual gcapi::GlueCodeReturn_t
        GetGcDefinitions(gcapi::HabanaKernelParams_t*      in_defs,
                     gcapi::HabanaKernelInstantiation_t* out_defs);

        virtual gcapi::GlueCodeReturn_t GetKernelName(
                char kernelName [gcapi::MAX_NODE_NAME]);                            

    private:
        OverlayBlendU8Gaudi2(const OverlayBlendU8Gaudi2& other) = delete;
        OverlayBlendU8Gaudi2& operator=(const OverlayBlendU8Gaudi2& other) = delete;
};

#endif
