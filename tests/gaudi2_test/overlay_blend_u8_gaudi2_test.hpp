#ifndef OVERLAY_BLEND_U8_TEST_HPP
#define OVERLAY_BLEND_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "overlay_blend_u8_gaudi2.hpp"

class OverlayBlendU8Gaudi2Test : public TestBase
{
public:
    OverlayBlendU8Gaudi2Test() {}
    ~OverlayBlendU8Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n);

    inline static void overlayblend_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    OverlayBlendU8Gaudi2Test(const OverlayBlendU8Gaudi2Test& other) = delete;
    OverlayBlendU8Gaudi2Test& operator=(const OverlayBlendU8Gaudi2Test& other) = delete;
};


#endif /* OVERLAY_BLEND_U8_TEST_HPP */
