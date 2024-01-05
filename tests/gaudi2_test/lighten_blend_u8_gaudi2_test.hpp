#ifndef LIGHTEN_BLEND_U8_TEST_HPP
#define LIGHTEN_BLEND_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "lighten_blend_u8_gaudi2.hpp"

class LightenBlendU8Gaudi2Test : public TestBase
{
public:
    LightenBlendU8Gaudi2Test() {}
    ~LightenBlendU8Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n);

    inline static void lightenblend_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    LightenBlendU8Gaudi2Test(const LightenBlendU8Gaudi2Test& other) = delete;
    LightenBlendU8Gaudi2Test& operator=(const LightenBlendU8Gaudi2Test& other) = delete;
};


#endif /* LIGHTEN_BLEND_U8_TEST_HPP */
