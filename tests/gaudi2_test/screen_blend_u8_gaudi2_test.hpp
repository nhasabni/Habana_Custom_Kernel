#ifndef SCREEN_BLEND_U8_TEST_HPP
#define SCREEN_BLEND_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "screen_blend_u8_gaudi2.hpp"

class ScreenBlendU8Gaudi2Test : public TestBase
{
public:
    ScreenBlendU8Gaudi2Test() {}
    ~ScreenBlendU8Gaudi2Test() {}
    int runTest();

    inline static void screenblend_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    ScreenBlendU8Gaudi2Test(const ScreenBlendU8Gaudi2Test& other) = delete;
    ScreenBlendU8Gaudi2Test& operator=(const ScreenBlendU8Gaudi2Test& other) = delete;
};


#endif /* SCREEN_BLEND_U8_TEST_HPP */
