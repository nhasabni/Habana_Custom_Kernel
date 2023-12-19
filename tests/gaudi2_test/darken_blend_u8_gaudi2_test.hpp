#ifndef DARKEN_BLEND_U8_TEST_HPP
#define DARKEN_BLEND_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "darken_blend_u8_gaudi2.hpp"

class DarkenBlendU8Gaudi2Test : public TestBase
{
public:
    DarkenBlendU8Gaudi2Test() {}
    ~DarkenBlendU8Gaudi2Test() {}
    int runTest();

    inline static void darkenblend_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    DarkenBlendU8Gaudi2Test(const DarkenBlendU8Gaudi2Test& other) = delete;
    DarkenBlendU8Gaudi2Test& operator=(const DarkenBlendU8Gaudi2Test& other) = delete;
};


#endif /* DARKEN_BLEND_U8_TEST_HPP */
