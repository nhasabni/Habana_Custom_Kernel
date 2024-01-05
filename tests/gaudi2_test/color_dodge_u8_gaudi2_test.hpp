#ifndef COLOR_DODGE_U8_TEST_HPP
#define COLOR_DODGE_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "color_dodge_u8_gaudi2.hpp"

class ColorDodgeU8Gaudi2Test : public TestBase
{
public:
    ColorDodgeU8Gaudi2Test() {}
    ~ColorDodgeU8Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n);

    inline static void colordodge_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    ColorDodgeU8Gaudi2Test(const ColorDodgeU8Gaudi2Test& other) = delete;
    ColorDodgeU8Gaudi2Test& operator=(const ColorDodgeU8Gaudi2Test& other) = delete;
};


#endif /* COLOR_DODGE_U8_TEST_HPP */
