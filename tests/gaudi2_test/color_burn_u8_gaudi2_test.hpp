#ifndef COLOR_BURN_U8_TEST_HPP
#define COLOR_BURN_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "color_burn_u8_gaudi2.hpp"

class ColorBurnU8Gaudi2Test : public TestBase
{
public:
    ColorBurnU8Gaudi2Test() {}
    ~ColorBurnU8Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n);

    inline static void colorburn_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    ColorBurnU8Gaudi2Test(const ColorBurnU8Gaudi2Test& other) = delete;
    ColorBurnU8Gaudi2Test& operator=(const ColorBurnU8Gaudi2Test& other) = delete;

};


#endif /* COLOR_BURN_U8_TEST_HPP */
