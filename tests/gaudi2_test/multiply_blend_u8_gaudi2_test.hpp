#ifndef MULTIPLY_BLEND_U8_TEST_HPP
#define MULTIPLY_BLEND_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "multiply_blend_u8_gaudi2.hpp"

class MultiplyBlendU8Gaudi2Test : public TestBase
{
public:
    MultiplyBlendU8Gaudi2Test() {}
    ~MultiplyBlendU8Gaudi2Test() {}
    int runTest();

    inline static void multiplyblend_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    MultiplyBlendU8Gaudi2Test(const MultiplyBlendU8Gaudi2Test& other) = delete;
    MultiplyBlendU8Gaudi2Test& operator=(const MultiplyBlendU8Gaudi2Test& other) = delete;
};


#endif /* MULTIPLY_BLEND_U8_TEST_HPP */
