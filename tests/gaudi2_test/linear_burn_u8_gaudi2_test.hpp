#ifndef LINEAR_BURN_U8_TEST_HPP
#define LINEAR_BURN_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "linear_burn_u8_gaudi2.hpp"

class LinearBurnU8Gaudi2Test : public TestBase
{
public:
    LinearBurnU8Gaudi2Test() {}
    ~LinearBurnU8Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n);

    inline static void linearburn_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    LinearBurnU8Gaudi2Test(const LinearBurnU8Gaudi2Test& other) = delete;
    LinearBurnU8Gaudi2Test& operator=(const LinearBurnU8Gaudi2Test& other) = delete;
};


#endif /* LINEAR_BURN_U8_TEST_HPP */
