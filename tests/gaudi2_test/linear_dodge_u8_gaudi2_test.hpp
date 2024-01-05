#ifndef LINEAR_DODGE_U8_TEST_HPP
#define LINEAR_DODGE_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "linear_dodge_u8_gaudi2.hpp"

class LinearDodgeU8Gaudi2Test : public TestBase
{
public:
    LinearDodgeU8Gaudi2Test() {}
    ~LinearDodgeU8Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n);

    inline static void lineardodge_u8_reference_implementation(
            const uint8_2DTensor& base,
            const uint8_2DTensor& active,
            uint8_2DTensor& out);
private:
    LinearDodgeU8Gaudi2Test(const LinearDodgeU8Gaudi2Test& other) = delete;
    LinearDodgeU8Gaudi2Test& operator=(const LinearDodgeU8Gaudi2Test& other) = delete;
};


#endif /* LINEAR_DODGE_U8_TEST_HPP */
