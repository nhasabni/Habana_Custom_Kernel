#ifndef NORMAL_BLEND_U8_TEST_HPP
#define NORMAL_BLEND_U8_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "normal_blend_u8_gaudi2.hpp"

class NormalBlendU8Gaudi2Test : public TestBase
{
public:
    NormalBlendU8Gaudi2Test() {}
    ~NormalBlendU8Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void normalblend_u8_reference_implementation(
            const uint8_1DTensor& base,
            const uint8_1DTensor& active,
            uint8_1DTensor& out,
            NormalBlendU8Gaudi2::NormalBlendParam& param_def);
private:
    NormalBlendU8Gaudi2Test(const NormalBlendU8Gaudi2Test& other) = delete;
    NormalBlendU8Gaudi2Test& operator=(const NormalBlendU8Gaudi2Test& other) = delete;

};


#endif /* NORMAL_BLEND_U8_TEST_HPP */
