#ifndef NORMAL_BLEND_F32_TEST_HPP
#define NORMAL_BLEND_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "normal_blend_f32_gaudi2.hpp"

class NormalBlendF32Gaudi2Test : public TestBase
{
public:
    NormalBlendF32Gaudi2Test() {}
    ~NormalBlendF32Gaudi2Test() {}
    int runTest();

    inline static void normalblend_f32_reference_implementation(
            const float_1DTensor& base,
            const float_1DTensor& active,
            float_1DTensor& out,
            NormalBlendF32Gaudi2::NormalBlendParam& param_def);
private:
    NormalBlendF32Gaudi2Test(const NormalBlendF32Gaudi2Test& other) = delete;
    NormalBlendF32Gaudi2Test& operator=(const NormalBlendF32Gaudi2Test& other) = delete;

};


#endif /* NORMAL_BLEND_F32_TEST_HPP */
