#ifndef DARKEN_BLEND_F32_TEST_HPP
#define DARKEN_BLEND_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "darken_blend_f32_gaudi2.hpp"

class DarkenBlendF32Gaudi2Test : public TestBase
{
public:
    DarkenBlendF32Gaudi2Test() {}
    ~DarkenBlendF32Gaudi2Test() {}
    int runTest();

    inline static void darkenblendf32_reference_implementation(
            const float_2DTensor& base,
            const float_2DTensor& active,
            float_2DTensor& out);
private:
    DarkenBlendF32Gaudi2Test(const DarkenBlendF32Gaudi2Test& other) = delete;
    DarkenBlendF32Gaudi2Test& operator=(const DarkenBlendF32Gaudi2Test& other) = delete;
};


#endif /* DARKEN_BLEND_F32_TEST_HPP */
