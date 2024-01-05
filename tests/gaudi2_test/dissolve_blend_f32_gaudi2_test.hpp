#ifndef DISSOLVE_BLEND_F32_TEST_HPP
#define DISSOLVE_BLEND_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "dissolve_blend_f32_gaudi2.hpp"

class DissolveBlendF32Gaudi2Test : public TestBase
{
public:
    DissolveBlendF32Gaudi2Test() {}
    ~DissolveBlendF32Gaudi2Test() {}
    int runTest();

    inline static void dissolveblend_f32_reference_implementation(
            const float_2DTensor& base,
            const float_2DTensor& active,
            const float_2DTensor& rand,
            float_2DTensor& out,
            DissolveBlendF32Gaudi2::DissolveBlendParam& param_def);
private:
    DissolveBlendF32Gaudi2Test(const DissolveBlendF32Gaudi2Test& other) = delete;
    DissolveBlendF32Gaudi2Test& operator=(const DissolveBlendF32Gaudi2Test& other) = delete;
};


#endif /* DISSOLVE_BLEND_F32_TEST_HPP */
