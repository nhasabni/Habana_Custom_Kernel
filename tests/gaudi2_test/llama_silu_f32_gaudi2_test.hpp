#ifndef LLAMA_SILU_F32_TEST_HPP
#define LLAMA_SILU_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_silu_f32_gaudi2.hpp"

class LlamaSiluF32Gaudi2Test : public TestBase
{
public:
    LlamaSiluF32Gaudi2Test() {}
    ~LlamaSiluF32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamasilu_f32_reference_implementation(
            const float_1DTensor& in, float_1DTensor& out);
private:
    LlamaSiluF32Gaudi2Test(const LlamaSiluF32Gaudi2Test& other) = delete;
    LlamaSiluF32Gaudi2Test& operator=(const LlamaSiluF32Gaudi2Test& other) = delete;
};

#endif /* LLAMA_SILU_F32_TEST_HPP */
