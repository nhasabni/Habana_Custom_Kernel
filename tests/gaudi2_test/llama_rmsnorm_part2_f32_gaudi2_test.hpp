#ifndef LLAMA_RMSNORM_PART2_F32_TEST_HPP
#define LLAMA_RMSNORM_PART2_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_rmsnorm_part2_f32_gaudi2.hpp"

class LlamaRmsnormPart2F32Gaudi2Test : public TestBase
{
public:
    LlamaRmsnormPart2F32Gaudi2Test() {}
    ~LlamaRmsnormPart2F32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamarmsnormpart2_f32_reference_implementation(
            const float_1DTensor& in, const float_1DTensor& weight,
            float ss, float_1DTensor& out);
private:
    LlamaRmsnormPart2F32Gaudi2Test(const LlamaRmsnormPart2F32Gaudi2Test& other) = delete;
    LlamaRmsnormPart2F32Gaudi2Test& operator=(const LlamaRmsnormPart2F32Gaudi2Test& other) = delete;
};

#endif /* LLAMA_RMSNORM_PART2_F32_TEST_HPP */
