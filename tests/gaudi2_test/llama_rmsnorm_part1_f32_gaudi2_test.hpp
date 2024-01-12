#ifndef LLAMA_RMSNORM_PART1_F32_TEST_HPP
#define LLAMA_RMSNORM_PART1_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_rmsnorm_part1_f32_gaudi2.hpp"

class LlamaRmsnormPart1F32Gaudi2Test : public TestBase
{
public:
    LlamaRmsnormPart1F32Gaudi2Test() {}
    ~LlamaRmsnormPart1F32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamarmsnormpart1_f32_reference_implementation(
            const float_1DTensor& input, const float_1DTensor& weight,
            float_1DTensor& out);
private:
    LlamaRmsnormPart1F32Gaudi2Test(const LlamaRmsnormPart1F32Gaudi2Test& other) = delete;
    LlamaRmsnormPart1F32Gaudi2Test& operator=(const LlamaRmsnormPart1F32Gaudi2Test& other) = delete;
};

#endif /* LLAMA_RMSNORM_PART1_F32_TEST_HPP */
