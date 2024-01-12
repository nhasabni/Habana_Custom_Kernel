#ifndef LLAMA_SOFTMAX_PART1_F32_TEST_HPP
#define LLAMA_SOFTMAX_PART1_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_softmax_part1_f32_gaudi2.hpp"

class LlamaSoftmaxPart1F32Gaudi2Test : public TestBase
{
public:
    LlamaSoftmaxPart1F32Gaudi2Test() {}
    ~LlamaSoftmaxPart1F32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamasoftmaxpart1_f32_reference_implementation(
            const float_1DTensor& in,
            float_1DTensor& out);
private:
    LlamaSoftmaxPart1F32Gaudi2Test(const LlamaSoftmaxPart1F32Gaudi2Test& other) = delete;
    LlamaSoftmaxPart1F32Gaudi2Test& operator=(const LlamaSoftmaxPart1F32Gaudi2Test& other) = delete;
};

#endif /* LLAMA_SOFTMAX_PART1_F32_TEST_HPP */
