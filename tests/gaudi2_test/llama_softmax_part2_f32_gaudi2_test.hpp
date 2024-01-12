#ifndef LLAMA_SOFTMAX_PART2_F32_TEST_HPP
#define LLAMA_SOFTMAX_PART2_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_softmax_part2_f32_gaudi2.hpp"

class LlamaSoftmaxPart2F32Gaudi2Test : public TestBase
{
public:
    LlamaSoftmaxPart2F32Gaudi2Test() {}
    ~LlamaSoftmaxPart2F32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamasoftmaxpart2_f32_reference_implementation(
            const float_1DTensor& in,
            float max_val,
            float_1DTensor& out);
private:
    LlamaSoftmaxPart2F32Gaudi2Test(const LlamaSoftmaxPart2F32Gaudi2Test& other) = delete;
    LlamaSoftmaxPart2F32Gaudi2Test& operator=(const LlamaSoftmaxPart2F32Gaudi2Test& other) = delete;
};


#endif /* LLAMA_SOFTMAX_PART2_F32_TEST_HPP */
