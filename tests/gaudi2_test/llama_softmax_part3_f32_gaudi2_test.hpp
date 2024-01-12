#ifndef LLAMA_SOFTMAX_PART3_F32_TEST_HPP
#define LLAMA_SOFTMAX_PART3_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_softmax_part3_f32_gaudi2.hpp"

class LlamaSoftmaxPart3F32Gaudi2Test : public TestBase
{
public:
    LlamaSoftmaxPart3F32Gaudi2Test() {}
    ~LlamaSoftmaxPart3F32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamasoftmaxpart3_f32_reference_implementation(
            const float_1DTensor& in,
            float_1DTensor& out);
private:
    LlamaSoftmaxPart3F32Gaudi2Test(const LlamaSoftmaxPart3F32Gaudi2Test& other) = delete;
    LlamaSoftmaxPart3F32Gaudi2Test& operator=(const LlamaSoftmaxPart3F32Gaudi2Test& other) = delete;
};

#endif /* LLAMA_SOFTMAX_PART1_F32_TEST_HPP */
