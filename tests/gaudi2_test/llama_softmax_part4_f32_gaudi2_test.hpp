#ifndef LLAMA_SOFTMAX_PART4_F32_TEST_HPP
#define LLAMA_SOFTMAX_PART4_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_softmax_part4_f32_gaudi2.hpp"

class LlamaSoftmaxPart4F32Gaudi2Test : public TestBase
{
public:
    LlamaSoftmaxPart4F32Gaudi2Test() {}
    ~LlamaSoftmaxPart4F32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamasoftmaxpart4_f32_reference_implementation(
            const float_1DTensor& in, float sum, float_1DTensor& out);
private:
    LlamaSoftmaxPart4F32Gaudi2Test(const LlamaSoftmaxPart4F32Gaudi2Test& other) = delete;
    LlamaSoftmaxPart4F32Gaudi2Test& operator=(const LlamaSoftmaxPart4F32Gaudi2Test& other) = delete;
};

#endif /* LLAMA_SOFTMAX_PART4_F32_TEST_HPP */
