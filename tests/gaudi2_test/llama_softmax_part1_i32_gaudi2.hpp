#ifndef LLAMA_SOFTMAX_PART1_I32_TEST_HPP
#define LLAMA_SOFTMAX_PART1_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_softmax_part1_i32_gaudi2.hpp"

class LlamaSoftmaxPart1I32Gaudi2 : public TestBase
{
public:
    LlamaSoftmaxPart1I32Gaudi2() {}
    ~LlamaSoftmaxPart1I32Gaudi2() {}
    int runTest();

    inline static void llamasoftmaxpart1_i32_reference_implementation(
            const int32_1DTensor& in,
            int32_1DTensor& out);
private:
    LlamaSoftmaxPart1I32Gaudi2(const LlamaSoftmaxPart1I32Gaudi2& other) = delete;
    LlamaSoftmaxPart1I32Gaudi2& operator=(const LlamaSoftmaxPart1I32Gaudi2& other) = delete;

};


#endif /* LLAMA_SOFTMAX_PART1_I32_TEST_HPP */
