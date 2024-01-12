#ifndef LLAMA_ELEMWISE_MUL_F32_TEST_HPP
#define LLAMA_ELEMWISE_MUL_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "llama_elemwise_mul_f32_gaudi2.hpp"

class LlamaElemwiseMulF32Gaudi2Test : public TestBase
{
public:
    LlamaElemwiseMulF32Gaudi2Test() {}
    ~LlamaElemwiseMulF32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void llamaelemwisemul_f32_reference_implementation(
            const float_1DTensor& input1,
            const float_1DTensor& input2,
            float_1DTensor& out);
private:
    LlamaElemwiseMulF32Gaudi2Test(const LlamaElemwiseMulF32Gaudi2Test& other) = delete;
    LlamaElemwiseMulF32Gaudi2Test& operator=(const LlamaElemwiseMulF32Gaudi2Test& other) = delete;
};


#endif /* LLAMA_ELEMWISE_MUL_F32_TEST_HPP */
