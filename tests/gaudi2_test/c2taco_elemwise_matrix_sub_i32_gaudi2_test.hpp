#ifndef C2TACO_ELEMWISE_MATRIX_SUB_I32_TEST_HPP
#define C2TACO_ELEMWISE_MATRIX_SUB_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "c2taco_elemwise_matrix_sub_i32_gaudi2.hpp"

class C2TacoElemwiseMatrixSubI32Gaudi2Test : public TestBase
{
public:
    C2TacoElemwiseMatrixSubI32Gaudi2Test() {}
    ~C2TacoElemwiseMatrixSubI32Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n);

    inline static void elemwise_matrix_sub_i32_reference_implementation(
            const int32_2DTensor& a, const int32_2DTensor& b,
            int32_2DTensor& out);
private:
    C2TacoElemwiseMatrixSubI32Gaudi2Test(const C2TacoElemwiseMatrixSubI32Gaudi2Test& other) = delete;
    C2TacoElemwiseMatrixSubI32Gaudi2Test& operator=(const C2TacoElemwiseMatrixSubI32Gaudi2Test& other) = delete;
};


#endif /* C2TACO_ELEMWISE_MATRIX_SUB_I32_TEST_HPP */
