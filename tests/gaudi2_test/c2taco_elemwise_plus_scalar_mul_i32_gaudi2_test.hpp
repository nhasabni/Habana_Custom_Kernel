#ifndef C2TACO_ELEMWISE_PLUS_SCALAR_MUL_I32_TEST_HPP
#define C2TACO_ELEMWISE_PLUS_SCALAR_MUL_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "c2taco_elemwise_plus_scalar_mul_i32_gaudi2.hpp"

class C2TacoElemwisePlusScalarMulI32Gaudi2Test : public TestBase
{
public:
    C2TacoElemwisePlusScalarMulI32Gaudi2Test() {}
    ~C2TacoElemwisePlusScalarMulI32Gaudi2Test() {}
    int runTest(uint32_t m, int scalar);

    inline static void c2taco_elemwise_plus_scalar_mul_i32_reference_implementation(
            const int32_1DTensor& a, const int32_1DTensor& b,
            C2TacoElemwisePlusScalarMulI32Gaudi2::Param& param_def,
            int32_1DTensor& out);
private:
    C2TacoElemwisePlusScalarMulI32Gaudi2Test(const C2TacoElemwisePlusScalarMulI32Gaudi2Test& other) = delete;
    C2TacoElemwisePlusScalarMulI32Gaudi2Test& operator=(const C2TacoElemwisePlusScalarMulI32Gaudi2Test& other) = delete;
};


#endif /* C2TACO_PLUS_SCALAR_MULEQ_I32_TEST_HPP */
