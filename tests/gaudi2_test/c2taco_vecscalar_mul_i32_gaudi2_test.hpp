#ifndef C2TACO_VECSCALAR_MUL_I32_TEST_HPP
#define C2TACO_VECSCALAR_MUL_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "c2taco_vecscalar_mul_i32_gaudi2.hpp"

class C2TacoVecScalarMulI32Gaudi2Test : public TestBase
{
public:
    C2TacoVecScalarMulI32Gaudi2Test() {}
    ~C2TacoVecScalarMulI32Gaudi2Test() {}
    int runTest(uint32_t m, int scalar);

    inline static void c2taco_vecscalar_mul_i32_reference_implementation(
            const int32_1DTensor& a, int32_1DTensor& out,
            C2TacoVecScalarMulI32Gaudi2::Param& param_def);
private:
    C2TacoVecScalarMulI32Gaudi2Test(const C2TacoVecScalarMulI32Gaudi2Test& other) = delete;
    C2TacoVecScalarMulI32Gaudi2Test& operator=(const C2TacoVecScalarMulI32Gaudi2Test& other) = delete;
};


#endif /* C2TACO_VECSCALAR_MUL_I32_TEST_HPP */
