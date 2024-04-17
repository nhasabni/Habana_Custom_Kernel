#ifndef C2TACO_MATSCALAR_MUL_I32_TEST_HPP
#define C2TACO_MATSCALAR_MUL_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "c2taco_matscalar_mul_i32_gaudi2.hpp"

class C2TacoMatScalarMulI32Gaudi2Test : public TestBase
{
public:
    C2TacoMatScalarMulI32Gaudi2Test() {}
    ~C2TacoMatScalarMulI32Gaudi2Test() {}
    int runTest(uint32_t m, uint32_t n, int scalar);

    inline static void matscalar_mul_i32_reference_implementation(
            const int32_2DTensor& a, C2TacoMatScalarMulI32Gaudi2::Param& param,
            int32_2DTensor& out);
private:
    C2TacoMatScalarMulI32Gaudi2Test(const C2TacoMatScalarMulI32Gaudi2Test& other) = delete;
    C2TacoMatScalarMulI32Gaudi2Test& operator=(const C2TacoMatScalarMulI32Gaudi2Test& other) = delete;
};


#endif /* C2TACO_MATSCALAR_MUL_I32_TEST_HPP */
