#ifndef CC2TACO_ARRAYSUM_I32_TEST_HPP
#define CC2TACO_ARRAYSUM_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "c2taco_arraysum_i32_gaudi2.hpp"

class C2TacoArraySumI32Gaudi2Test : public TestBase
{
public:
    C2TacoArraySumI32Gaudi2Test() {}
    ~C2TacoArraySumI32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void c2taco_arraysum_i32_reference_implementation(
            const int32_1DTensor& a, int32_1DTensor& sum);
private:
    C2TacoArraySumI32Gaudi2Test(const C2TacoArraySumI32Gaudi2Test& other) = delete;
    C2TacoArraySumI32Gaudi2Test& operator=(const C2TacoArraySumI32Gaudi2Test& other) = delete;
};

#endif /* CC2TACO_ARRAYSUM_I32_TEST_HPP */
