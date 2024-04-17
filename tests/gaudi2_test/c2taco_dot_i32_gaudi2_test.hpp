#ifndef C2TACO_DOT_I32_TEST_HPP
#define C2TACO_DOT_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "c2taco_dot_i32_gaudi2.hpp"

class C2TacoDotI32Gaudi2Test : public TestBase
{
public:
    C2TacoDotI32Gaudi2Test() {}
    ~C2TacoDotI32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void c2taco_dot_i32_reference_implementation(
            const int32_1DTensor& a, const int32_1DTensor& b,
            int32_1DTensor& sum);
private:
    C2TacoDotI32Gaudi2Test(const C2TacoDotI32Gaudi2Test& other) = delete;
    C2TacoDotI32Gaudi2Test& operator=(const C2TacoDotI32Gaudi2Test& other) = delete;
};

#endif /* C2TACO_DOT_I32_TEST_HPP */
