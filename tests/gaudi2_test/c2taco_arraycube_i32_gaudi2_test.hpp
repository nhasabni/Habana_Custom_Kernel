#ifndef CC2TACO_ARRAYCUBE_I32_TEST_HPP
#define CC2TACO_ARRAYCUBE_I32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "c2taco_arraycube_i32_gaudi2.hpp"

class C2TacoArrayCubeI32Gaudi2Test : public TestBase
{
public:
    C2TacoArrayCubeI32Gaudi2Test() {}
    ~C2TacoArrayCubeI32Gaudi2Test() {}
    int runTest(uint32_t m);

    inline static void c2taco_arraycube_i32_reference_implementation(
            const int32_1DTensor& a, int32_1DTensor& cube);
private:
    C2TacoArrayCubeI32Gaudi2Test(const C2TacoArrayCubeI32Gaudi2Test& other) = delete;
    C2TacoArrayCubeI32Gaudi2Test& operator=(const C2TacoArrayCubeI32Gaudi2Test& other) = delete;
};

#endif /* CC2TACO_ARRAYCUBE_I32_TEST_HPP */
