/**********************************************************************
Copyright (c) 2022 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <iostream>
#include <cstdlib>
#include "filter_fwd_2d_bf16_test.hpp"
#include "softmax_bf16_test.hpp"
#include "softmax_bf16_gaudi2_test.hpp"
#include "cast_gaudi_test.hpp"
#include "batchnorm_f32_test.hpp"
#include "leakyrelu_f32_gaudi_test.hpp"
#include "sparse_lengths_sum_bf16_test.hpp"
#include "customdiv_fwd_f32_test.hpp"
#include "relu6_all_test.hpp"
#include "matrix_mul_fwd_f32_test.hpp"
#include "spatial_conv_f32_test.hpp"
#include "sin_f32_test.hpp"
#include "add_f32_test.hpp"
#include "avg_pool_2d_f32_test.hpp"
#include "avg_pool_2d_f32_gaudi2_test.hpp"
#include "cast_f16_to_i16_gaudi2_test.hpp"
#include "searchsorted_f32_test.hpp"
#include "gather_fwd_i32_test.hpp"
#include "kl_div_all_test.hpp"

#include "normal_blend_f32_gaudi2_test.hpp"
#include "normal_blend_u8_gaudi2_test.hpp"
#include "dissolve_blend_f32_gaudi2_test.hpp"
#include "darken_blend_u8_gaudi2_test.hpp"
#include "multiply_blend_u8_gaudi2_test.hpp"
#include "lighten_blend_u8_gaudi2_test.hpp"
#include "color_burn_u8_gaudi2_test.hpp"
#include "color_dodge_u8_gaudi2_test.hpp"
#include "screen_blend_u8_gaudi2_test.hpp"
#include "overlay_blend_u8_gaudi2_test.hpp"
#include "linear_burn_u8_gaudi2_test.hpp"
#include "linear_dodge_u8_gaudi2_test.hpp"

#include "llama_elemwise_mul_f32_gaudi2_test.hpp"
#include "llama_softmax_part1_f32_gaudi2_test.hpp"
#include "llama_softmax_part2_f32_gaudi2_test.hpp"
#include "llama_softmax_part3_f32_gaudi2_test.hpp"
#include "llama_softmax_part4_f32_gaudi2_test.hpp"
#include "llama_rmsnorm_part1_f32_gaudi2_test.hpp"
#include "llama_rmsnorm_part2_f32_gaudi2_test.hpp"
#include "llama_silu_f32_gaudi2_test.hpp"

#include "c2taco_dot_i32_gaudi2_test.hpp"
#include "c2taco_lensq_i32_gaudi2_test.hpp"
#include "c2taco_arraysum_i32_gaudi2_test.hpp"
#include "c2taco_arraycube_i32_gaudi2_test.hpp"
#include "c2taco_arrayfourth_i32_gaudi2_test.hpp"

#include "c2taco_elemwise_plus_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_sub_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_mul_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_div_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_mul_add_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_sub_square_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_plus_scalar_mul_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_matrix_add_i32_gaudi2_test.hpp"
#include "c2taco_elemwise_matrix_sub_i32_gaudi2_test.hpp"
#include "c2taco_matscalar_mul_i32_gaudi2_test.hpp"
#include "c2taco_vecscalar_add_i32_gaudi2_test.hpp"
#include "c2taco_vecscalar_sub_i32_gaudi2_test.hpp"
#include "c2taco_vecscalar_mul_i32_gaudi2_test.hpp"
#include "c2taco_vecscalar_div_i32_gaudi2_test.hpp"

int main(int argc, char** argv)
{
    int result = 0;
    static int testCount = 0;

    if(argc == 2 && ((strcmp(argv[1], "--help") ==0) || (strcmp(argv[1],"-h") ==0)))
    {
        std::cout << argv[0] << " " << "[options]" << std::endl <<
            "Options:" << std::endl <<
            "N/A                        Run all test cases" << std::endl <<
            "-h | --help                Print this help" << std::endl <<
            "-d | --device <DeviceName> Run only kernels for the DeviceName" << std::endl <<
            "-t | --test  <TestName>    Run <TestName>> only   " << std::endl <<
            "DeviceName:" << std::endl <<
            "Gaudi                      Run all Gaudi kernels only   " << std::endl <<
            "TestName:" << std::endl <<
            "FilterFwd2DBF16Test        Run FilterFwd2DBF16Test only   " << std::endl <<
            "SoftMaxBF16Test            Run SoftMaxBF16Test only   " << std::endl <<
            "CastGaudiTest              Run CastGaudiTest only   " << std::endl <<
            "BatchNormF32Test           Run BatchNormF32Test only   " << std::endl <<
            "LeakyReluF32GaudiTest      Run LeakyReluF32GaudiTest only   " << std::endl <<
            "SparseLengthsBF16Test      Run SparseLengthsBF16Test only   " << std::endl <<
            "CustomdivFwdF32Test        Run CustomdivFwdF32Test only   " << std::endl <<
            "Relu6FwdF32                Run Relu6FwdF32 only   " << std::endl <<
            "Relu6BwdF32                Run Relu6BwdF32 only   " << std::endl <<
            "Relu6FwdBF16               Run Relu6FwdBF16 only   " << std::endl <<
            "Relu6BwdBF16               Run Relu6BwdBF16 only   " << std::endl <<
            "ReluFwdF32                 Run ReluFwdF32 only   " << std::endl <<
            "ReluBwdF32                 Run ReluBwdF32 only   " << std::endl <<
            "ReluFwdBF16                Run ReluFwdBF16 only   " << std::endl <<
            "ReluBwdBF16                Run ReluBwdBF16 only   " << std::endl <<
            "MatrixMulFwdF32Test        Run MatrixMulFwdF32Test only   " << std::endl <<
            "SpatialConvF32Test         Run SpatialConvF32Test only   " << std::endl <<
            "SinF32Test                 Run SinF32Test only   " << std::endl <<
            "AddF32Test                 Run AddF32Test only   " << std::endl <<
            "AvgPool2DFwdF32Test        Run AvgPool2DFwdF32Test only   " << std::endl <<
            "AvgPool2DBwdF32Test        Run AvgPool2DBwdF32Test only   " << std::endl <<
            "SearchSortedFwdF32Test     Run SearchSortedFwdF32Test only   " << std::endl <<
            "GatherFwdDim0I32Test       Run GatherFwdDim0I32Test only   " << std::endl <<
            "KLDivFwdF32                Run KLDivFwdF32 only   "          << std::endl <<

            "AvgPool2DFwdF32Gaudi2Test  Run AvgPool2DFwdF32Gaudi2Test only   " << std::endl <<
            "AvgPool2DBwdF32Gaudi2Test  Run AvgPool2DBwdF32Gaudi2Test only   " << std::endl <<
            "CastF16toI16Gaudi2Test     Run CastF16toI16Gaudi2Test only   " << std::endl <<

            "SoftMaxBF16Gaudi2Test      Run SoftMaxBF16Gaudi2Test only   " << std::endl <<
            "NormalBlendF32Gaudi2Test   Run NormalBlendF32Gaudi2Test only " << std::endl <<
            "NormalBlendU8Gaudi2Test    Run NormalBlendU8Gaudi2Test only " << std::endl <<
            "DissolveBlendF32Gaudi2Test Run DissolveBlendF32Gaudi2Test only " << std::endl <<
            "DarkenBlendU8Gaudi2Test    Run DarkenBlendU8Gaudi2Test only " << std::endl <<
            "MultiplyBlendU8Gaudi2Test  Run DarkenBlendU8Gaudi2Test only " << std::endl <<
            "LightenBlendU8Gaudi2Test   Run LightenBlendU8Gaudi2Test only " << std::endl <<
            "ColorBurnU8Gaudi2Test      Run ColorBurnU8Gaudi2Test only " << std::endl <<
            "ColorDodgeU8Gaudi2Test     Run ColorDodgeU8Gaudi2Test only " << std::endl <<
            "ScreenBlendU8Gaudi2Test    Run ScreenBlendU8Gaudi2Test only " << std::endl <<
            "OverlayBlendU8Gaudi2Test   Run OverlayBlendU8Gaudi2Test only " << std::endl <<
            "LinearBurnU8Gaudi2Test     Run LinearBurnU8Gaudi2Test only " << std::endl <<
            "LinearDodgeU8Gaudi2Test    Run LinearDodgeU8Gaudi2Test only " << std::endl <<

            "LlamaElemwiseMulF32Gaudi2Test   Run LLamaElemwiseMulGaudi2Test only" << std::endl <<
            "LlamaSoftmaxPart1F32Gaudi2Test  Run LLamaSoftmaxPart1Gaudi2Test only" << std::endl <<
            "LlamaSoftmaxPart2F32Gaudi2Test  Run LLamaSoftmaxPart2Gaudi2Test only" << std::endl <<
            "LlamaSoftmaxPart3F32Gaudi2Test  Run LLamaSoftmaxPart3Gaudi2Test only" << std::endl <<
            "LlamaSoftmaxPart4F32Gaudi2Test  Run LLamaSoftmaxPart4Gaudi2Test only" << std::endl <<
            "LlamaRmsnormPart1F32Gaudi2Test  Run LLamaRmsnormPart1Gaudi2Test only" << std::endl <<
            "LlamaRmsnormPart2F32Gaudi2Test  Run LLamaRmsnormPart2Gaudi2Test only" << std::endl <<
            "LlamaSiluF32Gaudi2Test          Run LLamaSiluGaudi2Test only" << std::endl <<

            "C2TacoDotI32Gaudi2Test          Run C2TacoDotGaudi2Test only" << std::endl <<
            "C2TacoLenSqI32Gaudi2Test        Run C2TacoLenSqGaudi2Test only" << std::endl <<
            "C2TacoArraySumI32Gaudi2Test     Run C2TacoArraySumGaudi2Test only" << std::endl <<
            "C2TacoArrayCubeI32Gaudi2Test    Run C2TacoArrayCubeGaudi2Test only" << std::endl <<
            "C2TacoArrayFourthI32Gaudi2Test  Run C2TacoArrayFourthGaudi2Test only" << std::endl <<
            "C2TacoElemwisePlusI32Gaudi2Test Run C2TacoElemwisePlusI32Gaudi2Test only" << std::endl <<
            "C2TacoElemwiseSubI32Gaudi2Test  Run C2TacoElemwiseSubI32Gaudi2Test only" << std::endl <<
            "C2TacoElemwiseMulI32Gaudi2Test  Run C2TacoElemwiseMulI32Gaudi2Test only" << std::endl <<
            "C2TacoElemwiseDivI32Gaudi2Test  Run C2TacoElemwiseDivI32Gaudi2Test only" << std::endl <<
            "C2TacoElemwiseMulAddI32Gaudi2Test Run C2TacoElemwiseMulAddI32Gaudi2Test only" << std::endl <<
            "C2TacoElemwiseSubSquareI32Gaudi2Test  Run C2TacoElemwiseSubSquareI32Gaudi2Test only" << std::endl <<
            "C2TacoElemwisePlusScalarMulI32Gaudi2Test Run C2TacoElemwisePlusScalarMulI32Gaudi2Test only" << std::endl <<

            "C2TacoElemwiseMatrixAddI32Gaudi2Test  Run C2TacoElemwiseMatrixAddI32Gaudi2Test only" << std::endl <<
            "C2TacoElemwiseMatrixSubI32Gaudi2Test  Run C2TacoElemwiseMatrixSubI32Gaudi2Test only" << std::endl <<
            "C2TacoMatScalarMulI32Gaudi2Test  Run C2TacoMatScalarMulI32Gaudi2Test only" << std::endl <<

            "C2TacoVecScalarAddI32Gaudi2Test  Run C2TacoVecScalarAddGaudi2Test only" << std::endl <<
            "C2TacoVecScalarSubI32Gaudi2Test  Run C2TacoVecScalarSubGaudi2Test only" << std::endl << 
            "C2TacoVecScalarMulI32Gaudi2Test  Run C2TacoVecScalarMulGaudi2Test only" << std::endl <<
            "C2TacoVecScalarDivI32Gaudi2Test  Run C2TacoVecScalarDivGaudi2Test only" << std::endl;

        exit(0);
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"FilterFwd2DBF16Test") ==0))))
    {
        FilterFwd2DBF16Test test_bf16;
        test_bf16.SetUp();
        result = test_bf16.runTest();
        test_bf16.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SoftMaxBF16Test") ==0))))
    {
        SoftMaxBF16Test testSoftMaxBF16;
        testSoftMaxBF16.SetUp();
        result = testSoftMaxBF16.runTest();
        testSoftMaxBF16.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"CastGaudiTest") ==0))))
    {
        CastGaudiTest testCaseGaudi;
        testCaseGaudi.SetUp();
        result = testCaseGaudi.runTest();
        testCaseGaudi.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"BatchNormF32Test") ==0))))
    {
        BatchNormF32Test testBatchNorm;
        testBatchNorm.SetUp();
        result = testBatchNorm.runTest();
        testBatchNorm.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LeakyReluF32GaudiTest") ==0))))
    {
        LeakyReluF32GaudiTest testLeakyReluGaudi;
        testLeakyReluGaudi.SetUp();
        result = testLeakyReluGaudi.runTest();
        testLeakyReluGaudi.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SparseLengthsBF16Test") ==0))))
    {
        SparseLengthsSumBF16Test testSparseLenGaudi;
        testSparseLenGaudi.SetUp();
        result = testSparseLenGaudi.runTest();
        testSparseLenGaudi.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"CustomdivFwdF32Test") ==0))))
    {
        CustomdivFwdF32Test testCustomDivFwdF32;
        testCustomDivFwdF32.SetUp();
        result = testCustomDivFwdF32.runTest();
        testCustomDivFwdF32.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    Relu6AllTest testRelu6;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6FwdF32") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_F32);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6BwdF32") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_F32);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6FwdBF16") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_BF16);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6BwdBF16") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_BF16);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    Relu6AllTest testRelu;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluFwdF32") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_FWD_F32);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluBwdF32") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_BWD_F32);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluFwdBF16") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_FWD_BF16);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluBwdBF16") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_BWD_BF16);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"MatrixMulFwdF32Test") ==0))))
    {
        MatrixMulFwdF32Test testMatrixMulFwdF32;
        testMatrixMulFwdF32.SetUp();
        result = testMatrixMulFwdF32.runTest();
        testMatrixMulFwdF32.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SpatialConvF32Test") ==0))))
    {
        SpatialConvF32Test spatialConv;
        spatialConv.SetUp();
        result = spatialConv.runTest();
        spatialConv.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SinF32Test") ==0))))
    {
        SinF32Test sinf32ins;
        sinf32ins.SetUp();
        result = sinf32ins.runTest();
        sinf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AddF32Test") ==0))))
    {
        AddF32Test addf32ins;
        addf32ins.SetUp();
        result = addf32ins.runTest();
        addf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    AvgPool2DF32Test avgpool2df32ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DFwdF32Test") ==0))))
    {
        avgpool2df32ins.SetUp();
        result = avgpool2df32ins.runTest(GAUDI_KERNEL_AVG_POOL_2D_FWD_F32);
        avgpool2df32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DBwdF32Test") ==0))))
    {
        avgpool2df32ins.SetUp();
        result = avgpool2df32ins.runTest(GAUDI_KERNEL_AVG_POOL_2D_BWD_F32);
        avgpool2df32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    SearchSortedF32Test searchsortedf32ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SearchSortedFwdF32Test") ==0)))) 
    {
        searchsortedf32ins.SetUp();
        result = searchsortedf32ins.runTest();
        searchsortedf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    GatherFwdI32Test gatheri32ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
         && (strcmp(argv[2],"GatherFwdDim0I32Test") ==0))))
    {
        gatheri32ins.SetUp();
        result = gatheri32ins.runTest(GAUDI_KERNEL_GATHER_FWD_DIM0_I32);
        gatheri32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    KLDivAllTest testKLDiv;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"KLDivFwdF32") ==0))))
    {
        testKLDiv.SetUp();
        result = testKLDiv.runTest(GAUDI_KERNEL_KL_DIV_FWD_F32);
        testKLDiv.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    // The following ones are for Gaudi2
    AvgPool2DF32Gaudi2Test avgpool2df32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DFwdF32Gaudi2Test") ==0))))
    {
        avgpool2df32Gaudi2ins.SetUp();
        result = avgpool2df32Gaudi2ins.runTest(GAUDI2_KERNEL_AVG_POOL_2D_FWD_F32);
        avgpool2df32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DBwdF32Gaudi2Test") ==0))))
    {
        avgpool2df32Gaudi2ins.SetUp();
        result = avgpool2df32Gaudi2ins.runTest(GAUDI2_KERNEL_AVG_POOL_2D_BWD_F32);
        avgpool2df32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    CastF16toI16Gaudi2Test castf16tpi16Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"CastF16toI16Gaudi2Test") ==0))))
    {
        castf16tpi16Gaudi2ins.SetUp();
        result = castf16tpi16Gaudi2ins.runTest();
        castf16tpi16Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SoftMaxBF16Gaudi2Test") ==0))))
    {
        SoftMaxBF16Gaudi2Test testSoftMaxBF16Gaudi2;
        testSoftMaxBF16Gaudi2.SetUp();
        result = testSoftMaxBF16Gaudi2.runTest();
        testSoftMaxBF16Gaudi2.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    NormalBlendF32Gaudi2Test normalblendf32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"NormalBlendF32Gaudi2Test") ==0))))
    {
        normalblendf32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = normalblendf32Gaudi2ins.runTest(m);
        normalblendf32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    NormalBlendU8Gaudi2Test normalblendu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"NormalBlendU8Gaudi2Test") ==0))))
    {
        normalblendu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = normalblendu8Gaudi2ins.runTest(m);
        normalblendu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    DissolveBlendF32Gaudi2Test dissolveblendf32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"DissolveBlendF32Gaudi2Test") ==0))))
    {
        dissolveblendf32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = dissolveblendf32Gaudi2ins.runTest(m, n);
        dissolveblendf32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    DarkenBlendU8Gaudi2Test darkenblendu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"DarkenBlendU8Gaudi2Test") ==0))))
    {
        darkenblendu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = darkenblendu8Gaudi2ins.runTest(m, n);
        darkenblendu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    MultiplyBlendU8Gaudi2Test multiplyblendu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"MultiplyBlendU8Gaudi2Test") ==0))))
    {
        multiplyblendu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = multiplyblendu8Gaudi2ins.runTest(m, n);
        multiplyblendu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LightenBlendU8Gaudi2Test lightenblendu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LightenBlendU8Gaudi2Test") ==0))))
    {
        lightenblendu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = lightenblendu8Gaudi2ins.runTest(m, n);
        lightenblendu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    ColorBurnU8Gaudi2Test colorburnu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ColorBurnU8Gaudi2Test") ==0))))
    {
        colorburnu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = colorburnu8Gaudi2ins.runTest(m, n);
        colorburnu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    ColorDodgeU8Gaudi2Test colordodgeu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ColorDodgeU8Gaudi2Test") ==0))))
    {
        colordodgeu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = colordodgeu8Gaudi2ins.runTest(m, n);
        colordodgeu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    ScreenBlendU8Gaudi2Test screenblendu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ScreenBlendU8Gaudi2Test") ==0))))
    {
        screenblendu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = screenblendu8Gaudi2ins.runTest(m, n);
        screenblendu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    OverlayBlendU8Gaudi2Test overlayblendu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"OverlayBlendU8Gaudi2Test") ==0))))
    {
        overlayblendu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = overlayblendu8Gaudi2ins.runTest(m, n);
        overlayblendu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LinearBurnU8Gaudi2Test linearburnu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LinearBurnU8Gaudi2Test") ==0))))
    {
        linearburnu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = linearburnu8Gaudi2ins.runTest(m, n);
        linearburnu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LinearDodgeU8Gaudi2Test lineardodgeu8Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LinearDodgeU8Gaudi2Test") ==0))))
    {
        lineardodgeu8Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = lineardodgeu8Gaudi2ins.runTest(m, n);
        lineardodgeu8Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaElemwiseMulF32Gaudi2Test llamaelemwisemulf32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaElemwiseMulF32Gaudi2Test") ==0))))
    {
        llamaelemwisemulf32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamaelemwisemulf32Gaudi2ins.runTest(m);
        llamaelemwisemulf32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaSoftmaxPart1F32Gaudi2Test llamasoftmaxpart1f32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaSoftmaxPart1F32Gaudi2Test") ==0))))
    {
        llamasoftmaxpart1f32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamasoftmaxpart1f32Gaudi2ins.runTest(m);
        llamasoftmaxpart1f32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaSoftmaxPart2F32Gaudi2Test llamasoftmaxpart2f32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaSoftmaxPart2F32Gaudi2Test") ==0))))
    {
        llamasoftmaxpart2f32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamasoftmaxpart2f32Gaudi2ins.runTest(m);
        llamasoftmaxpart2f32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaSoftmaxPart3F32Gaudi2Test llamasoftmaxpart3f32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaSoftmaxPart3F32Gaudi2Test") ==0))))
    {
        llamasoftmaxpart3f32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamasoftmaxpart3f32Gaudi2ins.runTest(m);
        llamasoftmaxpart3f32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaSoftmaxPart4F32Gaudi2Test llamasoftmaxpart4f32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaSoftmaxPart4F32Gaudi2Test") ==0))))
    {
        llamasoftmaxpart4f32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamasoftmaxpart4f32Gaudi2ins.runTest(m);
        llamasoftmaxpart4f32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaRmsnormPart1F32Gaudi2Test llamarmsnormpart1f32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaRmsnormPart1F32Gaudi2Test") ==0))))
    {
        llamarmsnormpart1f32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamarmsnormpart1f32Gaudi2ins.runTest(m);
        llamarmsnormpart1f32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaRmsnormPart2F32Gaudi2Test llamarmsnormpart2f32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaRmsnormPart2F32Gaudi2Test") ==0))))
    {
        llamarmsnormpart2f32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamarmsnormpart2f32Gaudi2ins.runTest(m);
        llamarmsnormpart2f32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    LlamaSiluF32Gaudi2Test llamasiluf32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LlamaSiluF32Gaudi2Test") ==0))))
    {
        llamasiluf32Gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = llamasiluf32Gaudi2ins.runTest(m);
        llamasiluf32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoDotI32Gaudi2Test c2tacodoti32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoDotI32Gaudi2Test") ==0))))
    {
        c2tacodoti32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacodoti32gaudi2ins.runTest(m);
        c2tacodoti32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoLenSqI32Gaudi2Test c2tacolensqi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoLenSqI32Gaudi2Test") ==0))))
    {
        c2tacolensqi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacolensqi32gaudi2ins.runTest(m);
        c2tacolensqi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoArraySumI32Gaudi2Test c2tacoarraysumi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoArraySumI32Gaudi2Test") ==0))))
    {
        c2tacoarraysumi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacoarraysumi32gaudi2ins.runTest(m);
        c2tacoarraysumi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoArrayCubeI32Gaudi2Test c2tacoarraycubei32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoArrayCubeI32Gaudi2Test") ==0))))
    {
        c2tacoarraycubei32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacoarraycubei32gaudi2ins.runTest(m);
        c2tacoarraycubei32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoArrayFourthI32Gaudi2Test c2tacoarrayfourthi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoArrayFourthI32Gaudi2Test") ==0))))
    {
        c2tacoarrayfourthi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacoarrayfourthi32gaudi2ins.runTest(m);
        c2tacoarrayfourthi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwiseSubI32Gaudi2Test c2tacosubeqi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwiseSubI32Gaudi2Test") ==0))))
    {
        c2tacosubeqi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacosubeqi32gaudi2ins.runTest(m);
        c2tacosubeqi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwiseSubSquareI32Gaudi2Test c2tacosubsqi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwiseSubSquareI32Gaudi2Test") ==0))))
    {
        c2tacosubsqi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacosubsqi32gaudi2ins.runTest(m);
        c2tacosubsqi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwiseMulI32Gaudi2Test c2tacomuleqi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwiseMulI32Gaudi2Test") ==0))))
    {
        c2tacomuleqi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacomuleqi32gaudi2ins.runTest(m);
        c2tacomuleqi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwisePlusI32Gaudi2Test c2tacopluseqi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwisePlusI32Gaudi2Test") ==0))))
    {
        c2tacopluseqi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacopluseqi32gaudi2ins.runTest(m);
        c2tacopluseqi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwiseDivI32Gaudi2Test c2tacodiveqi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwiseDivI32Gaudi2Test") ==0))))
    {
        c2tacodiveqi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacodiveqi32gaudi2ins.runTest(m);
        c2tacodiveqi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwiseMulAddI32Gaudi2Test c2tacomuladdi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwiseMulAddI32Gaudi2Test") ==0))))
    {
        c2tacomuladdi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        result = c2tacomuladdi32gaudi2ins.runTest(m);
        c2tacomuladdi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwisePlusScalarMulI32Gaudi2Test c2tacoplussmuli32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwisePlusScalarMulI32Gaudi2Test") ==0))))
    {
        c2tacoplussmuli32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        int scalar = atoi(getenv("BLEND_S"));
        result = c2tacoplussmuli32gaudi2ins.runTest(m, scalar);
        c2tacoplussmuli32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwiseMatrixAddI32Gaudi2Test c2tacomataddi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwiseMatrixAddI32Gaudi2Test") ==0))))
    {
        c2tacomataddi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = c2tacomataddi32gaudi2ins.runTest(m, n);
        c2tacomataddi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoElemwiseMatrixSubI32Gaudi2Test c2tacomatsubi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoElemwiseMatrixSubI32Gaudi2Test") ==0))))
    {
        c2tacomatsubi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        result = c2tacomatsubi32gaudi2ins.runTest(m, n);
        c2tacomatsubi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoMatScalarMulI32Gaudi2Test c2tacomatscalarmuli32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoMatScalarMulI32Gaudi2Test") ==0))))
    {
        c2tacomatscalarmuli32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        size_t n = atoi(getenv("BLEND_N"));
        int scalar = atoi(getenv("BLEND_S"));
        result = c2tacomatscalarmuli32gaudi2ins.runTest(m, n, scalar);
        c2tacomatscalarmuli32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoVecScalarAddI32Gaudi2Test c2tacovecscalaraddi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoVecScalarAddI32Gaudi2Test") ==0))))
    {
        c2tacovecscalaraddi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        int scalar = atoi(getenv("BLEND_S"));
        result = c2tacovecscalaraddi32gaudi2ins.runTest(m, scalar);
        c2tacovecscalaraddi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }
    

    C2TacoVecScalarSubI32Gaudi2Test c2tacovecscalarsubi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoVecScalarSubI32Gaudi2Test") ==0))))
    {
        c2tacovecscalarsubi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        int scalar = atoi(getenv("BLEND_S"));
        result = c2tacovecscalarsubi32gaudi2ins.runTest(m, scalar);
        c2tacovecscalarsubi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoVecScalarMulI32Gaudi2Test c2tacovecscalarmuli32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoVecScalarMulI32Gaudi2Test") ==0))))
    {
        c2tacovecscalarmuli32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        int scalar = atoi(getenv("BLEND_S"));
        result = c2tacovecscalarmuli32gaudi2ins.runTest(m, scalar);
        c2tacovecscalarmuli32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    C2TacoVecScalarDivI32Gaudi2Test c2tacovecscalardivi32gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"C2TacoVecScalarDivI32Gaudi2Test") ==0))))
    {
        c2tacovecscalardivi32gaudi2ins.SetUp();
        size_t m = atoi(getenv("BLEND_M"));
        int scalar = atoi(getenv("BLEND_S"));
        result = c2tacovecscalardivi32gaudi2ins.runTest(m, scalar);
        c2tacovecscalardivi32gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }



    std::cout << "All " << testCount  <<" tests pass!" <<std::endl;
    return 0;
}
