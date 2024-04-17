#include "c2taco_elemwise_matrix_sub_i32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<vector<int>> matsub(vector<vector<int>> matA, vector<vector<int>> matB, int m, int n) {
//     vector<vector<int>> out;
//     for (int i = 0; i < m; ++i) {
//         vector<int> row_vec;
//         for (int j = 0; j < n; ++j) {
//             row_vec.push_back(matA[i][j] - matB[i][j]);
//         }
//         out.push_back(row_vec);
//     }
//     return out;
// }

void C2TacoElemwiseMatrixSubI32Gaudi2Test::elemwise_matrix_sub_i32_reference_implementation(
        const int32_2DTensor& a, const int32_2DTensor& b, int32_2DTensor& out)
{
    int coords[5] = {0};

    int maxRows = out.Size(0);
    int maxCols = out.Size(1);

    for (int row = 0; row < maxRows; row++) {
        for (int col = 0; col < maxCols; col++) {
            coords[0] = row; coords[1] = col;
            out.SetElement(coords, a.ElementAt(coords) - b.ElementAt(coords));
        }
    }
}

int C2TacoElemwiseMatrixSubI32Gaudi2Test::runTest(uint32_t m, uint32_t n)
{
    // 2D matrix of size 128x3
    // If the first dimension is multiple of 64, the test delivers optimal result
    // (most likely because 64-elements needs to be contiguous to read as a vec)
    // If I change the shape to 3x128, then test delivers poor result - no
    // vector operation.
    unsigned int tensor_shape[] = {m, n};

    int32_2DTensor a(tensor_shape);
    a.InitRand(0, 10);

    int32_2DTensor b(tensor_shape);
    b.InitRand(0, 10);

    int32_2DTensor out(tensor_shape);
    int32_2DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    elemwise_matrix_sub_i32_reference_implementation(a, b, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), a);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), b);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), out);

    char**   kernelNames = nullptr;
    unsigned kernelCount = 0;
    gcapi::GlueCodeReturn_t result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    kernelNames = new char*[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[gcapi::MAX_NODE_NAME];
    }    
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_C2TACO_ELEMWISE_MATRIX_SUB_I32]);
    result  = HabanaKernel(&m_in_defs, &m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(a.GetTensorDescriptor());
    vec.push_back(b.GetTensorDescriptor());
    vec.push_back(out.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //a.Print(0);
    //b.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (out.Data()[element] != out_ref.Data()[element])
        {
            std::cout << "C2Taco Elemwise Matrix Sub test failed (at: " << element
                      << ", value: " << out.Data()[element] << ", exp:" << out_ref.Data()[element] << std::endl;
            return -1;
        }
    }
    std::cout << "C2Taco Elemwise Matrix Sub test pass!!" << std::endl;
    return 0;
}

