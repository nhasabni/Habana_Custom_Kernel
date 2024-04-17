#include "c2taco_elemwise_mul_add_i32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<int> n_real_updates(int N, vector<int> A, vector<int> B, vector<int> C) {
//     vector<int> D;
//     for (int i = 0; i < N; i++) {
//         int curr = A[i] + B[i] * C[i];
//         D.push_back(curr);
//     }
//     return D;
// }

void C2TacoElemwiseMulAddI32Gaudi2Test::c2taco_elemwise_mul_add_i32_reference_implementation(
        const int32_1DTensor& a, const int32_1DTensor& b, const int32_1DTensor& c,
        int32_1DTensor& out)
{
   int coords[5] = {0};
   unsigned int hidden_dim = a.Size(0);

   for (unsigned int i = 0; i < hidden_dim; i++) {
        coords[0] = i;
        out.SetElement(coords, a.ElementAt(coords) + (b.ElementAt(coords) * c.ElementAt(coords)));
   }
}

int C2TacoElemwiseMulAddI32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    int32_1DTensor a(tensor_shape);
    a.InitRand(1, 10);

    int32_1DTensor b(tensor_shape);
    b.InitRand(1, 10);

    int32_1DTensor c(tensor_shape);
    c.InitRand(1, 10);

    int32_1DTensor out(tensor_shape);
    int32_1DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    c2taco_elemwise_mul_add_i32_reference_implementation(a, b, c, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 3;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), a);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), b);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), c);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_C2TACO_ELEMWISE_MUL_ADD_I32]);
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
    vec.push_back(c.GetTensorDescriptor());
    vec.push_back(out.GetTensorDescriptor());
    // execute a simul_addation of the kernel using TPC simul_addator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //a.Print(0);
    //b.Print(0);
    //c.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (out.Data()[element] != out_ref.Data()[element])
        {
            std::cout << "C2Taco Elemwise Mul Add I32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "C2Taco Elemwise Mul Add I32 test pass!!" << std::endl;
    return 0;
}

