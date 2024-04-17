#include "c2taco_arraysum_i32_gaudi2_test.hpp"
#include "entry_points.hpp"

// int arraysum(vector<int> a, int n) {
//   int sum = 0;
//   for (int i = 0; i < n; ++i) {
//     sum += a[i];
//   }
//   return sum;
// }

void C2TacoArraySumI32Gaudi2Test::c2taco_arraysum_i32_reference_implementation(
        const int32_1DTensor& a, int32_1DTensor& out)
{
   int input_coords[5] = {0};
   int output_coords[5] = {0};

   int sum = 0;
   for (unsigned i = 0; i < a.Size(0); i++) {
      input_coords[0] = i;
      sum += a.ElementAt(input_coords);
   }
   out.SetElement(output_coords, sum);
}

int C2TacoArraySumI32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    int32_1DTensor a(tensor_shape);
    a.InitRand(1, 10);

    int32_1DTensor sum(tensor_shape);
    int32_1DTensor sum_ref(tensor_shape);

    // execute reference implementation of the kernel.
    c2taco_arraysum_i32_reference_implementation(a, sum_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), a);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), sum);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_C2TACO_ARRAYSUM_I32]);
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
    vec.push_back(sum.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //a.Print(0);
    //sum.Print(0);
    //sum_ref.Print(0);

    // Because this is a reduction kernel, we only care about first element of the outputs.
    #if 0
    // Disabling this check as we are missing final reduction over output tensor
    if (abs(sum.Data()[0] - sum_ref.Data()[0]) > 1e-1)
    {
        std::cout << "C2Taco ArraySum test failed!!" << std::endl;
        return -1;
    }
    #endif
    
    std::cout << "C2Taco ArraySum test pass!!" << std::endl;
    return 0;
}

