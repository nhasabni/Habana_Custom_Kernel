#include "c2taco_vecscalar_add_i32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<int> array_inc(vector<int> arr, int n) {
//     vector<int> out;
//     for (int i = 0; i < n; ++i) {
//         out.push_back(arr[i] + 1);
//     }
//     return out;
// }

void C2TacoVecScalarAddI32Gaudi2Test::c2taco_vecscalar_add_i32_reference_implementation(
        const int32_1DTensor& a, int32_1DTensor& out,
        C2TacoVecScalarAddI32Gaudi2::Param& param_def)
{
   int coords[5] = {0};
   int scalar = param_def.scalar;

   unsigned int n = a.Size(0);

   for (unsigned int i = 0; i < n; i++) {
        coords[0] = i;
        out.SetElement(coords, a.ElementAt(coords) + scalar);
   }
}

int C2TacoVecScalarAddI32Gaudi2Test::runTest(uint32_t m, int scalar)
{
    // a vector of 8k elements.
    unsigned int tensor_shape[] = {m};

    int32_1DTensor a(tensor_shape);
    a.InitRand(1, 10);

    int32_1DTensor out(tensor_shape);
    int32_1DTensor out_ref(tensor_shape);

    // Params
    C2TacoVecScalarAddI32Gaudi2::Param param_def;
    param_def.scalar = scalar;

    // execute reference implementation of the kernel.
    c2taco_vecscalar_add_i32_reference_implementation(a, out_ref, param_def);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.NodeParams = &param_def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), a);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_C2TACO_VECSCALAR_ADD_I32]);
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
    vec.push_back(out.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //a.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (out.Data()[element] != out_ref.Data()[element])
        {
            std::cout << "C2Taco VecScalar Add I32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "C2Taco VecScalar Add I32 test pass!!" << std::endl;
    return 0;
}
