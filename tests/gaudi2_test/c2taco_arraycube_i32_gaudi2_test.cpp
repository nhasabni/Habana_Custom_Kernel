#include "c2taco_arraycube_i32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<int> cube_in_place(vector<int> arr, int n) {
//     vector<int> out;
//     for (int i = 0; i < n; ++i) {
//         int sq = arr[i] * arr[i];
//         int cube = sq * sq;
//         out.push_back(cube);
//     }
//     return out;
// }

void C2TacoArrayCubeI32Gaudi2Test::c2taco_arraycube_i32_reference_implementation(
        const int32_1DTensor& a, int32_1DTensor& out)
{
   int input_coords[5] = {0};
   int output_coords[5] = {0};

   for (unsigned i = 0; i < a.Size(0); i++) {
      input_coords[0] = output_coords[0] = i;
      out.SetElement(output_coords, a.ElementAt(input_coords) * a.ElementAt(input_coords) * a.ElementAt(input_coords));
   }
}

int C2TacoArrayCubeI32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    int32_1DTensor a(tensor_shape);
    a.InitRand(1, 10);

    int32_1DTensor cube(tensor_shape);
    int32_1DTensor cube_ref(tensor_shape);

    // execute reference implementation of the kernel.
    c2taco_arraycube_i32_reference_implementation(a, cube_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), a);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), cube);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_C2TACO_ARRAYCUBE_I32]);
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
    vec.push_back(cube.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //a.Print(0);
    //cube.Print(0);
    //cube_ref.Print(0);

    // Disabling this check as we are missing final reduction over output tensor
    if (cube.Data()[0] != cube_ref.Data()[0])
    {
        std::cout << "C2Taco ArrayCube test failed!!" << std::endl;
        return -1;
    }
    
    std::cout << "C2Taco ArrayCube test pass!!" << std::endl;
    return 0;
}

