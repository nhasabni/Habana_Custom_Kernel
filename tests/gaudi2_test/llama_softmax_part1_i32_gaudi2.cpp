#include "normal_blend_u8_gaudi2_test.hpp"
#include "entry_points.hpp"

// int softmax_part1(vector<int> input, int max_pos) {
//     int max_val = input[0];
//     for (int i = 1; i < max_pos; i++)
//         if (input[i] > max_val)
//             max_val = input[i];
//     return max_val;
// }
// def softmax_part1_ps(input max_pos softmax_part1_rv)
// softmax_part1_rv == reduce_max(list_take(input, max_pos))

void LlamaSoftmaxPart1I32Gaudi2::llamasoftmaxpart1_i32_reference_implementation(
        const int32_1DTensor& in, int32_1DTensor& out)
{
   int coords[5] = {0};

   int32_t max_val = INT_MIN;
   for (unsigned i = 0; i < in.Size(0); i++) {
      coords[0] = i;
      if (max_val < in.ElementAt(coords))
        max_val = in.ElementAt(coords);
   }
   out.SetElement(coords, max_val);
}

int LlamaSoftmaxPart1I32Gaudi2::runTest()
{
    // a vector of 8k elements.
    const int width  = 256;
    unsigned int tensor_shape[] = {width};

    uint8_1DTensor base(tensor_shape);
    base.InitRand(0, 255);

    uint8_1DTensor active(tensor_shape);
    active.InitRand(0, 255);

    uint16_1DTensor out(tensor_shape);
    uint8_1DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    llamasoftmaxpart1_i32_reference_implementation(in, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.NodeParams = &param_def;
    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), base);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), active);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_NORMAL_BLEND_U8]);
    result  = HabanaKernel(&m_in_defs, &m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(base.GetTensorDescriptor());
    vec.push_back(active.GetTensorDescriptor());
    vec.push_back(out.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    base.Print(0);
    active.Print(0);
    out.Print(0);
    out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Normal Blend U8 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Normal Blend U8 test pass!!" << std::endl;
    return 0;
}

