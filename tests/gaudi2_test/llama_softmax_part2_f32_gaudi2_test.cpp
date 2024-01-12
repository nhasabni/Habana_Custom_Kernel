#include "llama_softmax_part2_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<float> softmax_part2(vector<float> input, int max_pos, float max_val) {
//     vector<int> output;
//     for (int i = 0; i < max_pos; i++) {
//         float cur = exp(input[i] - max_val);
//         output.push_back(cur);
//     }
//     return output;
// }

void LlamaSoftmaxPart2F32Gaudi2Test::llamasoftmaxpart2_f32_reference_implementation(
        const float_1DTensor& in, float max_val, float_1DTensor& out)
{
   int coords[5] = {0};

   for (unsigned i = 0; i < in.Size(0); i++) {
        coords[0] = i;
        float cur = std::exp(in.ElementAt(coords) - max_val);
        out.SetElement(coords, cur);
   }
}

int LlamaSoftmaxPart2F32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    float_1DTensor in(tensor_shape);
    in.InitRand(0.0, 1.0);

    float_1DTensor out(tensor_shape);
    float_1DTensor out_ref(tensor_shape);

    // params
    LlamaSoftmaxPart2F32Gaudi2::LlamaSoftmaxPart2Param param_def;
    param_def.max_val = 0.1;

    // execute reference implementation of the kernel.
    llamasoftmaxpart2_f32_reference_implementation(in, param_def.max_val, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.NodeParams = &param_def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), in);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_LLAMA_SOFTMAX_PART2_F32]);
    result  = HabanaKernel(&m_in_defs, &m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(in.GetTensorDescriptor());
    vec.push_back(out.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //in.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1e-6)
        {
            std::cout << "LLaMa Softmax Part2 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "LLaMa Softmax Part2 test pass!!" << std::endl;
    return 0;
}

