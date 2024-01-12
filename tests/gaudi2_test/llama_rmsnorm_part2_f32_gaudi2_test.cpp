#include "llama_rmsnorm_part2_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<float> rmsnorm_part2(vector<float> input, vector<float> weight, float ss) {
//     vector<float> output;
//     int size = input.size();
//     float inv_ss = 1 / sqrt(ss / size + 1);
//     for (int i = 0; i < input.size(); i++)
//         output.push_back(inv_ss * input[i] * weight[i]);
//     return output;
// }

void LlamaRmsnormPart2F32Gaudi2Test::llamarmsnormpart2_f32_reference_implementation(
        const float_1DTensor& in, const float_1DTensor& weight, float ss, float_1DTensor& out)
{
   int coords[5] = {0};

   int size = in.Size(0);
   float inv_ss = 1 / std::sqrt(ss / (size + 1));

   for (unsigned i = 0; i < in.Size(0); i++) {
        coords[0] = i;
        float r = inv_ss * in.ElementAt(coords) * weight.ElementAt(coords);
        out.SetElement(coords, r);
   }
}

int LlamaRmsnormPart2F32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    float_1DTensor in(tensor_shape);
    in.InitRand(-100, 100);

    float_1DTensor weight(tensor_shape);
    weight.InitRand(-100, 100);

    float_1DTensor out(tensor_shape);
    float_1DTensor out_ref(tensor_shape);

    // params
    LlamaRmsnormPart2F32Gaudi2::LlamaRmsnormPart2Param param_def;
    param_def.ss = (float) 64;

    // execute reference implementation of the kernel.
    llamarmsnormpart2_f32_reference_implementation(in, weight, param_def.ss, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.NodeParams = &param_def;
    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), in);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), weight);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_LLAMA_RMSNORM_PART2_F32]);
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
    vec.push_back(weight.GetTensorDescriptor());
    vec.push_back(out.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //in.Print(0);
    //weight.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        // account for rounding differences
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1e-4)
        {
            std::cout << "LLaMa RMSNorm Part2 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "LLaMa RMSNorm Part2 test pass!!" << std::endl;
    return 0;
}

