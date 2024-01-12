#include "llama_rmsnorm_part1_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

// float rmsnorm_part1(vector<float> input, vector<float> weight) {
//     float ss = 0;
//     for (int i = 0; i < input.size(); i++)
//         ss += input[i] * input[i];
//     return ss;
// }

void LlamaRmsnormPart1F32Gaudi2Test::llamarmsnormpart1_f32_reference_implementation(
        const float_1DTensor& input, const float_1DTensor& weight,
        float_1DTensor& out)
{
   int input_coords[5] = {0};
   int output_coords[5] = {0};

   float ss = 0.0;
   for (unsigned i = 0; i < input.Size(0); i++) {
      input_coords[0] = i;
      ss += (input.ElementAt(input_coords) * input.ElementAt(input_coords));
   }
   out.SetElement(output_coords, ss);
}

int LlamaRmsnormPart1F32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    float_1DTensor input(tensor_shape);
    input.InitRand(-1, 1);

    float_1DTensor weight(tensor_shape);
    weight.InitRand(-1, 1);

    float_1DTensor out(tensor_shape);
    float_1DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    llamarmsnormpart1_f32_reference_implementation(input, weight, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);
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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_LLAMA_RMSNORM_PART1_F32]);
    result  = HabanaKernel(&m_in_defs, &m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(weight.GetTensorDescriptor());
    vec.push_back(out.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    input.Print(0);
    out.Print(0);
    out_ref.Print(0);

    // Because this is a reduction kernel, we only care about first element of the outputs.
    // Need to account for floating point adds.
    if (abs(out.Data()[0] - out_ref.Data()[0]) > 1e-1)
    {
        std::cout << "LLaMa Rmsnorm Part1 test failed!!" << std::endl;
        return -1;
    }
    
    std::cout << "LLaMa Rmsnorm Part1 test pass!!" << std::endl;
    return 0;
}

