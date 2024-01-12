#include "llama_silu_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<float> silu(vector<float> input, int hidden_dim) {
//     vector<float> output;
//     for (int i = 0; i < hidden_dim; i++) {
//         float curr = 1 / (1 + sqrt(input[i])) * input[i];
//         output.push_back(curr);
//     }
//     return output;
// }

void LlamaSiluF32Gaudi2Test::llamasilu_f32_reference_implementation(
        const float_1DTensor& in, float_1DTensor& out)
{
   int coords[5] = {0};
   unsigned int hidden_dim = in.Size(0);

   for (unsigned i = 0; i < hidden_dim; i++) {
        coords[0] = i;
        float cur = in.ElementAt(coords) / (1 + std::sqrt(in.ElementAt(coords)));
        out.SetElement(coords, cur);
   }
}

int LlamaSiluF32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    float_1DTensor in(tensor_shape);
    in.InitRand(0.0, 100.0);

    float_1DTensor out(tensor_shape);
    float_1DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    llamasilu_f32_reference_implementation(in, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_LLAMA_SILU_F32]);
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
        // account for rounding differences
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1e-6)
        {
            std::cout << "LLaMa Silu F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "LLaMa Silu F32 test pass!!" << std::endl;
    return 0;
}

