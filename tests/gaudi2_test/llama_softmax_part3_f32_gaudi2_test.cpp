#include "llama_softmax_part3_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

// float softmax_part3(vector<float> output, int max_pos) {
//     float sum = 0;
//     for (int i = 0; i < max_pos; i++) {
//         sum += output[i];
//     }
//     return sum;
// }

void LlamaSoftmaxPart3F32Gaudi2Test::llamasoftmaxpart3_f32_reference_implementation(
        const float_1DTensor& in, float_1DTensor& out)
{
   int input_coords[5] = {0};
   int output_coords[5] = {0};

   float sum = 0;
   for (unsigned i = 0; i < in.Size(0); i++) {
      input_coords[0] = i;
      sum += in.ElementAt(input_coords);
   }
   out.SetElement(output_coords, sum);
}

int LlamaSoftmaxPart3F32Gaudi2Test::runTest(uint32_t m)
{
    unsigned int tensor_shape[] = {m};

    float_1DTensor in(tensor_shape);
    in.InitRand(-10, 10);

    float_1DTensor out(tensor_shape);
    float_1DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    llamasoftmaxpart3_f32_reference_implementation(in, out_ref);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_LLAMA_SOFTMAX_PART3_F32]);
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

    // Because this is a reduction kernel, we only care about first element of the outputs.
    if (abs(out.Data()[0] - out_ref.Data()[0]) > 1e-2)
    {
        std::cout << "LLaMa Softmax Part3 test failed !!" << std::endl;
        return -1;
    }
    
    std::cout << "LLaMa Softmax Part3 test pass!!" << std::endl;
    return 0;
}

