#include "llama_elemwise_mul_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

// vector<float> elemwise_mul(vector<float> input1, vector<float> input2, int hidden_dim) {
//     vector<float> output;
//     for (int i = 0; i < hidden_dim; i++) {
//         output.push_back(input1[i] * input2[i]);
//     }
//     return output;
// }

void LlamaElemwiseMulF32Gaudi2Test::llamaelemwisemul_f32_reference_implementation(
        const float_1DTensor& input1,
        const float_1DTensor& input2,
        float_1DTensor& out)
{
   int coords[5] = {0};
   unsigned int hidden_dim = input1.Size(0);

   for (unsigned int i = 0; i < hidden_dim; i++) {
        coords[0] = i;
        out.SetElement(coords, input1.ElementAt(coords) * input2.ElementAt(coords));
   }
}

int LlamaElemwiseMulF32Gaudi2Test::runTest(uint32_t m)
{
    // a vector of 8k elements.
    unsigned int tensor_shape[] = {m};

    float_1DTensor input1(tensor_shape);
    input1.InitRand(-10.0f, 10.0f);

    float_1DTensor input2(tensor_shape);
    input2.InitRand(-10.0f, 10.0f);

    float_1DTensor out(tensor_shape);
    float_1DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    llamaelemwisemul_f32_reference_implementation(input1, input2, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input1);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input2);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_LLAMA_ELEMWISE_MUL_F32]);
    result  = HabanaKernel(&m_in_defs, &m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(input1.GetTensorDescriptor());
    vec.push_back(input2.GetTensorDescriptor());
    vec.push_back(out.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    //input1.Print(0);
    //input2.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1e-6)
        {
            std::cout << "LLama Elemwise Mul F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "LLama Elemwise Mul F32 test pass!!" << std::endl;
    return 0;
}

