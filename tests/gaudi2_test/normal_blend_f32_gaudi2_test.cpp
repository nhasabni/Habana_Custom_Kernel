#include "normal_blend_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

//
// void normalBlendf (Buffer<float,1> base, Buffer<float,1> active, Buffer<float,1> out, float opacity)
// {
// 	for (int pixel=0; pixel<out.width(); pixel++) {
// 		out(pixel) = opacity * active(pixel) + (1.0f - opacity) * base(pixel);
// 	}
// }

void NormalBlendF32Gaudi2Test::normalblend_f32_reference_implementation(
        const float_1DTensor& base,
        const float_1DTensor& active,
        float_1DTensor& out,
        NormalBlendF32Gaudi2::NormalBlendParam& param_def)
{
   int coords[5] = {0};
   float opacity = param_def.opacity;

   for (unsigned pixel = 0; pixel < base.Size(0); pixel++) {
      coords[0] = pixel;
      float y = opacity * active.ElementAt(coords) +
                  (1.0f - opacity) * base.ElementAt(coords);
      out.SetElement(coords, y);
   }
}

int NormalBlendF32Gaudi2Test::runTest(uint32_t m)
{
    // a vector of 8k elements.
    unsigned int tensor_shape[] = {m};

    float_1DTensor base(tensor_shape);
    base.InitRand(-10.0f, 10.0f);

    float_1DTensor active(tensor_shape);
    active.InitRand(-10.0f, 10.0f);

    float_1DTensor out(tensor_shape);
    float_1DTensor out_ref(tensor_shape);

    // Params
    NormalBlendF32Gaudi2::NormalBlendParam param_def;
    param_def.opacity = 1;

    // execute reference implementation of the kernel.
    normalblend_f32_reference_implementation(base, active, out_ref, param_def);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_NORMAL_BLEND_F32]);
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
    //base.Print(0);
    //active.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Normal Blend F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Normal Blend F32 test pass!!" << std::endl;
    return 0;
}

