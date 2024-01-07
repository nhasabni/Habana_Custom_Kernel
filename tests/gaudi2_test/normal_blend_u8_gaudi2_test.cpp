#include "normal_blend_u8_gaudi2_test.hpp"
#include "entry_points.hpp"

// inline uint8_t Mul8x8Div255 (uint8_t a, uint8_t b)
// {
// 	return (a * b) / 255;
// }
// void normalBlend8 (Buffer<uint8_t,1> base, Buffer<uint8_t,1> active, Buffer<uint8_t,1> out, uint8_t opacity)
// {
// 	for (int pixel=0; pixel<out.width(); pixel++) {
// 		out(pixel) = Mul8x8Div255(opacity, active(pixel)) + Mul8x8Div255(255 - opacity, base(pixel));
// 	}
// }

void NormalBlendU8Gaudi2Test::normalblend_u8_reference_implementation(
        const uint8_1DTensor& base,
        const uint8_1DTensor& active,
        uint8_1DTensor& out,
        NormalBlendU8Gaudi2::NormalBlendParam& param_def)
{
   int coords[5] = {0};
   uint8_t opacity = param_def.opacity;

   for (unsigned pixel = 0; pixel < base.Size(0); pixel++) {
      coords[0] = pixel;
      uint8_t y = (opacity * active.ElementAt(coords)) / 255 +
                  ((255 - opacity) * base.ElementAt(coords)) / 255;
      out.SetElement(coords, y);
   }
}

int NormalBlendU8Gaudi2Test::runTest(uint32_t m)
{
    // a vector of 8k elements.
    unsigned int tensor_shape[] = {m};

    uint8_1DTensor base(tensor_shape);
    base.InitRand(0, 255);

    uint8_1DTensor active(tensor_shape);
    active.InitRand(0, 255);

    uint8_1DTensor out(tensor_shape);
    uint8_1DTensor out_ref(tensor_shape);

    // Params
    NormalBlendU8Gaudi2::NormalBlendParam param_def;
    param_def.opacity = (unsigned char) 21;

    // execute reference implementation of the kernel.
    normalblend_u8_reference_implementation(base, active, out_ref, param_def);

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
    //base.Print(0);
    //active.Print(0);
    //out.Print(0);
    //out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        // - Floating point based division in Gaudi leading to precision issues.
        // E.g., for NormalBlendU8 when both base and active=255, and opacity=21.
        // Equation says: (opacity * active)/255 + ((255 - opacity) * base)/255 = 21 + 234 = 255.
        // But when float type is used in Gaudi, it first calculates 1/255, and then
        // multiplies it with (21 * 255), which produces 20.99, while (234 * 255)/255 would produce
        // 233.99. If we round this nearest numbers, it works fine. But then ref code is not using float
        // type for division and hence would round 20.99 to 20, and 233.99 to 233 instead of 234.
        // This leads to max difference of 2 in out and out_ref.
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 2)
        {
            std::cout << "Normal Blend U8 test failed (element:" << element << ")" << std::endl;
            std::cout << "b:" << (uint32_t) base.Data()[element] << ",a:" << (uint32_t) active.Data()[element] << std::endl;
            std::cout << "out: " << (uint32_t) out.Data()[element] << ", out_ref:" << (uint32_t) out_ref.Data()[element] << std::endl;
            return -1;
        }
    }
    std::cout << "Normal Blend U8 test pass!!" << std::endl;
    return 0;
}

