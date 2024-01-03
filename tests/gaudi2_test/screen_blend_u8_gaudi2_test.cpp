#include "screen_blend_u8_gaudi2_test.hpp"
#include "entry_points.hpp"

// inline uint8_t Mul8x8Div255 (uint8_t a, uint8_t b)
// {
// 	return (a * b) / 255;
// }
// inline uint8_t Screen8x8 (uint8_t a, uint8_t b)
// {
// 	return a + b - Mul8x8Div255(a, b);
// }
// void screenBlend8 (Buffer<uint8_t,2> base, Buffer<uint8_t,2> active, Buffer<uint8_t,2> out)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			out(col,row) = Screen8x8(base(col,row), active(col,row));
// 		}
// 	}
// }

void ScreenBlendU8Gaudi2Test::screenblend_u8_reference_implementation(
        const uint8_1DTensor& base,
        const uint8_1DTensor& active,
        uint8_1DTensor& out)
{
   int coords[5] = {0};

   for (unsigned pixel = 0; pixel < base.Size(0); pixel++) {
      coords[0] = pixel;
      uint8_t x = (active.ElementAt(coords) * base.ElementAt(coords)) / 255;
      uint8_t y = active.ElementAt(coords) + base.ElementAt(coords) - x;
      out.SetElement(coords, y);
   }
}

int ScreenBlendU8Gaudi2Test::runTest()
{
    // a vector of 8k elements.
    const int width  = 256;
    unsigned int tensor_shape[] = {width};

    uint8_1DTensor base(tensor_shape);
    base.InitRand(0, 255);

    uint8_1DTensor active(tensor_shape);
    active.InitRand(0, 255);

    uint8_1DTensor out(tensor_shape);
    uint8_1DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    screenblend_u8_reference_implementation(base, active, out_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_SCREEN_BLEND_U8]);
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
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1)
        {
            std::cout << "Screen Blend U8 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Screen Blend U8 test pass!!" << std::endl;
    return 0;
}

