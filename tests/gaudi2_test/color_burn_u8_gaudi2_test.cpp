#include "color_burn_u8_gaudi2_test.hpp"
#include "entry_points.hpp"

// void colorBurn8 (Buffer<uint8_t,2> base, Buffer<uint8_t,2> active, Buffer<uint8_t,2> out)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			if (active(col, row) == 0)
// 				out(col,row) = 255;
// 			else
// 				out(col,row) = 255 - (255 - base(col,row)) / active(col,row);
// 		}
// 	}
// }

void ColorBurnU8Gaudi2Test::colorburn_u8_reference_implementation(
        const uint8_2DTensor& base,
        const uint8_2DTensor& active,
        uint8_2DTensor& out)
{
    int coords[5] = {0};

    int maxRows = out.Size(0);
    int maxCols = out.Size(1);

    for (int row = 0; row < maxRows; row++) {
        for (int col = 0; col < maxCols; col++) {
            coords[0] = row; coords[1] = col;
            if (active.ElementAt(coords) == 0) {
                out.SetElement(coords, (uint8_t) 255);
            } else {
                uint8_t x = (uint8_t) 255 - (((uint8_t) 255 - base.ElementAt(coords)) /
                                                                active.ElementAt(coords));
                out.SetElement(coords, x);
            }
        }
    }
}

int ColorBurnU8Gaudi2Test::runTest()
{
    // 2D matrix of size 128x3
    // If the first dimension is multiple of 64, the test delivers optimal result
    // (most likely because 64-elements needs to be contiguous to read as a vec)
    // If I change the shape to 3x128, then test delivers poor result - no
    // vector operation.
    unsigned int tensor_shape[] = {128, 3};

    uint8_2DTensor base(tensor_shape);
    base.InitRand(0, 255);

    uint8_2DTensor active(tensor_shape);
    active.InitRand(0, 255);

    uint8_2DTensor out(tensor_shape);
    uint8_2DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    colorburn_u8_reference_implementation(base, active, out_ref);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_COLOR_BURN_U8]);
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
            std::cout << "Color Burn U8 test failed!! (index:" << element << ")" << std::endl;
            return -1;
        }
    }
    std::cout << "Color Burn U8 test pass!!" << std::endl;
    return 0;
}

