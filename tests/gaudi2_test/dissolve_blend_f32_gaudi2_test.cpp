#include "dissolve_blend_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

// void dissolveBlend8 (Buffer<float,2> base, Buffer<float,2> active, Buffer<float,2> out, float opacity)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			float rand_val = ((rand() % 100) + 1) / 100.0f;
// 			if (opacity - rand_val >= 0.0f)
// 				out(col,row) = active(col,row);
// 			else
// 				out(col,row) = base(col,row);
// 		}
// 	}
// }

void DissolveBlendF32Gaudi2Test::dissolveblend_f32_reference_implementation(
        const float_2DTensor& base,
        const float_2DTensor& active,
        float_2DTensor& out)
{
    int coords[5] = {0};

    int maxRows = out.Size(0);
    int maxCols = out.Size(1);

    for (int row = 0; row < maxRows; row++) {
        for (int col = 0; col < maxCols; col++) {
            coords[0] = row; coords[1] = col;
            if (base.ElementAt(coords) > active.ElementAt(coords)) {
                out.SetElement(coords, active.ElementAt(coords));
            } else {
                out.SetElement(coords, base.ElementAt(coords));
            }
        }
    }
}

int DissolveBlendF32Gaudi2Test::runTest()
{
    // 2D matrix of size 128x3
    // If the first dimension is multiple of 64, the test delivers optimal result
    // (most likely because 64-elements needs to be contiguous to read as a vec)
    // If I change the shape to 3x128, then test delivers poor result - no
    // vector operation.
    const int dim0  = 128;
    const int dim1  = 64;
    unsigned int tensor_shape[] = {dim0, dim1};

    float_2DTensor base(tensor_shape);
    base.InitRand(-10.0f, 10.0f);

    float_2DTensor active(tensor_shape);
    active.InitRand(-10.0f, 10.0f);

    float_2DTensor out(tensor_shape);
    float_2DTensor out_ref(tensor_shape);

    // execute reference implementation of the kernel.
    dissolveblend_f32_reference_implementation(base, active, out_ref);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_DISSOLVE_BLEND_F32]);
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
    // out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (abs(out.Data()[element] - out_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Darken Blend F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Darken Blend F32 test pass!!" << std::endl;
    return 0;
}

