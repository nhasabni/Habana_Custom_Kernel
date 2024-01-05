#include <vector>
#include <cstring>
#include <iostream>
#include "dissolve_blend_f32_gaudi2.hpp"

extern unsigned char _binary___dissolve_blend_f32_gaudi2_o_start;
extern unsigned char _binary___dissolve_blend_f32_gaudi2_o_end;

 gcapi::GlueCodeReturn_t DissolveBlendF32Gaudi2::GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_dissolve_blend_f32_gaudi2");
     return gcapi::GLUE_SUCCESS;
 }


gcapi::GlueCodeReturn_t DissolveBlendF32Gaudi2::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* in_defs,
            gcapi::HabanaKernelInstantiation_t* out_defs)
{
    gcapi::GlueCodeReturn_t retVal;
    DissolveBlendParam* param_def = static_cast<DissolveBlendParam*>(in_defs->NodeParams);

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors: we pass rand values as a tensor to this kernel.
    // This is because TPCC does not have intrinsic for generating rand values in a kernel.
    if (in_defs->inputTensorNr != 3)
    {
        in_defs->inputTensorNr  = 3;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr !=1)
    {
        in_defs->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    //validate matrix dimensions
    if ((in_defs->inputTensors[0].geometry.sizes[0] != 
        in_defs->inputTensors[1].geometry.sizes[0]) ||
        (in_defs->inputTensors[1].geometry.sizes[0] != 
        in_defs->inputTensors[2].geometry.sizes[0]))
    {
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    // validate input and output data type
    if (in_defs->inputTensors[0].dataType != gcapi::DATA_F32 ||
        in_defs->inputTensors[1].dataType != gcapi::DATA_F32 ||
        in_defs->inputTensors[2].dataType != gcapi::DATA_F32 ||
        in_defs->outputTensors[0].dataType != gcapi::DATA_F32)
    {
        in_defs->inputTensors[0].dataType = gcapi::DATA_F32;
        in_defs->inputTensors[1].dataType = gcapi::DATA_F32;
        in_defs->inputTensors[2].dataType = gcapi::DATA_F32;
        in_defs->outputTensors[0].dataType = gcapi::DATA_F32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. Output size is same as input size.
    **************************************************************************************/
    // We operate on 2D index space for DarkenBlend because input tensors and output tensors
    // are 2D.
    unsigned int outputSizes[2] = {0, 0};
    memcpy(outputSizes, in_defs->inputTensors[0].geometry.sizes, sizeof(outputSizes));

    // We operate on a block of 64 elements at a time in dim0 only.
    // NOTE however that this may not be optimal blocking in case size of dim0 is less
    // than 64. In that case, we should consider blocking in dim1.
    int elementsInVec = 64;
    out_defs->indexSpaceGeometry.dims = 2;

    // round up to elementsInVec and divide by elementsInVec.
    unsigned maxDim0Index = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    unsigned maxDim1Index = outputSizes[1];
    out_defs->indexSpaceGeometry.sizes[0] = maxDim0Index;
    out_defs->indexSpaceGeometry.sizes[1] = maxDim1Index;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    //
    // Index space to tensor mapping looks like below: each index space operates on a block
    // of 64 elements. 
    //
    // 0                                          64                               128
    //   -------------------------------------------------------------------------------------
    //   |        Index (0,0)                      |            Index (0,1)              | ...
    // 1 -------------------------------------------------------------------------------------
    //   |        Index (1,0)                      |            Index (1,1)              | ...
    // 2 -------------------------------------------------------------------------------------
    // ...

    // 'i' is the index space member and A/B constants to be defined.
    // More details here - https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/TPC_Programming_Model.html#index-space-mapping
    //
    // Index space mapping is calculated using f(x) = start_a * x + start_b to end_a * x + end_b
    //
    gcapi::DimTransform_t dim1_mapping, dim0_mapping;

    // Because start_a and end_a are multipliers and because we operate on block of 1 element in dim1,
    // start_a and end_a are 1.
    // start_b and end_b are offsets inside 1-element block, so it needs to start at 0 and end at 0 also.   
    dim0_mapping.dim = 0;
    dim0_mapping.start_a = dim0_mapping.end_a = 1;
    dim0_mapping.start_b = 0; dim0_mapping.end_b = 0;
   
    // Because start_a and end_a are multipliers and because we operate on block of 64 elements in dim0,
    // start_a and end_a are elementsInVec.
    // start_b and end_b are offsets inside a 64-element block, so it needs to start at 0 and go
    // all the way upto elementsInVec - 1.
    dim1_mapping.dim = 0;
    dim1_mapping.start_a = dim1_mapping.end_a = elementsInVec;
    dim1_mapping.start_b = 0; dim1_mapping.end_b = elementsInVec - 1;

    out_defs->inputTensorAccessPattern[0].dim[0] = dim0_mapping;
    out_defs->inputTensorAccessPattern[0].dim[1] = dim1_mapping;
    out_defs->inputTensorAccessPattern[1].dim[0] = dim0_mapping;
    out_defs->inputTensorAccessPattern[1].dim[1] = dim1_mapping;
    out_defs->inputTensorAccessPattern[2].dim[0] = dim0_mapping;
    out_defs->inputTensorAccessPattern[2].dim[1] = dim1_mapping;
    out_defs->outputTensorAccessPattern[0].dim[0] = dim0_mapping;
    out_defs->outputTensorAccessPattern[0].dim[1] = dim1_mapping;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
	out_defs->kernel.paramsNr = sizeof(*param_def)/ sizeof(float);
    memcpy(&( out_defs->kernel.scalarParams[0]), param_def, sizeof(*param_def));

	/*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___dissolve_blend_f32_gaudi2_o_end - 
                        &_binary___dissolve_blend_f32_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernelElf,
                &_binary___dissolve_blend_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}

