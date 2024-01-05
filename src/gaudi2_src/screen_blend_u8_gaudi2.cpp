#include <vector>
#include <cstring>
#include <iostream>
#include "screen_blend_u8_gaudi2.hpp"


extern unsigned char _binary___screen_blend_u8_gaudi2_o_start;
extern unsigned char _binary___screen_blend_u8_gaudi2_o_end;

 gcapi::GlueCodeReturn_t ScreenBlendU8Gaudi2::GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_screen_blend_u8_gaudi2");
     return gcapi::GLUE_SUCCESS;
 }


gcapi::GlueCodeReturn_t ScreenBlendU8Gaudi2::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* in_defs,
            gcapi::HabanaKernelInstantiation_t* out_defs)
{
    gcapi::GlueCodeReturn_t retVal;

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 2)
    {
        in_defs->inputTensorNr  = 2;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr !=1)
    {
        in_defs->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    //validate matrix dimensions
    if (in_defs->inputTensors[0].geometry.sizes[0] != 
        in_defs->inputTensors[1].geometry.sizes[0])
    {
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    // validate input and output data type
    if (in_defs->inputTensors[0].dataType != gcapi::DATA_U8 ||
        in_defs->inputTensors[1].dataType != gcapi::DATA_U8 ||
        in_defs->outputTensors[0].dataType != gcapi::DATA_U8)
    {
        in_defs->inputTensors[0].dataType = gcapi::DATA_U8;
        in_defs->inputTensors[1].dataType = gcapi::DATA_U8;
        in_defs->outputTensors[0].dataType = gcapi::DATA_U8;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. Output size is same as input size.
    **************************************************************************************/
    unsigned int outputSizes[2] = {0, 0};
    memcpy(outputSizes, in_defs->inputTensors[0].geometry.sizes, sizeof(outputSizes));

    // We operate on a block of 256 uchar elements at a time.
    int elementsInVec = 256;
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
    out_defs->outputTensorAccessPattern[0].dim[0] = dim0_mapping;
    out_defs->outputTensorAccessPattern[0].dim[1] = dim1_mapping;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
	
	/*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___screen_blend_u8_gaudi2_o_end - 
                        &_binary___screen_blend_u8_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernelElf,
                &_binary___screen_blend_u8_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}

