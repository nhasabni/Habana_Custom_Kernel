#include <vector>
#include <cstring>
#include <iostream>
#include "llama_rmsnorm_part1_f32_gaudi2.hpp"

extern unsigned char _binary___llama_rmsnorm_part1_f32_gaudi2_o_start;
extern unsigned char _binary___llama_rmsnorm_part1_f32_gaudi2_o_end;

 gcapi::GlueCodeReturn_t LlamaRmsnormPart1F32Gaudi2::GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_llama_rmsnorm_part1_f32_gaudi2");
     return gcapi::GLUE_SUCCESS;
 }


gcapi::GlueCodeReturn_t LlamaRmsnormPart1F32Gaudi2::GetGcDefinitions(
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
        in_defs->inputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr !=1)
    {
        in_defs->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if (in_defs->inputTensors[0].dataType != gcapi::DATA_F32 ||
        in_defs->inputTensors[1].dataType != gcapi::DATA_F32 ||
        in_defs->outputTensors[0].dataType != gcapi::DATA_F32)
    {
        in_defs->inputTensors[0].dataType = gcapi::DATA_F32;
        in_defs->inputTensors[1].dataType = gcapi::DATA_F32;
        in_defs->outputTensors[0].dataType = gcapi::DATA_F32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. Output size is same as input size.
    **************************************************************************************/
    unsigned int inputSizes[1] = {0};
    inputSizes[0] = in_defs->inputTensors[0].geometry.sizes[0];

    // We operate on a block of 64 float elements at a time.
    out_defs->indexSpaceGeometry.dims = 1;

    // Because this is a reduction kernel, we do not partition the input space into
    // different index spaces. So the index spaces goes from [0-1]. This ensures that
    // the kernel is invoked only once. Setting for start_* and end_* below ensure
    // that that single invocation accesses whole of the input/output tensors.
    unsigned dim0Index = 1;
    out_defs->indexSpaceGeometry.sizes[0] = dim0Index;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
   //
   // Index space to tensor mapping looks like below: single invocation accessess whole of
   // the input tensor.
   //
   // 0                                    64                                 128
   // -------------------------------------------------------------------------------------
   // |                                        Index 0                            ....
   // -------------------------------------------------------------------------------------

    // Index space mapping is calculated using start_a * x + start_b to end_a * x + end_b.
    // x is the index space value. As we want single index space, we set end_b for input
    // and output to input_size - 1.
    out_defs->inputTensorAccessPattern[0].dim[0].dim      = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].start_a  = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].end_a    = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].end_b    = inputSizes[0] - 1;

    out_defs->inputTensorAccessPattern[1].dim[0].dim      = 0;
    out_defs->inputTensorAccessPattern[1].dim[0].start_a  = 0;
    out_defs->inputTensorAccessPattern[1].dim[0].end_a    = 0;
    out_defs->inputTensorAccessPattern[1].dim[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[1].dim[0].end_b    = inputSizes[0] - 1;

    out_defs->outputTensorAccessPattern[0].dim[0].dim      = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].start_a  = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].end_a    = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].end_b    = inputSizes[0] - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
	
	/*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___llama_rmsnorm_part1_f32_gaudi2_o_end - 
                        &_binary___llama_rmsnorm_part1_f32_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernelElf,
                &_binary___llama_rmsnorm_part1_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}

