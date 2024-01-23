#include <vector>
#include <cstring>
#include <iostream>
#include "llama_softmax_part3_f32_gaudi2.hpp"

extern unsigned char _binary___llama_softmax_part3_f32_gaudi2_o_start;
extern unsigned char _binary___llama_softmax_part3_f32_gaudi2_o_end;

 gcapi::GlueCodeReturn_t LlamaSoftmaxPart3F32Gaudi2::GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_llama_softmax_part3_f32_gaudi2");
     return gcapi::GLUE_SUCCESS;
 }


gcapi::GlueCodeReturn_t LlamaSoftmaxPart3F32Gaudi2::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* in_defs,
            gcapi::HabanaKernelInstantiation_t* out_defs)
{
    gcapi::GlueCodeReturn_t retVal;

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 1)
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
        in_defs->outputTensors[0].dataType != gcapi::DATA_F32)
    {
        in_defs->inputTensors[0].dataType = gcapi::DATA_F32;
        in_defs->outputTensors[0].dataType = gcapi::DATA_F32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. Output size is same as input size.
    **************************************************************************************/
    unsigned int outputSizes[1] = {0};
    memcpy(outputSizes, in_defs->inputTensors[0].geometry.sizes, sizeof(outputSizes));

    // We operate on a block of 64 float elements at a time.
    int elementsInVec = 64;
    out_defs->indexSpaceGeometry.dims = 1;

    // round up to elementsInVec and divide by elementsInVec.
    unsigned dim0Index = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
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

    // Index space mapping is calculated using f(i) = Ai + B
    // 'i' is the index space member and A/B constants to be defined.
    out_defs->inputTensorAccessPattern[0].dim[0].dim      = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].start_a  = elementsInVec;
    out_defs->inputTensorAccessPattern[0].dim[0].end_a    = elementsInVec;
    out_defs->inputTensorAccessPattern[0].dim[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].end_b    = elementsInVec - 1;

	// f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    out_defs->outputTensorAccessPattern[0].dim[0].dim      = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].start_a  = elementsInVec;
    out_defs->outputTensorAccessPattern[0].dim[0].end_a    = elementsInVec;
    out_defs->outputTensorAccessPattern[0].dim[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].end_b    = elementsInVec - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
	
    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___llama_softmax_part3_f32_gaudi2_o_end - 
                        &_binary___llama_softmax_part3_f32_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernelElf,
                &_binary___llama_softmax_part3_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}

