// float softmax_part3(vector<float> output, int max_pos) {
//     float sum = 0;
//     for (int i = 0; i < max_pos; i++) {
//         sum += output[i];
//     }
//     return sum;
// }

#pragma tpc_printf(enable)

void main(tensor input, tensor output) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 float64 overall_sum = 0.0;

 // This loop is different than other parallel kernels because this is a reduction
 // kernel. So we need to access whole of input tensor and not a chunk of it.
 // We ensure this in the glue code by asking it not to split the input space.
 // And then in this loop, we operate on whole of the input.
 //
 // In a sense, this could be unoptimal implementation as we are not parallelizing
 // the input space and as a result only using one TPC core. It seems that we could
 // potentially perform a tree reduction here by leveraging multiple TPC cores. But
 // for the time being, I will not get into this.
 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    inputCoord[0] = (i * vec_len);

    float64 a = v_f32_ld_tnsr_b(inputCoord, input);
    overall_sum = v_f32_add_b(overall_sum, a);
  }

  // Because this is a reduction kernel, we only care about 1st element of the
  // output tensor. Since we cannot access only 1 element of overall_sum, we
  // write 64 elements in the output. Rest of the output elements are left as 0.
  outputCoord[0] = index_space_start[0] * vec_len;
  overall_sum = v_f32_reduce_add(overall_sum);  // sum of 64 floats
  v_f32_st_tnsr(outputCoord, output, overall_sum);
}
