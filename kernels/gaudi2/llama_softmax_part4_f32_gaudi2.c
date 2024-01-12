// vector<float> softmax_part4(vector<float> unnormalized_output, int max_pos, float sum) {
//     vector<float> output;
//     for (int i = 0; i < max_pos; i++) {
//         output.push_back(unnormalized_output[i] / sum);
//     }
//     return output;
// }
// def softmax_part4_ps(unnormalized_output max_pos sum softmax_part4_rv)
// softmax_part4_rv == vec_scalar_div(sum, list_take(unnormalized_output, max_pos))

#pragma tpc_printf(enable)

void main(tensor input, tensor output, float sum) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 float64 sum_vec = sum;  // scalar broadcast

 // 1. max_pos is multiple of 64. Set it to maximum size of input.
 // 2. Index space mapping needs to operate from 0 to max_pos.
 // We don't care about the rest of the elements of output.
 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    float64 a = v_f32_ld_tnsr_b(inputCoord, input);
    float64 b = v_div_fast_f32(a, sum_vec);
    v_f32_st_tnsr(outputCoord, output, b);
  }
}
