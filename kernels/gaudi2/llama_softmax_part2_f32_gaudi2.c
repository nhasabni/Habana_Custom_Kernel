
// vector<float> softmax_part2(vector<float> input, int max_pos, float max_val) {
//     vector<float> output;
//     for (int i = 0; i < max_pos; i++) {
//         float cur = exp(input[i] - max_val);
//         output.push_back(cur);
//     }
//     return output;
// }
// def softmax_part2_ps(input max_pos max_val softmax_part2_rv)
// softmax_part2_rv == vec_map(vec_scalar_sub(max_val, list_take(input, max_pos)), map_int_to_int)

#pragma tpc_printf(enable)

void main(tensor input, tensor output, float max_val) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 float64 max_val_vec = max_val;  // scalar broadcast
 
 // TODO:
 // 1. max_pos is multiple of 64. Set it to maximum size of input.
 // 2. Index space mapping needs to operate from 0 to max_pos.
 // We don't care about the rest of the elements of output.
 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    float64 a = v_f32_ld_tnsr_b(inputCoord, input);
    float64 b = v_f32_sub_b(a, max_val_vec);
    float64 c = v_exp_fast_f32(b);
    v_f32_st_tnsr(outputCoord, output, c);
  }
}
