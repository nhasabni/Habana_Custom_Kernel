// int softmax_part1(vector<int> input, int max_pos) {
//     int max_val = input[0];
//     for (int i = 1; i < max_pos; i++)
//         if (input[i] > max_val)
//             max_val = input[i];
//     return max_val;
// }
// def softmax_part1_ps(input max_pos softmax_part1_rv)
// softmax_part1_rv == reduce_max(list_take(input, max_pos))

#pragma tpc_printf(enable)

void main(tensor input, tensor output) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 int max_val = 0;

 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    int64 a = v_i32_ld_tnsr_b(inputCoord, input);
    int64 max = v_i32_reduce_max(a);
    if (max[0] > max_val)
      max_val = max[0];
  }

  v_i32_st_tnsr(outputCoord, out, max_val);
}