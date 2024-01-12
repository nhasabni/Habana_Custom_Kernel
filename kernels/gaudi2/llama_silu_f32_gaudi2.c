// vector<float> silu(vector<float> input, int hidden_dim) {
//     vector<float> output;
//     for (int i = 0; i < hidden_dim; i++) {
//         float curr = 1 / (1 + sqrt(input[i])) * input[i];
//         output.push_back(curr);
//     }
//     return output;
// }
// def silu_ps(input hidden_dim silu_rv)
// silu_rv == vec_elemwise_mul(scalar_vec_div(1, vec_scalar_add(1, vec_map(list_take(input, hidden_dim), map_int_to_int))), list_take(input, hidden_dim))

#pragma tpc_printf(enable)

void main(tensor input, tensor output) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 float64 one_vec = 1.0;  // scalar to vector broadcast

 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    float64 a = v_f32_ld_tnsr_b(inputCoord, input);

    float64 sqrt_a = v_sqrt_fast_f32(a);
    float64 sqrt_a_plus_1 = v_f32_add_b(sqrt_a, one_vec);
    float64 a_div_sqrt_a_plus_1 = v_div_fast_f32(a, sqrt_a_plus_1);

    v_f32_st_tnsr(outputCoord, output, a_div_sqrt_a_plus_1);
  }
}
