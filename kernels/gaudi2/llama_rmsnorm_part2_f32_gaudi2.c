// int uninterp_sqrt(int x) { return x; }

// vector<float> rmsnorm_part2(vector<float> input, vector<float> weight, float ss) {
//     vector<float> output;
//     int size = input.size();
//     float inv_ss = 1 / sqrt(ss / size + 1);
//     for (int i = 0; i < input.size(); i++)
//         output.push_back(inv_ss * input[i] * weight[i]);
//     return output;
// }
// def rmsnorm_part2_ps(input weight ss rmsnorm_part2_rv)
// rmsnorm_part2_rv == vec_scalar_mul((1 / test_sqrt(((ss / list_length(input)) + 1))), vec_elemwise_mul(input, weight))

#pragma tpc_printf(enable)

void main(tensor input, tensor weight, tensor output, float ss) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;

 int size = get_dim_size(input, 0);
 int size_plus_1 = s_i32_add(size, 1);
 // scalar to vector broadcast
 float64 ss_div_size_plus_1 = (float) ss / (float) size_plus_1;
 
 // use reciprocal sqrt
 float64 sqrt = v_sqrt_fast_f32(ss_div_size_plus_1);
 float64 one = 1.0; // scalar broadcast
 float64 inv_ss = v_div_fast_f32(one, sqrt);
 
 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    float64 a = v_f32_ld_tnsr_b(inputCoord, input);
    float64 w = v_f32_ld_tnsr_b(inputCoord, weight);

    float64 b = v_f32_mul_b(inv_ss, a);
    float64 o = v_f32_mul_b(b, w);

    // output is a vector.
    v_f32_st_tnsr(outputCoord, output, o);
  }
}
