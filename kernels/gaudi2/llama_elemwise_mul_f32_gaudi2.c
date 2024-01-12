// vector<float> elemwise_mul(vector<float> input1, vector<float> input2, int hidden_dim) {
//     vector<float> output;
//     for (int i = 0; i < hidden_dim; i++) {
//         output.push_back(input1[i] * input2[i]);
//     }
//     return output;
// }
// def elemwise_mul_ps(input1 input2 hidden_dim elemwise_mul_rv)
// elemwise_mul_rv == vec_elemwise_mul(list_take(input2, hidden_dim), list_take(input1, hidden_dim))

#pragma tpc_printf(enable)

void main(tensor input1, tensor input2, tensor output) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 
 // TODO: set hidden_dim to max input size.
 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping: we operate on blocks of 64 elements.
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    float64 a = v_f32_ld_tnsr_b(inputCoord, input1);
    float64 b = v_f32_ld_tnsr_b(inputCoord, input2);
    float64 c = v_f32_mul_b(a, b);
    v_f32_st_tnsr(outputCoord, output, c); 
  }
}
