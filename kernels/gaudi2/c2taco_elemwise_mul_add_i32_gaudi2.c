// vector<int> n_real_updates(int N, vector<int> A, vector<int> B, vector<int> C) {
//     vector<int> D;
//     for (int i = 0; i < N; i++) {
//         int curr = C[i] + A[i] * B[i];
//         D.push_back(curr);
//     }
//     return D;
// }

#pragma tpc_printf(enable)

void main(tensor a, tensor b, tensor c, tensor output) {
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

    int64 a1 = v_i32_ld_tnsr_b(inputCoord, a);
    int64 b1 = v_i32_ld_tnsr_b(inputCoord, b);
    int64 c1 = v_i32_ld_tnsr_b(inputCoord, c);
    int128 d1 = v_i32_mul_b(b1, c1);
    int64 e1 = v_i32_add_b(d1.v1, a1);
    v_i32_st_tnsr(outputCoord, output, e1);
  }
}
