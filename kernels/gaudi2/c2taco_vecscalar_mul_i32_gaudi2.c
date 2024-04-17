// vector<int> muleq_sca(vector<int> a, int b, int n) {
//     vector<int> out;
//     for (int i = 0; i < n; ++i) {
//         out.push_back(a[i] * b);
//     }
//     return out;
// }

#pragma tpc_printf(enable)

void main(tensor a, int b, tensor output) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 int64 bv = b; // scalar to vector broadcasting
 
 // TODO: set hidden_dim to max input size.
 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping: we operate on blocks of 64 elements.
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    int64 a1 = v_i32_ld_tnsr_b(inputCoord, a);
    int128 c = v_i32_mul_b(a1, bv);
    v_i32_st_tnsr(outputCoord, output, c.v1); // ignorning the overflow by using .v1
  }
}