// vector<vector<int>> matscal(vector<vector<int>> mat, int val, int m, int n) {
//     vector<vector<int>> out;
//     for (int i = 0; i < m; ++i) {
//         vector<int> row_vec;
//         for (int j = 0; j < n; ++j) {
//             row_vec.push_back(mat[i][j] * val);
//         }
//         out.push_back(row_vec);
//     }
//     return out;
// }

#pragma tpc_printf (enable)

void main(tensor a, int scalar, tensor out) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 int64 sv = scalar; // scalar to vector broadcast

 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    #pragma loop_unroll(4)
    for (int j = index_space_start[1]; j < index_space_end[1]; j++) {
      // index space mapping
      // coordinate 0 is for dim0.
      inputCoord[0] = outputCoord[0] = (i * vec_len);
      // coordinate 1 is for dim1.
      inputCoord[1] = outputCoord[1] = j;

      int64 a1 = v_i32_ld_tnsr_b(inputCoord, a);
      int128 b1 = v_i32_mul_b(a1, sv);

      v_i32_st_tnsr(outputCoord, out, b1.v1);
    }
  }
}