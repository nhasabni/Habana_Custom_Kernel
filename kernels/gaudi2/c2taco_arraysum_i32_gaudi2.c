// int arraysum(vector<int> a, int n) {
//   int sum = 0;
//   for (int i = 0; i < n; ++i) {
//     sum += a[i];
//   }
//   return sum;
// }
#pragma tpc_printf(enable)
void main(tensor a, tensor output) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 int64 sum = 0;
 
 // We use n as vector length.
 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping: we operate on blocks of 64 elements.
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    int64 a1 = v_i32_ld_tnsr_b(inputCoord, a);
    int64 rsum = v_i32_reduce_add(a1);
    sum = v_i32_add_b(rsum, sum);
  }

  outputCoord[0] = index_space_start[0] * vec_len;
  v_i32_st_tnsr(outputCoord, output, sum);
}