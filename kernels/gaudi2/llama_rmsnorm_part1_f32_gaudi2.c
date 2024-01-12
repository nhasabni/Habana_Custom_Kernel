// float rmsnorm_part1(vector<float> input, vector<float> weight) {
//     float ss = 0;
//     for (int i = 0; i < input.size(); i++)
//         ss += input[i] * input[i];
//     return ss;
// }

#pragma tpc_printf(enable)

void main(tensor input, tensor weight, tensor output) {
 int input_size = get_dim_size(input, 0);

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 int elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 float64 ss = 0.0;

 // This loop is different than other parallel kernels because this is a reduction
 // kernel. So we need to access whole of input tensor and not a chunk of it.
 // We ensure this in the glue code by asking it not to split the input space.
 // And then in this loop, we operate on whole of the input.
 //
 // In a sense, this could be unoptimal implementation as we are not parallelizing
 // the input space and as a result only using one TPC core. It seems that we could
 // potentially perform a tree reduction here by leveraging multiple TPC cores. But
 // for the time being, I will not get into this.
 #pragma loop_unroll(8)
 for(int i = 0; i < input_size; i += vec_len) {
    inputCoord[0] = i;

    float64 a = v_f32_ld_tnsr_b(inputCoord, input);
    float64 w = v_f32_ld_tnsr_b(inputCoord, weight);
    float64 o = v_f32_mul_b(a, w);
    float64 sum_vec = v_f32_reduce_add(o);
    ss = v_f32_add_b(ss, sum_vec);
  }
  // Because this is a reduction kernel, we only care about 1st element of the
  // output tensor. Since we cannot access only 1 element of overall_max, we
  // write 64 elements in the output. Rest of the output elements are left as 0.
  v_f32_st_tnsr(outputCoord, output, ss);
}
