// Dissolve blending
// + Layers are represented as 2D buffers
// void dissolveBlend8 (Buffer<float,2> base, Buffer<float,2> active, Buffer<float,2> out, float opacity)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			float rand_val = ((rand() % 100) + 1) / 100.0f;
// 			if (opacity - rand_val >= 0.0f)
// 				out(col,row) = active(col,row);
// 			else
// 				out(col,row) = base(col,row);
// 		}
// 	}
// }

// TODO

#pragma tpc_printf (enable)

void main(tensor base, tensor active, tensor rand, tensor out, float opacity) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;
 float64 zero_vec = 0.0; // scalar to vector broadcasting

 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    #pragma loop_unroll(4)
    for (int j = index_space_start[1]; j < index_space_end[1]; j++) {
      // index space mapping
      // coordinate 0 is for dim0.
      inputCoord[0] = outputCoord[0] = (i * vec_len);
      // coordinate 1 is for dim1.
      inputCoord[1] = outputCoord[1] = j;

      float64 b = v_f32_ld_tnsr_b(inputCoord, base);
      float64 a = v_f32_ld_tnsr_b(inputCoord, active);
      float64 r = v_f32_ld_tnsr_b(inputCoord, rand);

      float64 opacity_vec = opacity;  // scalar to vector broadcasting
      float64 diff_vec = v_f32_sub_b(opacity_vec, r);

      // We operate on a block of 64 elements at a time.
      // https://docs.habana.ai/en/latest/TPC/TPC_Intrinsics_Guide/Select.html#
      // float64 v_f32_sel_geq_f32_vb(float64 a, float64 b, float64 c, float64 d, 
      //         int switches, float64 income, bool64 predicate, bool polarity=0)
      // a - Source #1 to compare (SRC1).
      // b - Source #2 to compare (SRC2).
      // c - Source #1 to select (SRC3).
      // d - Source #2 to select (SRC4).
      // switches - Switches of the instruction.
      // income - This value is returned if the predicate is false.
      // predicate - Predicate value for the instruction.
      // polarity - True if polarity of the predicate is inverted.
      //
      // return - One of the sources - c or d with respect to the comparison 
      // of the two other sources - a and b.
      float64 result = v_f32_sel_geq_f32_b(diff_vec, zero_vec, a, b);
      v_f32_st_tnsr(outputCoord, out, result);
    }
  }
}