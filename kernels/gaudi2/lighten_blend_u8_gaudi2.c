// Lighten blend for 8-bit images.
// void lightenBlend8 (Buffer<uint8_t,2> base, Buffer<uint8_t,2> active, Buffer<uint8_t,2> out)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			if (base(col,row) < active(col,row))
// 				out(col,row) = active(col,row);
// 			else
// 				out(col,row) = base(col,row);
// 		}
// 	}
// }

void main(tensor base, tensor active, tensor out) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 256 char elements at a time.
 // Our index space operates on the basis of vec_len of 256.
 unsigned vec_len = 256;

 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    #pragma loop_unroll(4)
    for (int j = index_space_start[1]; j < index_space_end[1]; j++) {
      // index space mapping
      // coordinate 0 is for dim0.
      inputCoord[0] = outputCoord[0] = (i * vec_len);
      // coordinate 1 is for dim1.
      inputCoord[1] = outputCoord[1] = j;

      uchar256 b = v_u8_ld_tnsr_b(inputCoord, base);
      uchar256 a = v_u8_ld_tnsr_b(inputCoord, active);

      // We operate on a block of 64 elements at a time.
      // https://docs.habana.ai/en/latest/TPC/TPC_Intrinsics_Guide/Select.html#
      // float64 v_f32_sel_grt_f32_vb(float64 a, float64 b, float64 c, float64 d, 
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
      uchar256 income = 0;
      bool predicate = 1;
      uchar256 result = v_u8_sel_grt_u8_b(b, a, b, a, 0 /*switches*/, income, predicate);
      v_u8_st_tnsr(outputCoord, out, result);
    }
  }
}