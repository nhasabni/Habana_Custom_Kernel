// Color dodge for 8-bit images.
// void colorDodge8 (Buffer<uint8_t,2> base, Buffer<uint8_t,2> active, Buffer<uint8_t,2> out)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			if (active(col, row) == 255)
// 				out(col,row) = 255;
// 			else
// 				out(col,row) = base(col,row) / (255 - active(col,row));
// 		}
// 	}
// }

#pragma tpc_printf (enable)

uchar256 U8Div(uchar256 a, uchar256 b) {
  // Implements a / b

  float256 fa = convert_uchar256_to_float256(a, 0);
  float256 fb = convert_uchar256_to_float256(b, 0);

  // perform division by multiplying with the reciprocal of the divisor.
  float64 reciprocal_fb1 = v_reciprocal_fast_f32(fb.v1);
  float64 reciprocal_fb2 = v_reciprocal_fast_f32(fb.v2);
  float64 reciprocal_fb3 = v_reciprocal_fast_f32(fb.v3);
  float64 reciprocal_fb4 = v_reciprocal_fast_f32(fb.v4);

  float256 f_result;
  f_result.v1 = v_f32_mul_b(fa.v1, reciprocal_fb1);
  f_result.v2 = v_f32_mul_b(fa.v2, reciprocal_fb2);
  f_result.v3 = v_f32_mul_b(fa.v3, reciprocal_fb3);
  f_result.v4 = v_f32_mul_b(fa.v4, reciprocal_fb4);

  uchar256 uc_result = convert_float256_to_uchar256(f_result, SW_RD);
  return uc_result;
}

uchar256 ElseBlock(uchar256 base, uchar256 active) {
  uchar256 two_five_five = (unsigned short) 255;  // scalar to vector broadcasting
  uchar256 x = v_u8_sub_b(two_five_five, active);  // 255 - active
  uchar256 y = U8Div(base, x);  // base / (255 - active)
  return y;
}

void main(tensor base, tensor active, tensor out) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 256 char elements at a time.
 // Our index space operates on the basis of vec_len of 256.
 unsigned vec_len = 256;
 uchar256 two_five_five = 255;

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
      // float64 v_u8_sel_eq_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d,
      //            int switches, uchar256 income, bool256 predicate, bool polarity=0)
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
      uchar256 else_expr = ElseBlock(b, a);
      uchar256 result = v_u8_sel_eq_u8_b(a, two_five_five, two_five_five, else_expr);
      v_u8_st_tnsr(outputCoord, out, result);
    }
  }
}