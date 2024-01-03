// inline uint8_t Mul8x8Div255 (uint8_t a, uint8_t b)
// {
//	return (a * b) / 255;
// }
// inline uint8_t Screen8x8 (uint8_t a, uint8_t b)
// {
// 	return a + b - Mul8x8Div255(a, b);
// }
// Overlay blend for 8-bit images.
// void overlayBlend8 (Buffer<uint8_t,2> base, Buffer<uint8_t,2> active, Buffer<uint8_t,2> out)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			int delta;
// 			uint8_t a = base(col,row);
// 			uint8_t b = base(col,row);

// 			if (b >= 128)
// 				out(col,row) = Screen8x8(2 * a, b) - 255;
// 			else
// 				out(col,row) = Mul8x8Div255(2 * a, b);
// 		}
// 	}
// }

#pragma tpc_printf (enable)

uchar256 Mul8x8Div255(uint256 a, uchar256 b) {
  // Implements (a * b) / 255

  uint256 ub = convert_uchar256_to_uint256(b, 0);

  // perform a * b
  uint128 x1 = v_u32_mul_b(a.v1, ub.v1);
  uint128 x2 = v_u32_mul_b(a.v2, ub.v2);
  uint128 x3 = v_u32_mul_b(a.v3, ub.v3);
  uint128 x4 = v_u32_mul_b(a.v4, ub.v4);
  // Ignore .v2 components of x* as they will be zero
  // because multiplication would fit in uint32_t. We
  // don't need uint64_t.
  uint256 ui;
  ui.v1 = x1.v1; ui.v2 = x2.v1; ui.v3 = x3.v1; ui.v4 = x4.v1;

  // Convert to float to use float division
  float256 f1 = convert_uint256_to_float256(ui, 1);

  // perform division by multiplying with the reciprocal of the divisor.
  float64 divisor = 255;
  float64 reciprocal_255 = v_reciprocal_fast_f32(divisor);

  float256 f2;
  f2.v1 = v_f32_mul_b(f1.v1, reciprocal_255);
  f2.v2 = v_f32_mul_b(f1.v2, reciprocal_255);
  f2.v3 = v_f32_mul_b(f1.v3, reciprocal_255);
  f2.v4 = v_f32_mul_b(f1.v4, reciprocal_255);

  uchar256 uc2 = convert_float256_to_uchar256(f2, SW_RD);
  return uc2;
}

uchar256 Screen8x8(uint256 a, uchar256 b) {
  // Implements a + b - Mul8x8Div255(a, b)

  //uint256 ub = convert_uchar256_to_uint256(b, 0);
  uchar256 c = Mul8x8Div255(a, b);
  uchar256 d = v_u8_sub_b(b, c, SW_SAT);

  uint256 ud = convert_uchar256_to_uint256(d, 0);
  uint256 e;
  e.v1 = v_u32_add_b(a.v1, ud.v1, SW_SAT);
  e.v2 = v_u32_add_b(a.v2, ud.v2, SW_SAT);
  e.v3 = v_u32_add_b(a.v3, ud.v3, SW_SAT);
  e.v4 = v_u32_add_b(a.v4, ud.v4, SW_SAT);

  uchar256 f = convert_uint256_to_uchar256(e, SW_RD);
  return f;
}

uchar256 IfBlock(uchar256 base, uint256 two_times_a) {
  // Screen8x8(2 * a, b) - 255;

  uchar256 two_five_five = 255;
  uchar256 x = Screen8x8(two_times_a, base);  // Screen8x8(2 * a, b)
  uchar256 y = v_u8_sub_b(x, two_five_five);  // Screen8x8(2 * a, b) - 255
  return y;
}

uchar256 ElseBlock(uchar256 base, uint256 two_times_a) {
  // Mul8x8Div255(2 * a, b);

  uchar256 x = Mul8x8Div255(two_times_a, base);
  return x;
}

void main(tensor base, tensor active, tensor out) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();

 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 256 char elements at a time.
 // Our index space operates on the basis of vec_len of 256.
 unsigned vec_len = 256;
 uchar256 one_two_eight = 128;

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
      // float64 v_u8_sel_ge_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d,
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

      uchar256 two = 2;
      uint256 two_times_a = v_u8_mul_b(a, two);  // 2 * a

      uchar256 if_expr = IfBlock(b, two_times_a);
      uchar256 else_expr = ElseBlock(b, two_times_a);
      uchar256 result = v_u8_sel_geq_u8_b(b, one_two_eight, if_expr, else_expr);
      v_u8_st_tnsr(outputCoord, out, result);
    }
  }
}