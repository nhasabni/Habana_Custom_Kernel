// inline uint8_t Mul8x8Div255 (uint8_t a, uint8_t b)
// {
//	return (a * b) / 255;
// }

// void multiplyBlend8 (Buffer<uint8_t,2> base, Buffer<uint8_t,2> active, Buffer<uint8_t,2> out)
// {
// 	for (int row=0; row<out.height(); row++) {
// 		for (int col=0; col<out.width(); col++) {
// 			out(col,row) = Mul8x8Div255(base(col,row), active(col,row));
// 		}
// 	}
// }

#pragma tpc_printf(enable)

uchar256 Mul8x8Div255(uchar256 a, uchar256 b) {
  // Implements (a * b) / 255

  // perform a * b
  uint256 ui1 = v_u8_mul_b(a, b);

  // Convert to float to use float division
  float256 f1 = convert_uint256_to_float256(ui1, 1);

  // perform division by multiplying with the reciprocal of the divisor.
  float64 divisor = 255;
  float64 reciprocal_255 = v_reciprocal_fast_f32(divisor);

  float256 f2;
  f2.v1 = f1.v1 * reciprocal_255; f2.v2 = f1.v2 * reciprocal_255;
  f2.v3 = f1.v3 * reciprocal_255; f2.v4 = f1.v4 * reciprocal_255;

  uchar256 uc2 = convert_float256_to_uchar256(f2, SW_RD);
  return uc2;
}

void main(tensor base, tensor active, tensor out) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 256 uchar elements at a time.
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

      uchar256 c = Mul8x8Div255(a, b);

      v_u8_st_tnsr(outputCoord, out, c);
    }
  }
}
