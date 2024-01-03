// inline uint8_t Mul8x8Div255 (uint8_t a, uint8_t b)
// {
//	return (a * b) / 255;
// }

// Normal blend (alpha compositing)
// + Implemented for 8-bit layers
// void normalBlend8 (Buffer<uint8_t,1> base, Buffer<uint8_t,1> active, Buffer<uint8_t,1> out, uint8_t opacity)
// {
//	for (int pixel=0; pixel<out.width(); pixel++) {
//		out(pixel) = Mul8x8Div255(opacity, active(pixel)) + Mul8x8Div255(255 - opacity, base(pixel));
//	}
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
  f2.v1 = v_f32_mul_b(f1.v1, reciprocal_255);
  f2.v2 = v_f32_mul_b(f1.v2, reciprocal_255);
  f2.v3 = v_f32_mul_b(f1.v3, reciprocal_255);
  f2.v4 = v_f32_mul_b(f1.v4, reciprocal_255);

  uchar256 uc2 = convert_float256_to_uchar256(f2, SW_RD);
  return uc2;
}

void main(tensor base, tensor active, tensor out, unsigned char opacity) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 256 uchar elements at a time.
 // Our index space operates on the basis of vec_len of 256.
 unsigned vec_len = 256;

 #pragma loop_unroll(8)
 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    uchar256 b = v_u8_ld_tnsr_b(inputCoord, base);
    uchar256 a = v_u8_ld_tnsr_b(inputCoord, active);

    // scalar to vector broadcasting
    uchar256 opacity_vec = opacity;
    uchar256 two_fifty_five_minus_opacity_vec = ((unsigned char) 255) - opacity;

    uchar256 uc2 = Mul8x8Div255(opacity_vec, a);
    uchar256 uc4 = Mul8x8Div255(two_fifty_five_minus_opacity_vec, b);
    uchar256 c = v_u8_add_b(uc2, uc4);

    v_u8_st_tnsr(outputCoord, out, c);
  }
}