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

    // We operate on a block of 256 elements at a time.
    // There is no TPCC API to handle overflow.
    uint256 ui1 = v_u8_mul_b(opacity_vec, a);
    // Shift by 8 bits produces incorrect result.
    char256 shift_bits = (char) 8;
    //uchar256 uc1 = v_u16_shr_b(ui1, shift_bits); // divide by 256
    uint256 ui2 = ui1 >> 8;
    uchar256 uc2 = convert_uint256_to_uchar256(ui2, 1 /* options */);
    
    // There is no TPCC API to handle overflow.
    uint256 ui3 = v_u8_mul_b(two_fifty_five_minus_opacity_vec, b);
    //uchar256 uc3 = v_u16_shr_b(ui3, shift_bits); // divide by 256
    uint256 ui4 = ui3 >> 8;
    uchar256 uc4 = convert_uint256_to_uchar256(ui4, 1 /* options */);

    // We can possibly add first and then divide by 255 to reduce number of divisions.
    // But that can possibly introduce rouding errors.    
    uchar256 c = v_u8_add_b(uc2, uc4);

    v_u8_st_tnsr(outputCoord, out, c);
  }
}