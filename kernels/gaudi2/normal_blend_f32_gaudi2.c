//
// void normalBlendf (Buffer<float,1> base, Buffer<float,1> active, Buffer<float,1> out, float opacity)
// {
// 	for (int pixel=0; pixel<out.width(); pixel++) {
// 		out(pixel) = opacity * active(pixel) + (1.0f - opacity) * base(pixel);
// 	}
// }

void main(tensor base, tensor active, tensor out, float opacity) {
 int5 index_space_start = get_index_space_offset();
 int5 index_space_end = index_space_start + get_index_space_size();
 
 int5 inputCoord = { 0 };
 int5 outputCoord = { 0 };

 // We operate on a block of 64 elements at a time.
 // Our index space operates on the basis of vec_len of 64.
 unsigned vec_len = 64;

 for(int i = index_space_start[0]; i < index_space_end[0]; i++) {
    // index space mapping
    inputCoord[0] = outputCoord[0] = (i * vec_len);

    float64 b = v_f32_ld_tnsr_b(inputCoord, base);
    float64 a = v_f32_ld_tnsr_b(inputCoord, active);

    // scalar to vector broadcasting
    float64 opacity_vec = opacity;
    float64 one_minus_opacity_vec = 1.0f - opacity;

    // We operate on a block of 64 elements at a time.
    float64 m1 = v_f32_mul_b(opacity_vec, a);
    float64 m2 = v_f32_mul_b(one_minus_opacity_vec, b);
    float64 c = v_f32_add_b(m1, m2);

    v_f32_st_tnsr(outputCoord, out, c);
  }
}