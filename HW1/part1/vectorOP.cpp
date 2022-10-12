#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  // 宣告中間會用到的變數，概念類似register
  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;

  // 宣告中間會用到的vector常數
  __pp_vec_float f_zero = _pp_vset_float(0.f);
  __pp_vec_float f_one = _pp_vset_float(1.f);
  __pp_vec_float f_clamp_value = _pp_vset_float(9.999999f);
  __pp_vec_int i_zero = _pp_vset_int(0);
  __pp_vec_int i_one = _pp_vset_int(1);

  // 宣告mask與初始化
  __pp_mask maskAll, maskThisTime, maskYIsZero, maskYIsNotZero, maskYIsGtZero, maskResultGtClamp;
  maskAll = _pp_init_ones();
  maskThisTime = maskAll;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // mask初始化
    maskYIsZero = _pp_init_ones(0); // all zeros
    maskResultGtClamp = _pp_init_ones(0); // all zeros
    maskYIsGtZero = _pp_init_ones(0); // all zeros

    if (i + VECTOR_WIDTH >= N && N % VECTOR_WIDTH != 0) // 如果N非vector寬度的倍數，且這次是最後一次
      maskThisTime = _pp_init_ones(N % VECTOR_WIDTH); // 只做剩下的部分，不能做超過

    _pp_vload_float(x, values + i, maskThisTime); // x = values[i]
    _pp_vload_int(y, exponents + i, maskThisTime); // y = exponents[i]
    // 只有vector(用陣列傳進來的)可以用vload，所以後面賦值會用vadd(??, 0, x, ??)之類的方法來表示vector賦值

    _pp_veq_int(maskYIsZero, y, i_zero, maskThisTime); // if (y == 0) {
    _pp_vadd_float(result, f_zero, f_one, maskYIsZero); // output[i] = 1.f }

    maskYIsNotZero = _pp_mask_not(maskYIsZero); // else {
    maskYIsNotZero = _pp_mask_and(maskYIsNotZero, maskThisTime); // 續上行，因為取not沒辦法設mask，所以要另外做and運算處理最後一次不要多做的部分
    _pp_vadd_float(result, f_zero, x, maskYIsNotZero); // result = x  (result和output[i]都用result沒差，因為等等都會把result存到output)
    _pp_vsub_int(y, y, i_one, maskYIsNotZero); // count = y - 1 (count用y表示就好)
    _pp_vgt_int(maskYIsGtZero, y, i_zero, maskYIsNotZero); // 做下行的準備
    while (_pp_cntbits(maskYIsGtZero) > 0) // while (count > 0) {
    {
      _pp_vmult_float(result, result, x, maskYIsGtZero); // result *= x
      _pp_vsub_int(y, y, i_one, maskYIsGtZero); // count--
      _pp_vgt_int(maskYIsGtZero, y, i_zero, maskYIsNotZero); // 準備下一次while }
    }

    _pp_vgt_float(maskResultGtClamp, result, f_clamp_value, maskThisTime); // if (result > 9.999999f) {    // 如果mask用maskYIsNotZer，且maskResultGtClamp沒有初始化的話，可能會錯，所以要記得初始化
    _pp_vadd_float(result, f_zero, f_clamp_value, maskResultGtClamp); // result = 9.999999f }

    _pp_vstore_float(output + i, result, maskThisTime); // output[i] = result
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  float sum = 0.0;

  // 宣告中間會用到的變數，概念類似register
  __pp_vec_float x, y;

  // 宣告mask與初始化
  __pp_mask maskAll;
  maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x, values + i, maskAll); // x = values[i]

    int count = VECTOR_WIDTH;
    while ((count /= 2) >= 1) // 做log2(VECTOR_WIDTH)次就好 (You can assume VECTOR_WIDTH is a power of 2)
    {
      _pp_hadd_float(y, x); // 相鄰的相加
      _pp_interleave_float(x, y); // 把偶數位置移到前半，奇數位置移到後半。因為_pp_interleave_float(x,x)會因src未備份而出問題，所以才用x和y兩個
    }
    sum += x.value[0]; // x的第一個值就是整個vector的sum
  }

  return sum;
}