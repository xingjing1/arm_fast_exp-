#include "arm_neon.h"
#define F16 __fp16

inline float16x8_t exp_compute_neon(float16x8_t x){
    static const float16x8_t const_Y = vdupq_n_f16(1477.3197217792);
    static const float16x8_t const_B = vdupq_n_f16(15301.3197217792);
    x = vmaxq_f16(-10, x);
    float16x8_t in, in3;
    int16x8_t in2;
    in = vfmaq_f16(const_B, const_Y, x);
    in2 = vcvtq_s16_f16(in);
    in3 = vreinterpretq_f16_s16(in2);
    return in3;
}


int fast_exp(F16* input, const F16* result, int length){
    float16x8_t inv, resv;
    if(length % 8){
        return 0;
    }
    for(int i = 0; i < length; i+=8){
        inv = vld1q_f16(&input[i]);
        resv = exp_compute_neon(inv);
        vst1q_f16(&result[i], resv);
    }
    return 0;
}

