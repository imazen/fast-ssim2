// Copyright (c) 2017-2026, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Rust adaptation of powf.c, powf_log2_data.c and exp2f_data.c from
// https://github.com/ARM-software/optimized-routines/tree/f5205d212ac67727689176b7fcfaef7864d898a5/math
// Uses the unscaled, non-contracting polynomial path. Only positive finite
// bases with finite, nonzero exponents enter this kernel; other cases retain
// the platform's IEEE special-value behavior. See ../../LICENSE.arm.

const LOG: [[u64; 2]; 16] = [
    [0x3ff661ec79f8f3be, 0xbfdefec65b963019],
    [0x3ff571ed4aaf883d, 0xbfdb0b6832d4fca4],
    [0x3ff49539f0f010b0, 0xbfd7418b0a1fb77b],
    [0x3ff3c995b0b80385, 0xbfd39de91a6dcf7b],
    [0x3ff30d190c8864a5, 0xbfd01d9bf3f2b631],
    [0x3ff25e227b0b8ea0, 0xbfc97c1d1b3b7af0],
    [0x3ff1bb4a4a1a343f, 0xbfc2f9e393af3c9f],
    [0x3ff12358f08ae5ba, 0xbfb960cbbf788d5c],
    [0x3ff0953f419900a7, 0xbfaa6f9db6475fce],
    [0x3ff0000000000000, 0x0000000000000000],
    [0x3fee608cfd9a47ac, 0x3fb338ca9f24f53d],
    [0x3feca4b31f026aa0, 0x3fc476a9543891ba],
    [0x3feb2036576afce6, 0x3fce840b4ac4e4d2],
    [0x3fe9c2d163a1aa2d, 0x3fd40645f0c6651c],
    [0x3fe886e6037841ed, 0x3fd88e9c2c1b9ff8],
    [0x3fe767dcf5534862, 0x3fdce0a44eb17bcc],
];

const A: [f64; 5] = [
    f64::from_bits(0x3fd27616c9496e0b),
    f64::from_bits(0xbfd71969a075c67a),
    f64::from_bits(0x3fdec70a6ca7badd),
    f64::from_bits(0xbfe7154748bef6c8),
    f64::from_bits(0x3ff71547652ab82b),
];

const EXP: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];

pub(super) fn powf(x: f32, y: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 || !y.is_finite() || y == 0.0 {
        return x.powf(y);
    }
    let mut ix = x.to_bits();
    if ix < 0x00800000 {
        ix = (x * 8388608.0).to_bits().wrapping_sub(23 << 23);
    }
    let tmp = ix.wrapping_sub(0x3f330000);
    let i = ((tmp >> 19) & 15) as usize;
    let top = tmp & 0xff800000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23;
    let invc = f64::from_bits(LOG[i][0]);
    let logc = f64::from_bits(LOG[i][1]);
    let z = f64::from(f32::from_bits(iz));
    let r = z * invc - 1.0;
    let y0 = logc + f64::from(k);
    let r2 = r * r;
    let p = A[2] * r + A[3];
    let r4 = r2 * r2;
    let q = A[4] * r + y0;
    let q = p * r2 + q;
    let logx = (A[0] * r + A[1]) * r4 + q;
    let ylogx = f64::from(y) * logx;
    if ylogx <= -150.0 {
        return 0.0;
    }
    if ylogx > f64::from_bits(0x405fffffffd1d571) {
        return f32::INFINITY;
    }
    // The large addition rounds to a multiple of 1/32. Its bit pattern also
    // supplies the exponent and table index without a float-to-int conversion.
    const SHIFT: f64 = 211106232532992.0; // 0x1.8p52 / 32
    let kd = ylogx + SHIFT;
    let ki = kd.to_bits();
    let r = ylogx - (kd - SHIFT);
    let scale = f64::from_bits(EXP[(ki & 31) as usize].wrapping_add(ki << 47));
    let z = f64::from_bits(0x3fac6af84b912394) * r + f64::from_bits(0x3fcebfce50fac4f3);
    let r2 = r * r;
    let y = f64::from_bits(0x3fe62e42ff0c52d6) * r + 1.0;
    ((z * r2 + y) * scale) as f32
}
