mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

/// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
/// Cosine Functions" by Charalampidis [2016].
pub struct RecursiveGaussian;

impl RecursiveGaussian {
    #[cfg(feature = "rayon")]
    pub fn horizontal_pass(&self, input: &[f32], output: &mut [f32], width: usize) {
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        use rayon::prelude::ParallelSliceMut;
        use rayon::slice::ParallelSlice;

        assert_eq!(input.len(), output.len());

        input
            .par_chunks_exact(width)
            .zip(output.par_chunks_exact_mut(width))
            .for_each(|(input, output)| self.horizontal_row(input, output, width));
    }

    #[cfg(not(feature = "rayon"))]
    pub fn horizontal_pass(&self, input: &[f32], output: &mut [f32], width: usize) {
        assert_eq!(input.len(), output.len());

        for (input, output) in input
            .chunks_exact(width)
            .zip(output.chunks_exact_mut(width))
        {
            self.horizontal_row(input, output, width);
        }
    }

    /// jpegli's `FastGaussian1D`, scalar form: four outputs per iteration from
    /// the closed forms for two, three and four recurrence steps.
    ///
    /// This must stay operation-for-operation identical to the vectorised body
    /// in `simd_gaussian::horizontal_pass_inner`, because `SimdImpl::Scalar`
    /// and `SimdImpl::Simd` are meant to be the same metric at different
    /// speeds — `simd_consistency` gates the two at 1e-4 end to end and the
    /// pair measured bit-identical when this was written. Before 0.9.1 both
    /// ran the plain single-step recurrence; that agreed with itself but not
    /// with the C++ reference, whose vector targets evaluate the unrolled form.
    fn horizontal_row(&self, input: &[f32], output: &mut [f32], width: usize) {
        const LANES: isize = 4;
        let big_n = consts::RADIUS as isize;
        let at = |i: isize| -> f32 {
            if i >= 0 && (i as usize) < width {
                input[i as usize]
            } else {
                0.0
            }
        };

        let (mut p1, mut p3, mut p5) = (0f32, 0f32, 0f32);
        let (mut q1, mut q3, mut q5) = (0f32, 0f32, 0f32);
        let step = |sum: f32, p: &mut (f32, f32, f32), q: &mut (f32, f32, f32)| -> f32 {
            let mut o1 = sum * consts::MUL_IN_1;
            let mut o3 = sum * consts::MUL_IN_3;
            let mut o5 = sum * consts::MUL_IN_5;
            // MUL_PREV2_k is exactly -1, so this product is exact and fusing
            // it changes nothing either way.
            o1 = consts::MUL_PREV2_1.mul_add(q.0, o1);
            o3 = consts::MUL_PREV2_3.mul_add(q.1, o3);
            o5 = consts::MUL_PREV2_5.mul_add(q.2, o5);
            *q = *p;
            o1 = consts::MUL_PREV_1.mul_add(p.0, o1);
            o3 = consts::MUL_PREV_3.mul_add(p.1, o3);
            o5 = consts::MUL_PREV_5.mul_add(p.2, o5);
            *p = (o1, o3, o5);
            o1 + o3 + o5
        };

        // Scalar until `RoundUpTo(N + 1, 4)`, exactly as the C++ does.
        let mut n = -big_n + 1;
        let first_aligned = ((big_n + 1) + LANES - 1) / LANES * LANES;
        while n < first_aligned.min(width as isize) {
            let sum = at(n - big_n - 1) + at(n + big_n - 1);
            let (mut pv, mut qv) = ((p1, p3, p5), (q1, q3, q5));
            let o = step(sum, &mut pv, &mut qv);
            (p1, p3, p5) = pv;
            (q1, q3, q5) = qv;
            if n >= 0 {
                output[n as usize] = o;
            }
            n += 1;
        }

        // Unrolled interior: `out[lane] = sum_{i<=lane} mul_in[lane-i] * s[i]
        // + mul_prev[lane] * prev + mul_prev2[lane] * prev2`, with the state
        // for the next iteration taken from lanes 2 and 3 (`Broadcast<N-2>`
        // and `Broadcast<N-1>` in the C++).
        while n < width as isize - big_n + 1 - (LANES - 1) {
            let s: [f32; 4] = core::array::from_fn(|k| {
                let m = n + k as isize;
                at(m - big_n - 1) + at(m + big_n - 1)
            });
            let mut o = [[0f32; 4]; 3];
            for (band, base) in [(0usize, 0usize), (1, 4), (2, 8)] {
                let (prev, prev2) = match band {
                    0 => (p1, q1),
                    1 => (p3, q3),
                    _ => (p5, q5),
                };
                for (lane, slot) in o[band].iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for (i, si) in s.iter().enumerate().take(lane + 1) {
                        acc = consts::CPP_MUL_IN[base + (lane - i)].mul_add(*si, acc);
                    }
                    acc = consts::CPP_MUL_PREV2[base + lane].mul_add(prev2, acc);
                    acc = consts::CPP_MUL_PREV[base + lane].mul_add(prev, acc);
                    *slot = acc;
                }
            }
            q1 = o[0][2];
            q3 = o[1][2];
            q5 = o[2][2];
            p1 = o[0][3];
            p3 = o[1][3];
            p5 = o[2][3];
            for lane in 0..4usize {
                output[(n + lane as isize) as usize] = o[0][lane] + o[1][lane] + o[2][lane];
            }
            n += LANES;
        }

        // Scalar tail.
        while n < width as isize {
            let sum = at(n - big_n - 1) + at(n + big_n - 1);
            let (mut pv, mut qv) = ((p1, p3, p5), (q1, q3, q5));
            let o = step(sum, &mut pv, &mut qv);
            (p1, p3, p5) = pv;
            (q1, q3, q5) = qv;
            if n >= 0 {
                output[n as usize] = o;
            }
            n += 1;
        }
    }

    pub fn vertical_pass_chunked<const J: usize, const K: usize>(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert!(J > K);
        assert!(K > 0);

        assert_eq!(input.len(), output.len());

        let mut x = 0;
        while x + J <= width {
            self.vertical_pass::<J>(&input[x..], &mut output[x..], width, height);
            x += J;
        }

        while x + K <= width {
            self.vertical_pass::<K>(&input[x..], &mut output[x..], width, height);
            x += K;
        }

        while x < width {
            self.vertical_pass::<1>(&input[x..], &mut output[x..], width, height);
            x += 1;
        }
    }

    // Apply 1D vertical scan on COLUMNS elements at a time
    pub fn vertical_pass<const COLUMNS: usize>(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert_eq!(input.len(), output.len());

        let big_n = consts::RADIUS as isize;

        let zeroes = vec![0f32; COLUMNS];
        let mut prev = vec![0f32; 3 * COLUMNS];
        let mut prev2 = vec![0f32; 3 * COLUMNS];
        let mut out = vec![0f32; 3 * COLUMNS];

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;
            let top_row = if top >= 0 {
                &input[top as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            let bottom_row = if bottom < height as isize {
                &input[bottom as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            for i in 0..COLUMNS {
                let sum = top_row[i] + bottom_row[i];

                let i1 = i;
                let i3 = i1 + COLUMNS;
                let i5 = i3 + COLUMNS;

                let out1 = prev[i1].mul_add(consts::VERT_MUL_PREV_1, prev2[i1]);
                let out3 = prev[i3].mul_add(consts::VERT_MUL_PREV_3, prev2[i3]);
                let out5 = prev[i5].mul_add(consts::VERT_MUL_PREV_5, prev2[i5]);

                let out1 = sum.mul_add(consts::VERT_MUL_IN_1, -out1);
                let out3 = sum.mul_add(consts::VERT_MUL_IN_3, -out3);
                let out5 = sum.mul_add(consts::VERT_MUL_IN_5, -out5);

                out[i1] = out1;
                out[i3] = out3;
                out[i5] = out5;

                if n >= 0 {
                    output[n as usize * width + i] = out1 + out3 + out5;
                }
            }

            prev2.copy_from_slice(&prev);
            prev.copy_from_slice(&out);

            n += 1;
        }
    }
}
