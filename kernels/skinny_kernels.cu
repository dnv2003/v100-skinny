// SM70 skinny NVFP4 dequant-GEMM: y[M,N] = x[M,K] @ W[K,N]
//
// Packed format (0.5625 bytes/weight, same density as the marlin path):
//   codes  uint8 [N][K/2]   two e2m1 codes per byte, low nibble = even k
//   scales uint8 [N][K/16]  fp8-e4m3 per 16-k group
//   gscale float            global scale, applied in the epilogue
//
// Two kernels, split by where V100 runs out of math:
//   simt (M<=8):  warp per output row, HFMA2 inner loop, k-pairs in the
//                 half2 lanes, x chunk in smem with an XOR bank swizzle.
//   wmma (M>=9):  block dequants a weight tile to smem, tensor cores
//                 (m16n16k16, fp32 accumulate) do the arithmetic.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define DEV_INLINE __device__ __forceinline__

// PRMT-LUT decoder (compile with -DSKINNY_LUT_CVT to select): loses
// to the TurboMind-derived shift+rebias decoder below by ~28% at M=1
// (504 vs 647 GB/s) and ~22% at M=16 - longer PRMT dependency chain
// and more INT-pipe ops per value. Kept for A/B reference.
// fp16 high bytes of the e2m1 magnitudes {0,.5,1,1.5} and {2,3,4,6};
// the low bytes are all 0x00, so one PRMT yields two packed halves.
constexpr unsigned LUT_LO = 0x3E3C3800u;
constexpr unsigned LUT_HI = 0x46444240u;

// Dequant one byte-pair position of an 8-code word. `q` holds 8 nibbles;
// byte `pi` gives codes (2p, 2p+1) -> returns them as one half2.
DEV_INLINE half2 dequant_pair(unsigned q, int pi, half2 sc2) {
  const unsigned mq = (q & 0x77777777u) >> (8 * pi);
  const unsigned sq = (q & 0x88888888u) >> (8 * pi);
  unsigned sel = ((mq & 0x7u) << 4) | ((mq & 0x70u) << 8);
  unsigned h = __byte_perm(LUT_LO, LUT_HI, sel);
  h |= ((sq & 0x8u) << 12) | ((sq & 0x80u) << 24);
  return __hmul2(*reinterpret_cast<half2 *>(&h), sc2);
}

DEV_INLINE half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));  // *256
  return __halves2half2(hs, hs);
}

// MXFP4 block scale: E8M0 is a bare exponent, value = 2^(b - 127). fp16
// stores exponent + 15, so the field is b - 112 and the mantissa stays 0 --
// one shift, no multiply. Callers keep the checkpoint's global factor in
// gm2, which is what holds b - 112 inside fp16's normal range (1..30).
DEV_INLINE half2 e8m0_to_half2(unsigned char b) {
  const unsigned short hb = (unsigned short)(((int)b - 112) << 10);
  const half hs = __ushort_as_half(hb);
  return __halves2half2(hs, hs);
}

// Group-scale layouts a weight tile can carry. NVFP4 ships one fp8-e4m3
// scale per 16 codes; MXFP4 one E8M0 scale per 32. Everything else -- code
// packing, fragment order, the decoder, the MMA -- is identical, so the
// kernels take this as a template argument instead of forking.
enum ScaleMode { SCALE_NVFP4_FP8_16 = 0, SCALE_MXFP4_E8M0_32 = 1 };

// Raw scale byte of group g and its decode, split so a caller can issue the
// load ahead of its use.
template <int MODE>
DEV_INLINE uint8_t group_scale_byte(const uint8_t *sb, int g) {
  if (MODE == SCALE_MXFP4_E8M0_32) return __ldg(sb + (size_t)(g >> 1) * 32);
  return __ldg(sb + (size_t)g * 32);
}

template <int MODE>
DEV_INLINE half2 decode_scale(uint8_t b) {
  if (MODE == SCALE_MXFP4_E8M0_32) return e8m0_to_half2(b);
  return fp8e4m3_to_half2(b);
}

template <int MODE>
DEV_INLINE half2 group_scale(const uint8_t *sb, int g) {
  return decode_scale<MODE>(group_scale_byte<MODE>(sb, g));
}

// XOR swizzle on the low 3 bits of a k-pair index; conflict-free for the
// simt read pattern (lane-groups sharing a bank base differ in p>>5).
DEV_INLINE int swz(int p) { return (p & ~7) | ((p ^ (p >> 5)) & 7); }

#ifndef SKINNY_LUT_CVT
// Alternative e2m1 decoder derived from TurboMind's cvt_f16x8_e2m1
// (Apache-2.0; 1Cat-vLLM csrc/sm70_turbomind/lmdeploy/src/turbomind/
// kernels/attention/quantization.h). Shifts sign/EM bits into fp16
// positions; the 2^14 exponent re-bias is folded into the caller's
// scale, so no extra multiply. Output half2 pairing is INTERLEAVED:
// out[i] holds codes (i, i+4) of the 8-code word.
DEV_INLINE void dequant8_tm(unsigned q, half2 sc2p, half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
  unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
  unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
  unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
  unsigned v3 = (q & S) | ((q >> 3) & EM);
  out[0] = __hmul2(*reinterpret_cast<half2 *>(&v0), sc2p);
  out[1] = __hmul2(*reinterpret_cast<half2 *>(&v1), sc2p);
  out[2] = __hmul2(*reinterpret_cast<half2 *>(&v2), sc2p);
  out[3] = __hmul2(*reinterpret_cast<half2 *>(&v3), sc2p);
}
#endif

// Stage 8 contiguous activation halves as four half2 pairs. Default
// pairing is adjacent-k; the TM decoder variant needs (k, k+4) pairs to
// match dequant8_tm's interleaved output.
DEV_INLINE void stage_pairs(half2 *dst, int base_pair, const uint4 &v) {
#ifndef SKINNY_LUT_CVT
  const unsigned *r = reinterpret_cast<const unsigned *>(&v);
  unsigned o[4] = {__byte_perm(r[0], r[2], 0x5410),
                   __byte_perm(r[0], r[2], 0x7632),
                   __byte_perm(r[1], r[3], 0x5410),
                   __byte_perm(r[1], r[3], 0x7632)};
#pragma unroll
  for (int j = 0; j < 4; j++)
    dst[swz(base_pair + j)] = *reinterpret_cast<half2 *>(&o[j]);
#else
  const half2 *hv = reinterpret_cast<const half2 *>(&v);
#pragma unroll
  for (int j = 0; j < 4; j++) dst[swz(base_pair + j)] = hv[j];
#endif
}

// ---------------------------------------------------------------------------
// SIMT kernel: 8 warps/block, one output row per warp.
// ---------------------------------------------------------------------------
template <int M, int KC, int R = 1, bool ARGMAX = false>
__global__ void skinny_nvfp4_simt(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K,
                                  float gscale, half *__restrict__ bvals,
                                  int *__restrict__ bidxs) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [M][KC/2] swizzled
  constexpr int P2 = KC / 2;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (blockIdx.x * 8 + warp) * R;
  const uint8_t *crow[R];
  const uint8_t *srow[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    crow[r] = codes + (size_t)(n + r) * (K >> 1);
    srow[r] = scales + (size_t)(n + r) * (K >> 4);
  }
  // Fold the global scale into the group scales so in-kernel weights sit
  // at their true O(0.1) magnitudes; otherwise code*fp8scale reaches
  // ~2.7e3 and fp16 products overflow on real activation outliers.
#ifndef SKINNY_LUT_CVT
  // dequant8_tm needs a 2^14 exponent re-bias; fold it here for free.
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
#else
  const half2 gm2 = __float2half2_rn(gscale);
#endif

  float accf[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) accf[r][m] = 0.f;

  int k0 = 0;
  for (; k0 + KC <= K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (KC / 8); idx += blockDim.x) {
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      const uint4 v =
          *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < KC / 512; i++) {
      const int s = lane + 32 * i;  // 16-code segment == one scale group
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      // fp16 accumulation window is one 16-code segment (8 products per
      // half2 lane); flushed to fp32 so real activation outliers cannot
      // overflow half range.
      half2 acch[R][M];
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acch[r][m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
#ifndef SKINNY_LUT_CVT
          dequant8_tm(qw, sc2[r], w4[r]);
#else
#pragma unroll
          for (int pi = 0; pi < 4; pi++)
            w4[r][pi] = dequant_pair(qw, pi, sc2[r]);
#endif
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const half2 xv = xs[m * P2 + psw];
#pragma unroll
            for (int r = 0; r < R; r++)
              acch[r][m] = __hfma2(w4[r][pi], xv, acch[r][m]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) {
          const float2 f = __half22float2(acch[r][m]);
          accf[r][m] += f.x + f.y;
        }
    }
  }

  // Tail chunk: K % KC remainder (any multiple of 128). Same layout and
  // swizzle, runtime segment bound with idle-lane guard.
  const int tail = K - k0;
  if (tail > 0) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (tail / 8); idx += blockDim.x) {
      const int m = idx / (tail / 8), j4 = idx % (tail / 8);
      const uint4 v =
          *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();
    const int nseg = tail >> 4;
    for (int s = lane; s < nseg; s += 32) {
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      half2 acch[R][M];
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acch[r][m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
#ifndef SKINNY_LUT_CVT
          dequant8_tm(qw, sc2[r], w4[r]);
#else
#pragma unroll
          for (int pi = 0; pi < 4; pi++)
            w4[r][pi] = dequant_pair(qw, pi, sc2[r]);
#endif
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const half2 xv = xs[m * P2 + psw];
#pragma unroll
            for (int r = 0; r < R; r++)
              acch[r][m] = __hfma2(w4[r][pi], xv, acch[r][m]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) {
          const float2 f = __half22float2(acch[r][m]);
          accf[r][m] += f.x + f.y;
        }
    }
  }

  if constexpr (!ARGMAX) {
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int m = 0; m < M; m++) {
        float v = accf[r][m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
        if (lane == 0) y[(size_t)m * N + n + r] = __float2half(v);
      }
  } else {
    // Fused greedy argmax (M=1): identical reduce, identical
    // __float2half rounding, then compare halfs with strict > so ties
    // keep the LOWEST index — matching argmax-over-half semantics of
    // the separate path. One (val, idx) pair per block; no logits hit
    // HBM.
    __shared__ half wval[8];
    __shared__ int widx[8];
    half best_h = __float2half(-3.0e38f);
    int best_i = -1;
#pragma unroll
    for (int r = 0; r < R; r++) {
      float v = accf[r][0];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) {
        const half h = __float2half(v);
        if (__hgt(h, best_h)) { best_h = h; best_i = n + r; }
      }
    }
    if (lane == 0) { wval[warp] = best_h; widx[warp] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0) {
      half bh = wval[0];
      int bi = widx[0];
#pragma unroll
      for (int w = 1; w < 8; w++)
        if (__hgt(wval[w], bh)) { bh = wval[w]; bi = widx[w]; }
      bvals[blockIdx.x] = bh;
      bidxs[blockIdx.x] = bi;
    }
  }
}

// ---------------------------------------------------------------------------
// WMMA kernel: WN x WM warps of 16x16 output tiles, KC-deep smem staging.
// Software-pipelined: while the tensor cores chew on chunk i, each thread's
// gmem loads for chunk i+1 are already in flight into registers.
// ---------------------------------------------------------------------------
template <int WN, int WM, int KC>
__global__ void skinny_nvfp4_wmma(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K,
                                  int m_real, float gscale) {
  constexpr int NT = WN * 16, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;  // padded smem pitches (halfs)
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;  // code segs per thread
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;   // x uint4s per thread
  static_assert(CSEG * NTHREADS == NT * (KC / 16), "code seg split");
  static_assert(XSEG * NTHREADS == MT * (KC / 8), "x seg split");

  extern __shared__ char smem_raw[];
  half *ws = reinterpret_cast<half *>(smem_raw);  // [NT][PW]
  half *xs = ws + NT * PW;                        // [MT][PX]

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int nb = blockIdx.x * NT;

  uint2 st_c[CSEG];
  unsigned char st_s[CSEG];
  uint4 st_x[XSEG];

  auto load_stage = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      st_c[i] = __ldcs(reinterpret_cast<const uint2 *>(
          codes + (size_t)(nb + n) * (K >> 1) + (k0 >> 1) + s * 8));
      st_s[i] = __ldcs(scales + (size_t)(nb + n) * (K >> 4) + (k0 >> 4) + s);
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      st_x[i] = (m < m_real)
                    ? *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 +
                                                       j4 * 8)
                    : make_uint4(0, 0, 0, 0);
    }
  };

  auto store_stage = [&]() {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      const half2 sc2 = fp8e4m3_to_half2(st_s[i]);
      half2 *wrow = reinterpret_cast<half2 *>(ws + n * PW + s * 16);
      const unsigned qs[2] = {st_c[i].x, st_c[i].y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
#ifndef SKINNY_LUT_CVT
        half2 t[4];
        dequant8_tm(qs[w], sc2, t);  // values carry a 2^-14 factor here
        const unsigned *tr = reinterpret_cast<const unsigned *>(t);
        unsigned lin[4] = {__byte_perm(tr[0], tr[1], 0x5410),
                           __byte_perm(tr[2], tr[3], 0x5410),
                           __byte_perm(tr[0], tr[1], 0x7632),
                           __byte_perm(tr[2], tr[3], 0x7632)};
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = *reinterpret_cast<half2 *>(&lin[pi]);
#else
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = dequant_pair(qs[w], pi, sc2);
#endif
      }
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4 *>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::fill_fragment(cfrag, 0.f);

  load_stage(0);
  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();  // previous chunk's mma done; smem free to overwrite
    store_stage();
    __syncthreads();
    if (k0 + KC < K) load_stage(k0 + KC);  // in flight during the mma loop

    // Double-buffered fragments: kk+1's shared loads overlap kk's mma.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
    wmma::load_matrix_sync(a[0], ws + wn * 16 * PW, PW);
    wmma::load_matrix_sync(b[0], xs + wm * 16 * PX, PX);
#pragma unroll
    for (int kk = 0; kk < KC / 16; kk++) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < KC / 16) {
        wmma::load_matrix_sync(a[nxt], ws + wn * 16 * PW + (kk + 1) * 16, PW);
        wmma::load_matrix_sync(b[nxt], xs + wm * 16 * PX + (kk + 1) * 16, PX);
      }
      wmma::mma_sync(cfrag, a[cur], b[cur], cfrag);
    }
  }

  __syncthreads();  // done with ws/xs; reuse for the fp32 epilogue stage
  float *cs = reinterpret_cast<float *>(smem_raw) + warp * 256;
  wmma::store_matrix_sync(cs, cfrag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;  // i: n within tile, j: m within tile
    const int gm = wm * 16 + j, gn = nb + wn * 16 + i;
    #ifndef SKINNY_LUT_CVT
    const float gs_eff = gscale * 16384.f;  // undo dequant8_tm's 2^-14
#else
    const float gs_eff = gscale;
#endif
    if (gm < m_real) y[(size_t)gm * N + gn] = __float2half(cs[e] * gs_eff);
  }
}

// ---------------------------------------------------------------------------
// DP4A kernel: int8 SIMT path for the compute-bound M band (4..16).
//
// The e2m1 lattice x2 is integer-exact ({0,1,2,3,4,6,8,12}), so the
// weight side is lossless int8; only activations are quantized
// (symmetric per-16-group int8, done host-side per GEMM call and
// amortized over all N rows). dp4a delivers 4 MACs per issue slot vs
// HFMA2's 2, which is the whole bet: the M-scaling penalty of the
// fp16 SIMT path is issue-bound MAC work.
// ---------------------------------------------------------------------------
DEV_INLINE unsigned dp4a_unpack_mag(unsigned nib4) {
  // nib4 holds 4 e2m1 codes in the low nibbles of each byte
  // (0x0c0c0c0c layout). Returns 4 packed uint8 magnitudes x2 via the
  // shift identity: mag2(c) = c < 2 ? c : (2 + (c&1)) << ((c>>1) - 1).
  unsigned out = 0;
#pragma unroll
  for (int b = 0; b < 4; b++) {
    const unsigned c = (nib4 >> (8 * b)) & 0x7u;
    const unsigned m = c < 2 ? c : (2u + (c & 1u)) << ((c >> 1) - 1u);
    out |= m << (8 * b);
  }
  return out;
}

// One thread per 16-element group: absmax -> scale -> int8 quantize.
// Single launch replaces an 8-op torch chain (which cost 168us of
// launch overhead at these sizes).
__global__ void skinny_quant_a8(const half *__restrict__ x,
                                int8_t *__restrict__ xq,
                                float *__restrict__ xs, int total_groups,
                                int k) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total_groups) return;
  const half2 *xp = reinterpret_cast<const half2 *>(x) + g * 8;
  float amax = 1e-8f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const half2 v = xp[i];
    amax = fmaxf(amax, fmaxf(fabsf(__half2float(__low2half(v))),
                             fabsf(__half2float(__high2half(v)))));
  }
  const float s = amax / 127.f, inv = 127.f / amax;
  char4 *out = reinterpret_cast<char4 *>(xq) + g * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const half2 a = xp[i * 2], b = xp[i * 2 + 1];
    char4 c;
    c.x = (signed char)__float2int_rn(__half2float(__low2half(a)) * inv);
    c.y = (signed char)__float2int_rn(__half2float(__high2half(a)) * inv);
    c.z = (signed char)__float2int_rn(__half2float(__low2half(b)) * inv);
    c.w = (signed char)__float2int_rn(__half2float(__high2half(b)) * inv);
    out[i] = c;
  }
  xs[g] = s;
  (void)k;
}

template <int M, int KC>
__global__ void skinny_nvfp4_dp4a(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const int8_t *__restrict__ xq,
                                  const float *__restrict__ xs_scale,
                                  half *__restrict__ y, int N, int K,
                                  float gscale) {
  extern __shared__ char smem_raw[];
  // staged activations: int32 words [M][KC/4] then scales [M][KC/16]
  int *xw = reinterpret_cast<int *>(smem_raw);
  float *xsc = reinterpret_cast<float *>(xw + M * (KC / 4));
  constexpr int PW4 = KC / 4, PS = KC / 16;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + warp;
  const uint8_t *crow = codes + (size_t)n * (K >> 1);
  const uint8_t *srow = scales + (size_t)n * (K >> 4);

  float accf[M];
#pragma unroll
  for (int m = 0; m < M; m++) accf[m] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (KC / 4); idx += blockDim.x) {
      const int m = idx / (KC / 4), j = idx % (KC / 4);
      const int js = (j & ~3) | ((j ^ (j >> 5)) & 3);  // bank swizzle
      xw[m * PW4 + js] = reinterpret_cast<const int *>(
          xq + (size_t)m * K + k0)[j];
    }
    for (int idx = threadIdx.x; idx < M * (KC / 16); idx += blockDim.x) {
      const int m = idx / (KC / 16), g = idx % (KC / 16);
      xsc[m * PS + g] = xs_scale[(size_t)m * (K / 16) + (k0 / 16) + g];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < (KC / 16 + 31) / 32; i++) {
      const int s = lane + 32 * i;  // 16-code segment == one scale group
      if (KC / 16 < 32 && s >= KC / 16) break;
      const uint2 q2 = *reinterpret_cast<const uint2 *>(
          crow + (k0 >> 1) + s * 8);
      // fp8 scale -> float, x0.5 compensates the x2 integer lattice
      const half2 sch = fp8e4m3_to_half2(srow[(k0 >> 4) + s]);
      const float wsc = __half2float(__low2half(sch)) * 0.5f;

      // unpack 16 codes -> 4 dp4a words (int8, sign applied)
      unsigned w4[4];
#pragma unroll
      for (int hw = 0; hw < 2; hw++) {
        const unsigned q = hw == 0 ? q2.x : q2.y;
        const unsigned lo = q & 0x0F0F0F0Fu;         // codes 0,2,4,6
        const unsigned hi = (q >> 4) & 0x0F0F0F0Fu;  // codes 1,3,5,7
        // interleave back to k-order nibble words: bytes of `lo` are
        // even k, bytes of `hi` odd k -> two words of 4 consecutive k
        const unsigned w01 = __byte_perm(lo, hi, 0x5140);  // k0..k3
        const unsigned w23 = __byte_perm(lo, hi, 0x7362);  // k4..k7
        const unsigned wds[2] = {w01, w23};
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const unsigned mag = dp4a_unpack_mag(wds[t]);
          const unsigned sgn = (wds[t] >> 3) & 0x01010101u;  // sign bits
          const unsigned msk = sgn * 0xFFu;  // 0x00 or 0xFF per byte
          // two's complement negate where sign: (mag ^ msk) + (msk & 1)
          w4[hw * 2 + t] = __vadd4(mag ^ msk, sgn);
        }
      }

      int acc32[M];
#pragma unroll
      for (int m = 0; m < M; m++) acc32[m] = 0;
#pragma unroll
      for (int m = 0; m < M; m++) {
        const int *xrow = xw + m * PW4;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const int j = s * 4 + t;
          const int js = (j & ~3) | ((j ^ (j >> 5)) & 3);
          acc32[m] = __dp4a((int)w4[t], xrow[js], acc32[m]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++)
        accf[m] += (float)acc32[m] * (wsc * xsc[m * PS + s]);
    }
  }

  // warp-reduce the per-lane segment partials before the single write
  const float gs = gscale;
#pragma unroll
  for (int m = 0; m < M; m++) {
    float v = accf[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) y[(size_t)m * N + n] = __float2half(v * gs);
  }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------
static void check_inputs(const torch::Tensor &x, const torch::Tensor &codes,
                         const torch::Tensor &scales, int64_t &m, int64_t &n,
                         int64_t &k) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 &&
              scales.is_contiguous());
  m = x.size(0);
  k = x.size(1);
  n = codes.size(0);
  TORCH_CHECK(codes.size(1) * 2 == k, "codes/x K mismatch");
  TORCH_CHECK(scales.size(0) == n && scales.size(1) * 16 == k);
}

// ---------------------------------------------------------------------------
// MMA8 kernel: Volta mma.sync.m8n8k4 register-fragment path for 2<=M<=8.
//
// The 16-row WMMA tile pads M=5 verify batches 3.2x; SIMT carries M FMAs
// per weight byte and goes compute-bound by M=5. This path streams weights
// through registers exactly like SIMT but hands the MACs to the tensor
// cores at their native 8-row tile, so M<=8 all run at the same cost.
//
// Fragment maps were derived empirically on V100 (mma8_probe.cu):
//   A row-major / B col-major: QP lanes {0-3,16-19} hold row/col
//   (lane&3)+4*(lane>=16), 4 contiguous k each.
//   C fp32: reg i of lane L -> row (i&2)|(L>=16?4:0)|(L&1),
//   col (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2).
// Warp = 8 weight rows; the 4 QPs split K in 64-element superchunks
// (32B-sector-coalesced code reads), butterfly-reduced once at the end.
// fp32 accumulation end to end: no fp16 overflow window.
// ---------------------------------------------------------------------------
template <int KC>
__global__ void skinny_nvfp4_mma8(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K, int M,
                                  float gscale) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [8][PITCH] plain k order
  constexpr int PITCH = KC / 2 + 1;  // odd pitch: 8 rows spread the banks

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int n0 = (blockIdx.x * (int)(blockDim.x >> 5) + warp) * 8;
  const uint8_t *crow = codes + (size_t)(n0 + idx) * (K >> 1);
  const uint8_t *srow = scales + (size_t)(n0 + idx) * (K >> 4);

#ifndef SKINNY_LUT_CVT
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
#else
  const half2 gm2 = __float2half2_rn(gscale);
#endif

  float c[8];
#pragma unroll
  for (int i = 0; i < 8; i++) c[i] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();
    for (int t = threadIdx.x; t < 8 * (KC / 2); t += blockDim.x) {
      const int m = t / (KC / 2), j = t % (KC / 2);
      half2 v = __float2half2_rn(0.f);
      if (m < M)
        v = *reinterpret_cast<const half2 *>(x + (size_t)m * K + k0 + 2 * j);
      xs[m * PITCH + j] = v;
    }
    __syncthreads();

#pragma unroll
    for (int s = 0; s < KC / 64; s++) {
      const int kb = k0 + s * 64 + qp * 16;  // this QP's 16-k window
      const uint2 q2 =
          __ldcs(reinterpret_cast<const uint2 *>(crow + (kb >> 1)));
      const half2 sc2 = __hmul2(fp8e4m3_to_half2(__ldg(srow + (kb >> 4))), gm2);
      // A fragments, k-contiguous quads for the 4 mmas of this window.
      half2 af[4][2];
#ifndef SKINNY_LUT_CVT
      half2 wa[4], wb[4];
      dequant8_tm(q2.x, sc2, wa);  // interleaved pairs (k, k+4)
      dequant8_tm(q2.y, sc2, wb);
      af[0][0] = __lows2half2(wa[0], wa[1]);
      af[0][1] = __lows2half2(wa[2], wa[3]);
      af[1][0] = __highs2half2(wa[0], wa[1]);
      af[1][1] = __highs2half2(wa[2], wa[3]);
      af[2][0] = __lows2half2(wb[0], wb[1]);
      af[2][1] = __lows2half2(wb[2], wb[3]);
      af[3][0] = __highs2half2(wb[0], wb[1]);
      af[3][1] = __highs2half2(wb[2], wb[3]);
#else
#pragma unroll
      for (int j = 0; j < 4; j++) {  // adjacent pairs are already k-order
        const unsigned qw = (j < 2) ? q2.x : q2.y;
        af[j][0] = dequant_pair(qw, (j & 1) * 2, sc2);
        af[j][1] = dequant_pair(qw, (j & 1) * 2 + 1, sc2);
      }
#endif
      const half2 *xrow = xs + idx * PITCH + ((kb - k0) >> 1);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const half2 b0 = xrow[2 * j], b1 = xrow[2 * j + 1];
        const unsigned *A = reinterpret_cast<const unsigned *>(af[j]);
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
            "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]),
              "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
            : "r"(A[0]), "r"(A[1]),
              "r"(*reinterpret_cast<const unsigned *>(&b0)),
              "r"(*reinterpret_cast<const unsigned *>(&b1)));
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    c[i] += __shfl_xor_sync(0xffffffffu, c[i], 4);
    c[i] += __shfl_xor_sync(0xffffffffu, c[i], 8);
  }
  if ((lane & 12) == 0) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (cc < M) y[(size_t)cc * N + n0 + r] = __float2half(c[i]);
    }
  }
}

std::vector<torch::Tensor> skinny_gemm_simt_argmax(torch::Tensor x,
                                                   torch::Tensor codes,
                                                   torch::Tensor scales,
                                                   double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  constexpr int KC = 1024;
  TORCH_CHECK(m == 1, "fused argmax is M=1 only");
  TORCH_CHECK(k % 128 == 0 && k >= 128, "K must be a multiple of 128");
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  const bool two_rows = (k <= 2048) && (n % 16 == 0);
  const int nblocks = two_rows ? (int)n / 16 : (int)n / 8;
  auto bvals = torch::empty({nblocks}, x.options());
  auto bidxs = torch::empty({nblocks},
                            x.options().dtype(torch::kInt32));
  const dim3 grid(nblocks), block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (KC / 2) * sizeof(half2);
  if (two_rows)
    skinny_nvfp4_simt<1, KC, 2, true><<<grid, block, smem, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()), nullptr,
        (int)n, (int)k, (float)gscale,
        reinterpret_cast<half *>(bvals.data_ptr<at::Half>()),
        bidxs.data_ptr<int>());
  else
    skinny_nvfp4_simt<1, KC, 1, true><<<grid, block, smem, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()), nullptr,
        (int)n, (int)k, (float)gscale,
        reinterpret_cast<half *>(bvals.data_ptr<at::Half>()),
        bidxs.data_ptr<int>());
  return {bvals, bidxs};
}

torch::Tensor skinny_gemm_simt(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  constexpr int KC = 1024;
  TORCH_CHECK(k % 128 == 0 && k >= 128, "K must be a multiple of 128");
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  auto y = torch::empty({m, n}, x.options());
  // Short-K rows leave <2 weight loads in flight per thread; two rows
  // per warp restores latency hiding (shape diagnostic: out_proj K=1536
  // ran at 66% of flagship bandwidth with one row per warp).
  const bool two_rows = (k <= 2048) && (n % 16 == 0);
  const dim3 grid(two_rows ? n / 16 : n / 8), block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (int)m * (KC / 2) * sizeof(half2);

#define LAUNCH_SIMT(MM)                                                   \
  if (two_rows)                                                           \
    skinny_nvfp4_simt<MM, KC, 2><<<grid, block, smem, stream>>>(          \
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),            \
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),           \
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k, \
        (float)gscale, nullptr, nullptr);                                 \
  else                                                                    \
    skinny_nvfp4_simt<MM, KC, 1><<<grid, block, smem, stream>>>(          \
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),              \
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
      reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
      (float)gscale, nullptr, nullptr)

  switch (m) {
    case 1: LAUNCH_SIMT(1); break;
    case 2: LAUNCH_SIMT(2); break;
    case 3: LAUNCH_SIMT(3); break;
    case 4: LAUNCH_SIMT(4); break;
    case 5: LAUNCH_SIMT(5); break;
    case 6: LAUNCH_SIMT(6); break;
    case 7: LAUNCH_SIMT(7); break;
    case 8: LAUNCH_SIMT(8); break;
    default: TORCH_CHECK(false, "simt kernel supports M in 1..8, got ", m);
  }
#undef LAUNCH_SIMT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_wmma(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_WMMA(WN, WM, KC)                                             \
  do {                                                                      \
    constexpr int NT = WN * 16, MT = WM * 16;                               \
    TORCH_CHECK(n % NT == 0, "N must be a multiple of ", NT);               \
    TORCH_CHECK(k % KC == 0, "K must be a multiple of ", KC);               \
    const int smem = (NT + MT) * (KC + 16) * (int)sizeof(half);             \
    skinny_nvfp4_wmma<WN, WM, KC>                                           \
        <<<dim3(n / NT), dim3(WN * WM * 32), smem, stream>>>(               \
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),          \
            reinterpret_cast<const half *>(x.data_ptr<at::Half>()),         \
            reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n,       \
            (int)k, (int)m, (float)gscale);                                 \
  } while (0)

  if (m <= 16) LAUNCH_WMMA(4, 1, 256);       // 128B code rows, 2 CTAs/SM
  // (2,2,128) for the 17-32 band: NT=32 doubles the grid on N-starved
  // shapes (occupancy sweep 2026-08: 0.327 vs 0.350 ms/layer-set, -6.6%).
  else if (m <= 32) LAUNCH_WMMA(2, 2, 128);
  else if (m <= 64) LAUNCH_WMMA(2, 4, 128);
  else TORCH_CHECK(false, "wmma kernel supports M <= 64, got ", m);
#undef LAUNCH_WMMA
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_mma8(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  constexpr int KC = 256, WARPS = 4;
  TORCH_CHECK(m >= 1 && m <= 8, "mma8 kernel supports M in 1..8, got ", m);
  TORCH_CHECK(k % KC == 0 && k >= KC, "K must be a multiple of ", KC);
  TORCH_CHECK(n % (WARPS * 8) == 0, "N must be a multiple of ", WARPS * 8);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = 8 * (KC / 2 + 1) * (int)sizeof(half2);
  skinny_nvfp4_mma8<KC>
      <<<dim3(n / (WARPS * 8)), dim3(WARPS * 32), smem, stream>>>(
          codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,
          (int)m, (float)gscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Split-K variant: each CTA covers one K-slice, accumulating fp32
// partials into ypart via atomicAdd (S contenders per element at most).
// Caller converts ypart * gs_eff -> half. Multiplies grid occupancy by
// the split factor on N-starved shapes.
template <int WN, int WM, int KC>
__global__ void skinny_nvfp4_wmma_ks(const uint8_t *__restrict__ codes,
                                     const uint8_t *__restrict__ scales,
                                     const half *__restrict__ x,
                                     float *__restrict__ ypart, int N, int K,
                                     int m_real, int k_slice) {
  constexpr int NT = WN * 16, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;
  static_assert(CSEG * NTHREADS == NT * (KC / 16), "code seg split");
  static_assert(XSEG * NTHREADS == MT * (KC / 8), "x seg split");

  extern __shared__ char smem_raw[];
  half *ws = reinterpret_cast<half *>(smem_raw);
  half *xs = ws + NT * PW;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int nb = blockIdx.x * NT;
  const int kbeg = blockIdx.y * k_slice;
  const int kend = min(kbeg + k_slice, K);

  uint2 st_c[CSEG];
  unsigned char st_s[CSEG];
  uint4 st_x[XSEG];

  auto load_stage = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      st_c[i] = __ldcs(reinterpret_cast<const uint2 *>(
          codes + (size_t)(nb + n) * (K >> 1) + (k0 >> 1) + s * 8));
      st_s[i] = __ldcs(scales + (size_t)(nb + n) * (K >> 4) + (k0 >> 4) + s);
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      st_x[i] = (m < m_real)
                    ? *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 +
                                                       j4 * 8)
                    : make_uint4(0, 0, 0, 0);
    }
  };
  auto store_stage = [&]() {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      const half2 sc2 = fp8e4m3_to_half2(st_s[i]);
      half2 *wrow = reinterpret_cast<half2 *>(ws + n * PW + s * 16);
      const unsigned qs[2] = {st_c[i].x, st_c[i].y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
#ifndef SKINNY_LUT_CVT
        half2 t[4];
        dequant8_tm(qs[w], sc2, t);
        const unsigned *tr = reinterpret_cast<const unsigned *>(t);
        unsigned lin[4] = {__byte_perm(tr[0], tr[1], 0x5410),
                           __byte_perm(tr[2], tr[3], 0x5410),
                           __byte_perm(tr[0], tr[1], 0x7632),
                           __byte_perm(tr[2], tr[3], 0x7632)};
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = *reinterpret_cast<half2 *>(&lin[pi]);
#else
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = dequant_pair(qs[w], pi, sc2);
#endif
      }
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4 *>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::fill_fragment(cfrag, 0.f);

  load_stage(kbeg);
  for (int k0 = kbeg; k0 < kend; k0 += KC) {
    __syncthreads();
    store_stage();
    __syncthreads();
    if (k0 + KC < kend) load_stage(k0 + KC);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
    wmma::load_matrix_sync(a[0], ws + wn * 16 * PW, PW);
    wmma::load_matrix_sync(b[0], xs + wm * 16 * PX, PX);
#pragma unroll
    for (int kk = 0; kk < KC / 16; kk++) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < KC / 16) {
        wmma::load_matrix_sync(a[nxt], ws + wn * 16 * PW + (kk + 1) * 16, PW);
        wmma::load_matrix_sync(b[nxt], xs + wm * 16 * PX + (kk + 1) * 16, PX);
      }
      wmma::mma_sync(cfrag, a[cur], b[cur], cfrag);
    }
  }

  __syncthreads();
  float *cs = reinterpret_cast<float *>(smem_raw) + warp * 256;
  wmma::store_matrix_sync(cs, cfrag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;
    const int gm = wm * 16 + j, gn = nb + wn * 16 + i;
    if (gm < m_real) atomicAdd(&ypart[(size_t)gm * N + gn], cs[e]);
  }
}

// Config-selectable WMMA entry for tile sweeps: cfg indexes the
// (WN, WM, KC) table below. Configs 8-9 exceed the default 48KB smem
// ceiling and require carveout=true (96KB opt-in). N must divide WN*16.
static void set_smem_opt(const void *kern, int smem) {
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       96 * 1024);
  cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout,
                       100);
  (void)smem;
}

torch::Tensor skinny_gemm_wmma_cfg(torch::Tensor x, torch::Tensor codes,
                                   torch::Tensor scales, double gscale,
                                   int64_t cfg, bool carveout) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_WMMA_CFG(WN, WM, KC)                                         \
  do {                                                                      \
    constexpr int NT = WN * 16, MT = WM * 16;                               \
    TORCH_CHECK(n % NT == 0, "N must be a multiple of ", NT);               \
    TORCH_CHECK(k % KC == 0, "K must be a multiple of ", KC);               \
    TORCH_CHECK(m <= MT, "cfg supports M <= ", MT);                         \
    const int smem = (NT + MT) * (KC + 16) * (int)sizeof(half);             \
    TORCH_CHECK(smem <= 48 * 1024 || carveout,                              \
                "cfg needs 96KB smem carveout");                            \
    if (carveout)                                                           \
      set_smem_opt((const void *)&skinny_nvfp4_wmma<WN, WM, KC>, smem);     \
    skinny_nvfp4_wmma<WN, WM, KC>                                           \
        <<<dim3(n / NT), dim3(WN * WM * 32), smem, stream>>>(               \
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),          \
            reinterpret_cast<const half *>(x.data_ptr<at::Half>()),         \
            reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n,       \
            (int)k, (int)m, (float)gscale);                                 \
  } while (0)

  switch (cfg) {
    case 0: LAUNCH_WMMA_CFG(4, 1, 256); break;
    case 1: LAUNCH_WMMA_CFG(4, 2, 128); break;
    case 2: LAUNCH_WMMA_CFG(2, 4, 128); break;
    case 3: LAUNCH_WMMA_CFG(8, 1, 128); break;
    case 4: LAUNCH_WMMA_CFG(8, 2, 128); break;
    case 5: LAUNCH_WMMA_CFG(4, 4, 128); break;
    case 6: LAUNCH_WMMA_CFG(2, 1, 256); break;
    case 7: LAUNCH_WMMA_CFG(2, 2, 128); break;
    case 8: LAUNCH_WMMA_CFG(4, 1, 512); break;
    case 9: LAUNCH_WMMA_CFG(2, 4, 256); break;
    default: TORCH_CHECK(false, "unknown wmma cfg ", cfg);
  }
#undef LAUNCH_WMMA_CFG
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Split-K entry: returns fp32 partial sums (caller applies gscale and
// converts). splits CTAs along K; k must divide evenly into
// KC-aligned slices.
torch::Tensor skinny_gemm_wmma_splitk(torch::Tensor x, torch::Tensor codes,
                                      torch::Tensor scales, int64_t cfg,
                                      int64_t splits, bool carveout) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  auto ypart = torch::zeros(
      {m, n}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_WMMA_KS(WN, WM, KC)                                         \
  do {                                                                     \
    constexpr int NT = WN * 16, MT = WM * 16;                              \
    TORCH_CHECK(n % NT == 0, "N must be a multiple of ", NT);              \
    TORCH_CHECK(m <= MT, "cfg supports M <= ", MT);                        \
    const int64_t kper = (k / splits / KC) * KC;                           \
    TORCH_CHECK(kper > 0 && kper * splits == k,                            \
                "K must split into KC-aligned slices");                    \
    const int smem = (NT + MT) * (KC + 16) * (int)sizeof(half);            \
    TORCH_CHECK(smem <= 48 * 1024 || carveout,                             \
                "cfg needs 96KB smem carveout");                           \
    if (carveout)                                                          \
      set_smem_opt((const void *)&skinny_nvfp4_wmma_ks<WN, WM, KC>, smem); \
    skinny_nvfp4_wmma_ks<WN, WM, KC>                                       \
        <<<dim3(n / NT, splits), dim3(WN * WM * 32), smem, stream>>>(      \
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),         \
            reinterpret_cast<const half *>(x.data_ptr<at::Half>()),        \
            ypart.data_ptr<float>(), (int)n, (int)k, (int)m, (int)kper);   \
  } while (0)

  switch (cfg) {
    case 0: LAUNCH_WMMA_KS(4, 1, 256); break;
    case 1: LAUNCH_WMMA_KS(4, 2, 128); break;
    case 2: LAUNCH_WMMA_KS(2, 4, 128); break;
    case 6: LAUNCH_WMMA_KS(2, 1, 256); break;
    case 7: LAUNCH_WMMA_KS(2, 2, 128); break;
    default: TORCH_CHECK(false, "splitk unsupported for cfg ", cfg);
  }
#undef LAUNCH_WMMA_KS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ypart;
}

torch::Tensor skinny_gemm_dp4a(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  const int kc = (k % 1024 == 0) ? 1024 : 256;
  TORCH_CHECK(k % kc == 0, "K must be a multiple of 256");

  auto xq = torch::empty({m, k}, x.options().dtype(torch::kChar));
  auto xsc = torch::empty({m, k / 16}, x.options().dtype(torch::kFloat32));
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int groups = (int)(m * (k / 16));
  skinny_quant_a8<<<(groups + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
      xq.data_ptr<int8_t>(), xsc.data_ptr<float>(), groups, (int)k);

#define LAUNCH_DP4A(MM, KCC)                                              \
  skinny_nvfp4_dp4a<MM, KCC><<<n / 8, 256,                                \
      MM * (KCC / 4) * 4 + MM * (KCC / 16) * 4, stream>>>(                \
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),              \
      xq.data_ptr<int8_t>(), xsc.data_ptr<float>(),                      \
      reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
      (float)gscale)
#define DP4A_CASE(MM)                                                     \
  case MM:                                                                \
    if (kc == 1024) LAUNCH_DP4A(MM, 1024); else LAUNCH_DP4A(MM, 256);    \
    break

  switch (m) {
    DP4A_CASE(4);
    DP4A_CASE(5);
    DP4A_CASE(7);
    DP4A_CASE(8);
    DP4A_CASE(11);
    DP4A_CASE(16);
    default: TORCH_CHECK(false, "dp4a supports M in {4,5,7,8,11,16}");
  }
#undef DP4A_CASE
#undef LAUNCH_DP4A
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// QPN kernel: Volta-native four-quadpair mma.m8n8k4, M 4..16 band.
//
// The quadpairs split the N dimension: one warp instruction = four
// independent 8x8x4 MMAs sharing a single 8x4 activation A tile (the A
// fragment map depends only on lane-position inside the quadpair, so
// QP-sibling lanes hold identical A registers). MT template = number of
// 8-row A tiles (MT=2 decodes B once for M 9..16). Weights arrive
// PREPACKED in fragment order ([tile N/32][group K/16][lane 32] x 8B,
// nibbles pre-interleaved so dequant8_tm's (i, i+4) output IS the
// adjacent-k B register pair) — built once at weight load by the shim's
// _qpn_prepack; the checkpoint-native layout stays for SIMT/WMMA.
// No smem in the main loop, no barriers except the cross-warp K-reduce
// at output (CTA = 4 warps splitting K to keep the grid at N/32).
// Frontier (qpn_sweep_20260810): simt M<=3, qpn 4..16, wmma 17..64 —
// 1.28x at M=5, 1.69x at M=8, 1.29x at M=11, 1.22x at M=16 vs the
// prior best incumbent on the 5-shape production set.
// ---------------------------------------------------------------------------
#define MMA_8N8K4(C, A0, A1, B0, B1)                                        \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int MT>
__global__ void skinny_nvfp4_qpn(const uint8_t *__restrict__ qcodes,
                                 const uint8_t *__restrict__ qscales,
                                 const half *__restrict__ x,
                                 half *__restrict__ y, int N, int K, int M,
                                 float gscale) {
  constexpr int WARPS = 4;
  __shared__ float cs[WARPS][MT * 256];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);  // A row & B local col
  const int G = K >> 4, Gq = G / WARPS;
  const int g0 = warp * Gq;
  const uint2 *cb = reinterpret_cast<const uint2 *>(qcodes) +
                    (size_t)tile * G * 32 + lane;
  const uint8_t *sb = qscales + (size_t)tile * G * 32 + lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[MT][8];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[t][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const half2 sc2 =
        __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
    half2 b[8];
    dequant8_tm(q2.x, sc2, b + 0);  // slices 0,1 (k0..7 adjacent pairs)
    dequant8_tm(q2.y, sc2, b + 4);  // slices 2,3 (k8..15)
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
#pragma unroll
    for (int t = 0; t < MT; t++) {
      const int ar = t * 8 + r;
      uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
      if (ar < M) {
        const half *xrow = x + (size_t)ar * K;
        a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
        a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
      }
      const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
      const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
      MMA_8N8K4(c[t], A0[0], A0[1], B[0], B[1]);
      MMA_8N8K4(c[t], A0[2], A0[3], B[2], B[3]);
      MMA_8N8K4(c[t], A1[0], A1[1], B[4], B[5]);
      MMA_8N8K4(c[t], A1[2], A1[3], B[6], B[7]);
    }
  }

  // C map (mma8_probe.cu, roles swapped): reg i of lane L ->
  //   A-row (i&2)|((L&16)?4:0)|(L&1); B-col (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2)
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp][(t * 8 + row) * 32 + qp * 8 + col] = c[t][i];
    }
  __syncthreads();  // the only barrier: cross-warp K reduce
  for (int e = threadIdx.x; e < MT * 256; e += blockDim.x) {
    const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
    const int row = e >> 5, col = e & 31;
    if (row < M)
      y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

// QPN-layout SIMT kernel (M<=3): serves the decode band from the SAME
// prepacked fragment-order buffers as gemm_qpn, so the CT-native stash
// can be dropped entirely (VLLM_SKINNY_DROP_CT=1 -> weights return to
// the pre-QPN footprint and the fp32 GDN state cache fits).
// Geometry mirrors gemm_qpn: CTA = 4 warps on one 32-column tile with K
// split across warps; lane owns one column, so every warp code load is
// one 256B line. Whole activation block (M<=3 rows) staged to smem once;
// korder nibble pairing makes dequant8_tm output the adjacent-k x pairs.
// fp32 accumulation throughout; single barrier before the cross-warp
// reduce epilogue.
template <int M>
__global__ void skinny_nvfp4_qpn_simt(const uint8_t *__restrict__ qcodes,
                                      const uint8_t *__restrict__ qscales,
                                      const half *__restrict__ x,
                                      half *__restrict__ y, int N, int K,
                                      float gscale) {
  constexpr int WARPS = 4;
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);      // [M][K/2]
  float *cs = reinterpret_cast<float *>(xs + (size_t)M * (K >> 1));
  // cs: [WARPS][32][M] fp32 partials

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int G = K >> 4, Gq = G / WARPS;
  const int g0 = warp * Gq;
  constexpr int KCH2 = 4096;  // half2 per x chunk (M=3 -> 48KB smem)
  const uint2 *cb = reinterpret_cast<const uint2 *>(qcodes) +
                    (size_t)tile * G * 32 + lane;
  const uint8_t *sb = qscales + (size_t)tile * G * 32 + lane;
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.f;

  // Chunked x staging: whole-x doesn't fit smem at M=3 x K=17408.
  // Accumulators persist across chunks; each warp consumes only its
  // K-quarter's groups that fall inside the staged window.
  for (int c0 = 0; c0 < (K >> 1); c0 += KCH2) {
    const int clen = min(KCH2, (K >> 1) - c0);
    __syncthreads();
    for (int t = threadIdx.x; t < M * clen; t += blockDim.x) {
      const int m = t / clen, j = t % clen;
      xs[m * KCH2 + j] =
          reinterpret_cast<const half2 *>(x)[(size_t)m * (K >> 1) + c0 + j];
    }
    __syncthreads();
    const int ga = max(g0, c0 >> 3);
    const int gb = min(g0 + Gq, (c0 + clen) >> 3);
#pragma unroll 4
    for (int g = ga; g < gb; g++) {
      const uint2 q2 = __ldcs(cb + (size_t)g * 32);
      const half2 sc2 =
          __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
      half2 w[8];
      dequant8_tm(q2.x, sc2, w + 0);  // adjacent-k pairs k0..7
      dequant8_tm(q2.y, sc2, w + 4);  // k8..15
#pragma unroll
      for (int m = 0; m < M; m++) {
        const half2 *xg = xs + (size_t)m * KCH2 + (g * 8 - c0);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const float2 wf = __half22float2(w[j]);
          const float2 xf = __half22float2(xg[j]);
          s += wf.x * xf.x + wf.y * xf.y;
        }
        acc[m] += s;
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M; m++)
    cs[(warp * 32 + lane) * M + m] = acc[m];
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int m = 0; m < M; m++) {
      float v = cs[lane * M + m] + cs[(32 + lane) * M + m] +
                cs[(64 + lane) * M + m] + cs[(96 + lane) * M + m];
      y[(size_t)m * N + (size_t)tile * 32 + lane] = __float2half(v);
    }
  }
}

torch::Tensor skinny_gemm_qpn_simt(torch::Tensor x, torch::Tensor qcodes,
                                   torch::Tensor qscales, double gscale,
                                   int64_t n) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 3, "qpn_simt supports M 1..3, got ", m);
  TORCH_CHECK(k % 64 == 0, "K % 64");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * (k >> 1) &&
              qscales.numel() == n * (k >> 4));
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (int)(m * 4096 * sizeof(half2)) +
                   4 * 32 * (int)m * (int)sizeof(float);  // KCH2 chunks

#define LAUNCH_QPN_SIMT(MM)                                                 \
  do {                                                                     \
    auto *fn = skinny_nvfp4_qpn_simt<MM>;                                  \
    if (smem > 48 * 1024)                                                  \
      cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,\
                           smem);                                          \
    fn<<<dim3((int)(n / 32)), dim3(128), smem, stream>>>(                  \
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),           \
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),            \
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,  \
        (float)gscale);                                                    \
  } while (0)

  switch (m) {
    case 1: LAUNCH_QPN_SIMT(1); break;
    case 2: LAUNCH_QPN_SIMT(2); break;
    case 3: LAUNCH_QPN_SIMT(3); break;
  }
#undef LAUNCH_QPN_SIMT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_qpn(torch::Tensor x, torch::Tensor qcodes,
                              torch::Tensor qscales, double gscale,
                              int64_t n) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 16, "qpn supports M 1..16, got ", m);
  TORCH_CHECK(k % 64 == 0, "K % 64 (4-warp split of 16-k groups)");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * (k >> 1), "qpn codes size");
  TORCH_CHECK(qscales.numel() == n * (k >> 4), "qpn scales size");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (m <= 8)
    skinny_nvfp4_qpn<1><<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,
        (int)m, (float)gscale);
  else
    skinny_nvfp4_qpn<2><<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,
        (int)m, (float)gscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


// ---------------------------------------------------------------------------
// Activation block-pack for the QP-N kernels (2026-09-09).
//
// The m8n8k4 A-fragment map gives lane L the row (L&3)+((L&16)?4:0), so a
// warp needs EIGHT activation rows per 16-k group. Read straight out of
// x[M][K] those rows sit K*2 bytes apart: eight separate 128B lines for 256
// distinct bytes, against ONE contiguous run for the weights beside them.
// The L1 pays per line touched, not per byte wanted, so the activation side
// costs more than the weights it multiplies -- and the cost grows with M,
// which is the whole M=1 -> M=8 slope.
//
// Established 2026-09-09 by building this file twice and collapsing the
// eight row pointers onto row 0 in one copy, so the warp hits one line
// instead of eight (result deliberately wrong; only the time is read). At
// M=8 that alone was worth 1.30x-2.00x on the RTX 8000 and 1.08x-1.19x on
// the V100; at M=1, where the eight rows are one row anyway, it changed
// nothing (1.00x) -- the control that says the effect is the row scatter and
// not the probe. Before that, two other explanations were measured and
// dropped: the MMA shape (benchmarks/mma_probe.py -- m8n8k4 and m16n8k8 are
// equally fast on Turing) and L1 capacity (the SMALLEST activation block
// collapses hardest, which is backwards for a capacity effect).
// The standing A/B is benchmarks/qpn2_pack_ab.py.
//
// So: pack x into xb[K/16][8][16], one contiguous 256B block per k-group
// holding all eight rows. The warp then touches TWO lines per group instead
// of eight. Rows >= M are zero-filled so the block is always eight rows
// wide, which keeps the buffer size and every address in the main loop
// independent of M.
//
// Arithmetic is untouched -- same values, same order, same registers, same
// fp32 accumulation, same reduction. Results are bit-identical to the
// unpacked path; that is the acceptance test, not a tolerance, and it holds
// end to end: the 27B DFlash2 answer hashes to 0106659946c064b1 packed and
// unpacked, on both card types.
//
// Cost is K*16 bytes written and read once per GEMM (80 KB at K=5120)
// against 4-25 MB of weights per call, plus one kernel launch -- which is
// what qpn2_pack_min_m() below has to earn back.
// ---------------------------------------------------------------------------
__global__ void skinny_pack_x8(const half *__restrict__ x,
                               half *__restrict__ xb, int K, int M) {
  const int total = (K >> 4) * 8 * 8;  // groups x 8 rows x 8 half2
  half2 *dst = reinterpret_cast<half2 *>(xb);
  const half2 *src = reinterpret_cast<const half2 *>(x);
  const int stride = gridDim.x * blockDim.x;
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    const int j = t & 7;         // half2 inside this row's 16-k slice
    const int r = (t >> 3) & 7;  // activation row
    const int g = t >> 6;        // k group
    half2 v = __float2half2_rn(0.f);
    if (r < M) v = src[(size_t)r * (K >> 1) + (size_t)g * 8 + j];
    dst[t] = v;  // linear in t: the write side is fully coalesced
  }
}

// Single call site for the pack, shared by the QP-N wrappers below and by
// the diagnostic binding, so the buffer shape and launch geometry are
// stated once.
static torch::Tensor pack_x8(const torch::Tensor &x, int64_t k, int64_t m,
                             cudaStream_t stream) {
  auto xb = torch::empty({k * 8}, x.options());
  const int total = (int)(k >> 4) * 64;  // groups x 8 rows x 8 half2
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  skinny_pack_x8<<<dim3(blocks), dim3(threads), 0, stream>>>(
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
      reinterpret_cast<half *>(xb.data_ptr<at::Half>()), (int)k, (int)m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return xb;
}

torch::Tensor skinny_pack_x8_op(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(x.size(1) % 16 == 0, "K % 16");
  return pack_x8(x, x.size(1), x.size(0), at::cuda::getCurrentCUDAStream());
}

// Smallest M at which the pack earns its own launch. Measured 2026-09-09
// under CUDA-graph replay -- the serving regime -- on the seven shapes the
// shim dispatches, packed against unpacked:
//
//   sm75 (RTX 8000, 96 KB unified L1): packed wins from M=5 (1.02x-1.26x)
//     up to M=8 (1.29x-1.62x). Below M=5 the launch costs 4-10%.
//   sm70 (V100, 128 KB L1): the larger L1 absorbs the row scatter, so the
//     pack only pays at M=8 (trunk aggregate 1.04x) and loses below it.
//
// The two numbers differ because the two L1s do, not because the code
// forks: one kernel, one layout, one dispatch point, one threshold. Asked
// of the WORKER's device, never device 0.
static int qpn2_pack_min_m() {
  const auto *prop = at::cuda::getCurrentDeviceProperties();
  return (prop->major == 7 && prop->minor == 5) ? 5 : 8;
}

// ---------------------------------------------------------------------------
// QPN2 (2026-08-17): the qpn_matrix/qpn_msweep geometry winner. Same QP-N
// architecture and prepacked fragment layout as skinny_nvfp4_qpn, with
// SPLITK (warps per CTA splitting K on one N=32 tile) and NACC
// (independent accumulator fragments across the four k-slice mma.sync
// ops) as template knobs. Measured frontier (results/
// qpn_matrix_20260817.csv): split16 for N*K small/mid shapes, split8 for
// gate_up, split32 for N=2048 — weighted 637 GB/s at M=8 vs 441 for the
// fixed-4-warp kernel; near-flat in M (704 GB/s weighted at M=1).
// M <= 8 only; M 9..16 stays on skinny_nvfp4_qpn<2>.
// PACKED selects the activation layout the SAME kernel reads: the
// block-packed xb[k/16][8][16] from skinny_pack_x8, or plain row-major x.
// It is a template parameter, not a pair of stride arguments: passing the
// strides at runtime turned the group stride from a shift into an IMAD in
// the innermost loop and cost 1-5% on the V100 (measured 2026-09-09).
template <int SPLITK, int NACC, bool PACKED>
__global__ void skinny_nvfp4_qpn2(const uint8_t *__restrict__ bcodes,
                                  const uint8_t *__restrict__ bscales,
                                  const half *__restrict__ xsrc,
                                  half *__restrict__ y, int N, int K, int M,
                                  float gscale) {
  __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const uint2 *cb = reinterpret_cast<const uint2 *>(bcodes) +
                    (size_t)tile * G * 32 + lane;
  const uint8_t *sb = bscales + (size_t)tile * G * 32 + lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[a][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const half2 sc2 =
        __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
    half2 b[8];
    dequant8_tm(q2.x, sc2, b + 0);
    dequant8_tm(q2.y, sc2, b + 4);
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
    // The `r < M` guard STAYS. Rows >= M are zeroed in the packed block, so
    // dropping it would be correct -- but at low M it is what keeps the warp
    // on one line instead of reading the whole 256B block, and at M=8 every
    // lane loads anyway, so it costs nothing where the block-pack pays.
    // (Measured 2026-09-09: unconditional cost 0.56x-0.67x at M<=4.)
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (r < M) {
      const half *xrow = PACKED ? xsrc + (size_t)g * 128 + r * 16
                                : xsrc + (size_t)r * K + g * 16;
      a01 = *reinterpret_cast<const uint4 *>(xrow);
      a23 = *reinterpret_cast<const uint4 *>(xrow + 8);
    }
    const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }

#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[0][i] += c[a][i];

  if (SPLITK == 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < M)
        y[(size_t)row * N + (size_t)tile * 32 + qp * 8 + col] =
            __float2half(c[0][i]);
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row * 32 + qp * 8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) v += cs[w][e];
    const int row = e >> 5, col = e & 31;
    if (row < M)
      y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

torch::Tensor skinny_gemm_qpn2(torch::Tensor x, torch::Tensor qcodes,
                               torch::Tensor qscales, double gscale,
                               int64_t n, int64_t splitk, int64_t nacc) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 8, "qpn2 supports M 1..8, got ", m);
  TORCH_CHECK(k % 64 == 0 && (k / 16) % splitk == 0, "K/SPLITK");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * (k >> 1), "qpn codes size");
  TORCH_CHECK(qscales.numel() == n * (k >> 4), "qpn scales size");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Block-pack the activations where it pays (see qpn2_pack_min_m): the
  // packed buffer is xb[k/16][8][16], one contiguous 256B block per k-group.
  // Below the threshold the kernel reads x itself with row-major strides --
  // same kernel, same result, one launch less.
  const bool packed = (m >= qpn2_pack_min_m());
  auto xsrc = packed ? pack_x8(x, k, m, stream) : x;

#define LAUNCH_QPN2_T(SPv, NAv, PKv)                                          skinny_nvfp4_qpn2<SPv, NAv, PKv>                                                <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                           qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),                    reinterpret_cast<const half *>(xsrc.data_ptr<at::Half>()),                  reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n,                   (int)k, (int)m, (float)gscale)

#define LAUNCH_QPN2(SPv, NAv)                                                 do {                                                                            if (packed) LAUNCH_QPN2_T(SPv, NAv, true);                                    else LAUNCH_QPN2_T(SPv, NAv, false);                                      } while (0)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 81: LAUNCH_QPN2(8, 1); break;
    case 82: LAUNCH_QPN2(8, 2); break;
    case 161: LAUNCH_QPN2(16, 1); break;
    case 162: LAUNCH_QPN2(16, 2); break;
    case 321: LAUNCH_QPN2(32, 1); break;
    case 322: LAUNCH_QPN2(32, 2); break;
    default: TORCH_CHECK(false, "qpn2 splitk in {8,16,32}, nacc in {1,2}");
  }
#undef LAUNCH_QPN2
#undef LAUNCH_QPN2_T
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


// ---------------------------------------------------------------------------
// QPN8: FP8 (E4M3) weight codec feeding the identical QPN2 dataflow.
//
// Same quadpair mapping, same N ownership, same split-K, same accumulators,
// same m8n8k4, same reduction. Only the storage width (1 byte/weight instead
// of a nibble) and the decoder change.
//
// The shift trick below is the one already used for NVFP4's group scales,
// exhaustively verified against PyTorch on all 256 byte patterns: exact for
// every finite E4M3 value including all 14 denormals (only the two NaN
// encodings map to +-480, which weights never contain). It reproduces
// value/256; that factor is folded into the per-tile epilogue scale, so the
// K loop carries no multiply at all -- strictly less work than dequant8_tm.
//
// Pairing mirrors dequant8_tm's (i, i+4) interleave so the SAME prepack
// permutation (korder) serves both codecs.
DEV_INLINE void fp8x8_to_half2x4(const uint2 q, half2 out[4]) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const unsigned b0 = (q.x >> (8 * i)) & 0xFFu;
    const unsigned b1 = (q.y >> (8 * i)) & 0xFFu;
    const unsigned h0 = ((b0 & 0x80u) << 8) | ((b0 & 0x7Fu) << 7);
    const unsigned h1 = ((b1 & 0x80u) << 8) | ((b1 & 0x7Fu) << 7);
    const unsigned p = h0 | (h1 << 16);
    out[i] = *reinterpret_cast<const half2 *>(&p);
  }
}

// Word-parallel E4M3 decode, mirroring dequant8_tm's style: one PRMT spreads
// two bytes into 0x00b1_00b0, then a single shift/mask pair per field builds
// both fp16 lanes at once. 6 ops per half2 vs 14 for the per-byte loop.
// Requires NATURAL k order in the packed bytes (adjacent-k pairs), i.e. the
// prepack must use identity korder rather than the (i, i+4) interleave.
// Word-parallel e4m3 -> fp16 decode.
//
// Two things this has to get right, and the original got both wrong:
//
//  1. e4m3 is S EEEE MMM, so exp+mantissa is SEVEN bits and occupies fp16
//     bits 7..13 after the <<7 (the resulting 2^-8 is folded into the
//     epilogue scale). The mask must stop at bit 13 -- 0x3F80, not 0x7F80.
//     With 0x7F80, bit 14 catches the SIGN, which lands in the fp16
//     exponent: every negative weight overflowed to inf and accumulated
//     to NaN.
//  2. The consumer expects the SAME pairing the scalar decoder produces,
//     out[i] = (q.x byte i, q.y byte i) -- the (i, i+4) interleave that
//     cancels the prepack's korder. Pairing sequentially within a word
//     instead ((x0,x1),(x2,x3),...) decodes every value correctly and
//     still gives a garbage GEMM.
//
// Only bits 0..7 and 16..23 of the permuted word are read (the shifts and
// masks ignore the rest), so the two unused byte lanes need no zeroing.
DEV_INLINE void fp8x8_to_half2x4_fast(const uint2 q, half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x3F803F80u;
  unsigned p[4];
  p[0] = __byte_perm(q.x, q.y, 0x0400);   // x0 at b0, y0 at b2
  p[1] = __byte_perm(q.x, q.y, 0x0501);
  p[2] = __byte_perm(q.x, q.y, 0x0602);
  p[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const unsigned v = ((p[i] << 8) & S) | ((p[i] << 7) & EM);
    out[i] = *reinterpret_cast<const half2 *>(&v);
  }
}

// BLOCKED variant: block-scaled FP8 checkpoints ([BN, BK] scale raster,
// typically 128x128) carry a K-VARYING scale, so the epilogue trick of the
// per-tile path no longer applies. The scale moves to the decode point as a
// half2 multiply per 16-k group -- the exact pattern the NVFP4 codec uses,
// just coarser. Two geometry facts keep it cheap and warp-uniform:
//   * an N=32 tile always sits inside ONE BN-row-block (BN % 32 == 0), so
//     the scale per (tile, k-block) is a scalar, no lane raster needed;
//   * a 16-k group always sits inside ONE BK-column-block (BK % 16 == 0),
//     so it is one broadcast __ldg per group (4 B against 512 B of codes).
// tscale is then the raster [ceil(N/BN)][kblocks] fp32, and the decoded
// weights sit at their TRUE magnitude in the mma (the 2^8 decoder factor is
// folded into sc2), so the epilogue writes the fp32 sum unscaled.
// NOT block-packed, unlike skinny_nvfp4_qpn2. Measured 2026-09-09: FP8
// reads twice the bytes per weight, so this kernel is far more DRAM-bound
// and already sat at 81-90% of the read ceiling on both cards. Packing the
// activations bought 1.05x on the RTX 8000 and COST 4% on the V100 -- the
// launch is not paid for. The row scatter is real here too, it is just not
// what limits this kernel.
template <int SPLITK, int NACC, bool FASTDEC = false, bool BLOCKED = false>
__global__ void skinny_fp8_qpn8(const uint8_t *__restrict__ bcodes,
                                const float *__restrict__ tscale,
                                const half *__restrict__ x,
                                half *__restrict__ y, int N, int K, int M,
                                int kb_groups, int nb_tiles, int kblocks) {
  __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  // 16 bytes per lane per group (vs 8 for FP4): one 128-bit load.
  const uint4 *cb = reinterpret_cast<const uint4 *>(bcodes) +
                    (size_t)tile * G * 32 + lane;
  // CTA-uniform: slice boundaries are 64-aligned so a tile never straddles.
  const float ws = BLOCKED ? 1.f : __ldg(tscale + tile);
  const int row_off = BLOCKED ? (tile / nb_tiles) * kblocks : 0;

  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[a][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint4 q4 = __ldcs(cb + (size_t)g * 32);
    half2 b[8];
    if (FASTDEC) {
      fp8x8_to_half2x4_fast(make_uint2(q4.x, q4.y), b + 0);
      fp8x8_to_half2x4_fast(make_uint2(q4.z, q4.w), b + 4);
    } else {
      fp8x8_to_half2x4(make_uint2(q4.x, q4.y), b + 0);
      fp8x8_to_half2x4(make_uint2(q4.z, q4.w), b + 4);
    }
    if (BLOCKED) {
      const half2 sc2 = __float2half2_rn(
          __ldg(tscale + row_off + g / kb_groups) * 256.f);
#pragma unroll
      for (int i = 0; i < 8; i++) b[i] = __hmul2(b[i], sc2);
    }
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (r < M) {
      const half *xrow = x + (size_t)r * K;
      a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
      a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
    }
    const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }

#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[0][i] += c[a][i];

  if (SPLITK == 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < M)
        y[(size_t)row * N + (size_t)tile * 32 + qp * 8 + col] =
            __float2half(c[0][i] * ws);
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row * 32 + qp * 8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) v += cs[w][e];
    const int row = e >> 5, col = e & 31;
    if (row < M)
      y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v * ws);
  }
}

// ---- MT=2: two m8n8k4 row-tiles against ONE weight stream ----------------
// m8n8k4 issues 8 rows per tile, so M=9..16 needs two of them. The chunked
// path runs the whole kernel twice and therefore streams the weights TWICE --
// that, not the tile quantisation, is what makes the k<=15 verify band cost
// ~2x the k<=7 band (298 vs 431 GB/s on the production dispatch curve).
//
// The kernel is DRAM-bound at M=8 (741 GB/s against ~800 GB/s achievable
// read, and faster than a pure torch streaming read of the same bytes), so a
// second row-tile that adds no weight traffic should ride the existing
// stream. This variant loads the B fragment once and issues both row-tiles
// against it.
//
// Costs: a second accumulator set (2*NACC*8 floats/lane) and a doubled
// split-K staging buffer (SPLITK*512 floats = 32 KB at SPLITK=16, inside the
// 48 KB static limit -- which is why SPLITK=32 is not instantiated here).
template <int SPLITK, int NACC, bool FASTDEC = false, bool BLOCKED = false>
__global__ void skinny_fp8_qpn8_mt2(const uint8_t *__restrict__ bcodes,
                                    const float *__restrict__ tscale,
                                    const half *__restrict__ x,
                                    half *__restrict__ y, int N, int K,
                                    int M, int kb_groups, int nb_tiles,
                                    int kblocks) {
  __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 512 : 1];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const uint4 *cb = reinterpret_cast<const uint4 *>(bcodes) +
                    (size_t)tile * G * 32 + lane;
  const float ws = BLOCKED ? 1.f : __ldg(tscale + tile);
  const int row_off = BLOCKED ? (tile / nb_tiles) * kblocks : 0;

  float c[2][NACC][8];
#pragma unroll
  for (int t = 0; t < 2; t++)
#pragma unroll
    for (int a = 0; a < NACC; a++)
#pragma unroll
      for (int i = 0; i < 8; i++) c[t][a][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint4 q4 = __ldcs(cb + (size_t)g * 32);
    half2 b[8];
    if (FASTDEC) {
      fp8x8_to_half2x4_fast(make_uint2(q4.x, q4.y), b + 0);
      fp8x8_to_half2x4_fast(make_uint2(q4.z, q4.w), b + 4);
    } else {
      fp8x8_to_half2x4(make_uint2(q4.x, q4.y), b + 0);
      fp8x8_to_half2x4(make_uint2(q4.z, q4.w), b + 4);
    }
    if (BLOCKED) {
      const half2 sc2 = __float2half2_rn(
          __ldg(tscale + row_off + g / kb_groups) * 256.f);
#pragma unroll
      for (int i = 0; i < 8; i++) b[i] = __hmul2(b[i], sc2);
    }
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
#pragma unroll
    for (int t = 0; t < 2; t++) {
      const int rr = r + (t << 3);
      uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
      if (rr < M) {
        const half *xrow = x + (size_t)rr * K;
        a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
        a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
      }
      const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
      const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
      MMA_8N8K4(c[t][0], A0[0], A0[1], B[0], B[1]);
      MMA_8N8K4(c[t][1 % NACC], A0[2], A0[3], B[2], B[3]);
      MMA_8N8K4(c[t][2 % NACC], A1[0], A1[1], B[4], B[5]);
      MMA_8N8K4(c[t][3 % NACC], A1[2], A1[3], B[6], B[7]);
    }
  }

#pragma unroll
  for (int t = 0; t < 2; t++)
#pragma unroll
    for (int a = 1; a < NACC; a++)
#pragma unroll
      for (int i = 0; i < 8; i++) c[t][0][i] += c[t][a][i];

  if (SPLITK == 1) {
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int row =
            ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1)) + (t << 3);
        const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        if (row < M)
          y[(size_t)row * N + (size_t)tile * 32 + qp * 8 + col] =
              __float2half(c[t][0][i] * ws);
      }
    return;
  }

#pragma unroll
  for (int t = 0; t < 2; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp][(t << 8) + row * 32 + qp * 8 + col] = c[t][0][i];
    }
  __syncthreads();
  for (int e = threadIdx.x; e < 512; e += blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) v += cs[w][e];
    const int t = e >> 8, rem = e & 255;
    const int row = (rem >> 5) + (t << 3), col = rem & 31;
    if (row < M)
      y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v * ws);
  }
}

torch::Tensor skinny_gemm_qpn8_mt2(torch::Tensor x, torch::Tensor qcodes,
                                   torch::Tensor tscale, int64_t n,
                                   int64_t splitk, int64_t nacc) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(tscale.is_cuda() && tscale.dtype() == torch::kFloat &&
              tscale.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 16, "qpn8_mt2 supports M 1..16, got ", m);
  TORCH_CHECK(k % 64 == 0 && (k / 16) % splitk == 0, "K/SPLITK");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * k, "qpn8 codes size");
  TORCH_CHECK(tscale.numel() == n / 32, "qpn8 needs one scale per N=32 tile");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_MT2_F(SPv, NAv)                                                \
  skinny_fp8_qpn8_mt2<SPv, NAv, true>                                         \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), tscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, 0, 0, 0)

#define LAUNCH_MT2(SPv, NAv)                                                  \
  skinny_fp8_qpn8_mt2<SPv, NAv>                                               \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), tscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, 0, 0, 0)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 43: LAUNCH_MT2_F(4, 1); break;
    case 44: LAUNCH_MT2_F(4, 2); break;
    case 83: LAUNCH_MT2_F(8, 1); break;
    case 84: LAUNCH_MT2_F(8, 2); break;
    case 163: LAUNCH_MT2_F(16, 1); break;
    case 164: LAUNCH_MT2_F(16, 2); break;
    case 41: LAUNCH_MT2(4, 1); break;
    case 42: LAUNCH_MT2(4, 2); break;
    case 81: LAUNCH_MT2(8, 1); break;
    case 82: LAUNCH_MT2(8, 2); break;
    case 161: LAUNCH_MT2(16, 1); break;
    case 162: LAUNCH_MT2(16, 2); break;
    default:
      TORCH_CHECK(false,
                  "qpn8_mt2 splitk in {4,8,16} (32 would need 64 KB of "
                  "shared), nacc in {1,2} (+2 selects the fast decoder)");
  }
#undef LAUNCH_MT2_F
#undef LAUNCH_MT2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_qpn8(torch::Tensor x, torch::Tensor qcodes,
                               torch::Tensor tscale, int64_t n,
                               int64_t splitk, int64_t nacc) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(tscale.is_cuda() && tscale.dtype() == torch::kFloat &&
              tscale.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 8, "qpn8 supports M 1..8, got ", m);
  TORCH_CHECK(k % 64 == 0 && (k / 16) % splitk == 0, "K/SPLITK");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * k, "qpn8 codes size");
  TORCH_CHECK(tscale.numel() == n / 32, "qpn8 needs one scale per N=32 tile");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_QPN8_F(SPv, NAv)                                               \
  skinny_fp8_qpn8<SPv, NAv, true>                                             \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), tscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, 0, 0, 0)

#define LAUNCH_QPN8(SPv, NAv)                                                 \
  skinny_fp8_qpn8<SPv, NAv>                                                   \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), tscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, 0, 0, 0)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 43: LAUNCH_QPN8_F(4, 1); break;
    case 83: LAUNCH_QPN8_F(8, 1); break;
    case 84: LAUNCH_QPN8_F(8, 2); break;
    case 163: LAUNCH_QPN8_F(16, 1); break;
    case 164: LAUNCH_QPN8_F(16, 2); break;
    case 323: LAUNCH_QPN8_F(32, 1); break;
    case 324: LAUNCH_QPN8_F(32, 2); break;
    case 41: LAUNCH_QPN8(4, 1); break;
    case 42: LAUNCH_QPN8(4, 2); break;
    case 81: LAUNCH_QPN8(8, 1); break;
    case 82: LAUNCH_QPN8(8, 2); break;
    case 161: LAUNCH_QPN8(16, 1); break;
    case 162: LAUNCH_QPN8(16, 2); break;
    case 321: LAUNCH_QPN8(32, 1); break;
    case 322: LAUNCH_QPN8(32, 2); break;
    default: TORCH_CHECK(false, "qpn8 splitk in {4,8,16,32}, nacc in {1,2} (+2 selects the fast decoder)");
  }
#undef LAUNCH_QPN8
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Block-scaled entries: bscale is the checkpoint's fp32 scale raster
// [ceil(N/bn)][ceil(K/bk)] (weight_block_size = [bn, bk], 128x128 for the
// DeepSeek-class checkpoints). Same prepacked code layout, same dispatch
// keys as the per-tile path; only the scale transport differs (see the
// BLOCKED comment on skinny_fp8_qpn8).
static void check_blk_scale(const torch::Tensor &bscale, int64_t n, int64_t k,
                            int64_t bn, int64_t bk) {
  TORCH_CHECK(bscale.is_cuda() && bscale.dtype() == torch::kFloat &&
              bscale.is_contiguous());
  TORCH_CHECK(bn % 32 == 0, "bn must be a multiple of 32 (tile rows)");
  TORCH_CHECK(bk % 16 == 0, "bk must be a multiple of 16 (group depth)");
  TORCH_CHECK(bscale.dim() == 2 && bscale.size(0) == (n + bn - 1) / bn &&
                  bscale.size(1) == (k + bk - 1) / bk,
              "bscale must be [ceil(N/bn), ceil(K/bk)], got ",
              bscale.size(0), "x", bscale.size(1));
}

// ---------------------------------------------------------------------------
// WMMA-BLK: tensor-core prefill kernel reading the QPN8-PACKED layout.
//
// The decode band (M<=16) is served by the qpn8/mt2 register kernels above;
// their chunked/reconstruct extensions lose 2-10x to a tiled kernel once M
// grows past ~16 (fp8_blk_backend_bench). This kernel closes that band from
// the SAME resident buffer -- no second weight format: a CTA owns one packed
// 32-column tile, stages KC-deep dequantized weights to smem straight from
// fragment order (lane -> column via the QPN8 col map; korder makes the fast
// decoder emit adjacent-k half2 pairs, so smem writes are linear), and feeds
// nvcuda::wmma 16x16x16 fragments exactly like skinny_nvfp4_wmma. Block
// scales (raster [ceil(N/bn)][ceil(K/bk)]) are applied at the decode point;
// an N=32 tile never straddles a BN block, so the lookup is lane-uniform.
// Grid: (N/32, ceil(M/MT)) -- full K per CTA, fp32 accumulate.
// ---------------------------------------------------------------------------
template <int WM, int KC>
__global__ void skinny_fp8_wmma_blk(const uint8_t *__restrict__ bcodes,
                                    const float *__restrict__ bscale,
                                    const half *__restrict__ x,
                                    half *__restrict__ y, int N, int K,
                                    int m_real, int kb_groups, int nb_tiles,
                                    int kblocks) {
  constexpr int WN = 2, NT = 32, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;  // packed uint4 per thread
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;   // x uint4 per thread
  static_assert(CSEG * NTHREADS == NT * (KC / 16), "code seg split");
  static_assert(XSEG * NTHREADS == MT * (KC / 8), "x seg split");

  extern __shared__ char smem_raw[];
  half *ws = reinterpret_cast<half *>(smem_raw);  // [NT][PW]
  half *xs = ws + NT * PW;                        // [MT][PX]

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int tile = blockIdx.x;
  const int m0 = blockIdx.y * MT;
  const int G = K >> 4;
  const uint4 *cb = reinterpret_cast<const uint4 *>(bcodes) +
                    (size_t)tile * G * 32;
  const int row_off = (tile / nb_tiles) * kblocks;

  uint4 st_c[CSEG];
  float st_s[CSEG];
  uint4 st_x[XSEG];

  auto load_stage = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int gg = idx >> 5, lp = idx & 31;
      const int g = (k0 >> 4) + gg;
      st_c[i] = __ldcs(cb + (size_t)g * 32 + lp);
      st_s[i] = __ldg(bscale + row_off + g / kb_groups) * 256.f;
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = m0 + idx / (KC / 8), j4 = idx % (KC / 8);
      st_x[i] = (m < m_real)
                    ? *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 +
                                                       j4 * 8)
                    : make_uint4(0, 0, 0, 0);
    }
  };

  auto store_stage = [&]() {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int gg = idx >> 5, lp = idx & 31;
      // QPN8 col map: which output column this lane's bytes belong to.
      const int col = ((lp >> 2) & 3) * 8 + (lp & 3) + ((lp & 16) ? 4 : 0);
      const half2 sc2 = __float2half2_rn(st_s[i]);
      half2 b[8];
      fp8x8_to_half2x4_fast(make_uint2(st_c[i].x, st_c[i].y), b + 0);
      fp8x8_to_half2x4_fast(make_uint2(st_c[i].z, st_c[i].w), b + 4);
      half2 *wrow = reinterpret_cast<half2 *>(ws + col * PW + gg * 16);
#pragma unroll
      for (int j = 0; j < 8; j++) wrow[j] = __hmul2(b[j], sc2);
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4 *>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::fill_fragment(cfrag, 0.f);

  load_stage(0);
  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();
    store_stage();
    __syncthreads();
    if (k0 + KC < K) load_stage(k0 + KC);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
    wmma::load_matrix_sync(a[0], ws + wn * 16 * PW, PW);
    wmma::load_matrix_sync(b[0], xs + wm * 16 * PX, PX);
#pragma unroll
    for (int kk = 0; kk < KC / 16; kk++) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < KC / 16) {
        wmma::load_matrix_sync(a[nxt], ws + wn * 16 * PW + (kk + 1) * 16, PW);
        wmma::load_matrix_sync(b[nxt], xs + wm * 16 * PX + (kk + 1) * 16, PX);
      }
      wmma::mma_sync(cfrag, a[cur], b[cur], cfrag);
    }
  }

  __syncthreads();  // reuse smem for the fp32 epilogue stage
  float *cs = reinterpret_cast<float *>(smem_raw) + warp * 256;
  wmma::store_matrix_sync(cs, cfrag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;  // i: n within warp tile, j: m
    const int gm = m0 + wm * 16 + j;
    const int gn = tile * 32 + wn * 16 + i;
    if (gm < m_real) y[(size_t)gm * N + gn] = __float2half(cs[e]);
  }
}

torch::Tensor skinny_gemm_qpn8_blk_wmma(torch::Tensor x, torch::Tensor qcodes,
                                        torch::Tensor bscale, int64_t n,
                                        int64_t bn, int64_t bk) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(k % 128 == 0, "K must be a multiple of 128");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * k, "qpn8 codes size");
  check_blk_scale(bscale, n, k, bn, bk);
  const int kbg = (int)(bk / 16), nbt = (int)(bn / 32);
  const int kbl = (int)bscale.size(1);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_WMMA_BLK(WM)                                                   \
  do {                                                                        \
    constexpr int KC = 128, NT = 32, MT = WM * 16;                            \
    const int smem = (NT + MT) * (KC + 16) * (int)sizeof(half);               \
    const dim3 grid((int)(n / 32), (int)((m + MT - 1) / MT));                 \
    skinny_fp8_wmma_blk<WM, KC><<<grid, dim3(2 * WM * 32), smem, stream>>>(   \
        qcodes.data_ptr<uint8_t>(), bscale.data_ptr<float>(),                 \
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),               \
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,     \
        (int)m, kbg, nbt, kbl);                                               \
  } while (0)

  if (m <= 32) LAUNCH_WMMA_BLK(2);
  else LAUNCH_WMMA_BLK(4);
#undef LAUNCH_WMMA_BLK
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Fast transient dequant for the large-M band: packed QPN8 bytes + scale
// raster -> fp16 [N, K] (checkpoint row order), one 16-byte read and one
// 32-byte write per thread-group entry. The caller feeds the result to
// cuBLAS hgemm; at prefill M the GEMM dominates and the pair matches a
// dedicated tiled kernel without keeping a second weight format resident
// (fp8_blk_backend_bench: torch-indexing unpack made reconstruct 2-10x
// slower than TurboMind; this kernel removes that overhead).
__global__ void skinny_fp8_dequant_blk(const uint8_t *__restrict__ bcodes,
                                       const float *__restrict__ bscale,
                                       half *__restrict__ out, int N, int K,
                                       int kb_groups, int nb_tiles,
                                       int kblocks) {
  const int G = K >> 4;
  const long long total = (long long)(N >> 5) * G * 32;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int lane = (int)(idx & 31);
  const long long tg = idx >> 5;
  const int g = (int)(tg % G);
  const int tile = (int)(tg / G);
  const int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const uint4 q4 = __ldcs(reinterpret_cast<const uint4 *>(bcodes) +
                          ((size_t)tile * G + g) * 32 + lane);
  const float ws =
      __ldg(bscale + (tile / nb_tiles) * kblocks + g / kb_groups);
  const half2 sc2 = __float2half2_rn(ws * 256.f);
  half2 b[8];
  fp8x8_to_half2x4_fast(make_uint2(q4.x, q4.y), b + 0);
  fp8x8_to_half2x4_fast(make_uint2(q4.z, q4.w), b + 4);
#pragma unroll
  for (int j = 0; j < 8; j++) b[j] = __hmul2(b[j], sc2);
  half2 *orow = reinterpret_cast<half2 *>(
      out + (size_t)(tile * 32 + col) * K + g * 16);
#pragma unroll
  for (int j = 0; j < 8; j++) orow[j] = b[j];
}

torch::Tensor skinny_qpn8_blk_dequant(torch::Tensor qcodes,
                                      torch::Tensor bscale, int64_t n,
                                      int64_t k, int64_t bn, int64_t bk) {
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(k % 16 == 0 && n % 32 == 0, "qpn8 geometry");
  TORCH_CHECK(qcodes.numel() == n * k, "qpn8 codes size");
  check_blk_scale(bscale, n, k, bn, bk);
  auto out = torch::empty(
      {n, k}, bscale.options().dtype(torch::kHalf));
  const long long total = (n >> 5) * (k >> 4) * 32;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  skinny_fp8_dequant_blk<<<(unsigned)blocks, threads, 0, stream>>>(
      qcodes.data_ptr<uint8_t>(), bscale.data_ptr<float>(),
      reinterpret_cast<half *>(out.data_ptr<at::Half>()), (int)n, (int)k,
      (int)(bk / 16), (int)(bn / 32), (int)bscale.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor skinny_gemm_qpn8_blk(torch::Tensor x, torch::Tensor qcodes,
                                   torch::Tensor bscale, int64_t n,
                                   int64_t bn, int64_t bk, int64_t splitk,
                                   int64_t nacc) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 8, "qpn8_blk supports M 1..8, got ", m);
  TORCH_CHECK(k % 64 == 0 && (k / 16) % splitk == 0, "K/SPLITK");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * k, "qpn8 codes size");
  check_blk_scale(bscale, n, k, bn, bk);
  const int kbg = (int)(bk / 16), nbt = (int)(bn / 32);
  const int kbl = (int)bscale.size(1);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_QPN8_BLK_F(SPv, NAv)                                           \
  skinny_fp8_qpn8<SPv, NAv, true, true>                                       \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), bscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, kbg, nbt, kbl)

#define LAUNCH_QPN8_BLK(SPv, NAv)                                             \
  skinny_fp8_qpn8<SPv, NAv, false, true>                                      \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), bscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, kbg, nbt, kbl)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 43: LAUNCH_QPN8_BLK_F(4, 1); break;
    case 83: LAUNCH_QPN8_BLK_F(8, 1); break;
    case 84: LAUNCH_QPN8_BLK_F(8, 2); break;
    case 163: LAUNCH_QPN8_BLK_F(16, 1); break;
    case 164: LAUNCH_QPN8_BLK_F(16, 2); break;
    case 323: LAUNCH_QPN8_BLK_F(32, 1); break;
    case 324: LAUNCH_QPN8_BLK_F(32, 2); break;
    case 41: LAUNCH_QPN8_BLK(4, 1); break;
    case 42: LAUNCH_QPN8_BLK(4, 2); break;
    case 81: LAUNCH_QPN8_BLK(8, 1); break;
    case 82: LAUNCH_QPN8_BLK(8, 2); break;
    case 161: LAUNCH_QPN8_BLK(16, 1); break;
    case 162: LAUNCH_QPN8_BLK(16, 2); break;
    case 321: LAUNCH_QPN8_BLK(32, 1); break;
    case 322: LAUNCH_QPN8_BLK(32, 2); break;
    default: TORCH_CHECK(false, "qpn8_blk splitk in {4,8,16,32}, nacc in "
                                "{1,2} (+2 selects the fast decoder)");
  }
#undef LAUNCH_QPN8_BLK_F
#undef LAUNCH_QPN8_BLK
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_qpn8_blk_mt2(torch::Tensor x, torch::Tensor qcodes,
                                       torch::Tensor bscale, int64_t n,
                                       int64_t bn, int64_t bk, int64_t splitk,
                                       int64_t nacc) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 16, "qpn8_blk_mt2 supports M 1..16, got ", m);
  TORCH_CHECK(k % 64 == 0 && (k / 16) % splitk == 0, "K/SPLITK");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * k, "qpn8 codes size");
  check_blk_scale(bscale, n, k, bn, bk);
  const int kbg = (int)(bk / 16), nbt = (int)(bn / 32);
  const int kbl = (int)bscale.size(1);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_MT2_BLK_F(SPv, NAv)                                            \
  skinny_fp8_qpn8_mt2<SPv, NAv, true, true>                                   \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), bscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, kbg, nbt, kbl)

#define LAUNCH_MT2_BLK(SPv, NAv)                                              \
  skinny_fp8_qpn8_mt2<SPv, NAv, false, true>                                  \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          qcodes.data_ptr<uint8_t>(), bscale.data_ptr<float>(),               \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m, kbg, nbt, kbl)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 43: LAUNCH_MT2_BLK_F(4, 1); break;
    case 44: LAUNCH_MT2_BLK_F(4, 2); break;
    case 83: LAUNCH_MT2_BLK_F(8, 1); break;
    case 84: LAUNCH_MT2_BLK_F(8, 2); break;
    case 163: LAUNCH_MT2_BLK_F(16, 1); break;
    case 164: LAUNCH_MT2_BLK_F(16, 2); break;
    case 41: LAUNCH_MT2_BLK(4, 1); break;
    case 42: LAUNCH_MT2_BLK(4, 2); break;
    case 81: LAUNCH_MT2_BLK(8, 1); break;
    case 82: LAUNCH_MT2_BLK(8, 2); break;
    case 161: LAUNCH_MT2_BLK(16, 1); break;
    case 162: LAUNCH_MT2_BLK(16, 2); break;
    default:
      TORCH_CHECK(false,
                  "qpn8_blk_mt2 splitk in {4,8,16} (32 would need 64 KB of "
                  "shared), nacc in {1,2} (+2 selects the fast decoder)");
  }
#undef LAUNCH_MT2_BLK_F
#undef LAUNCH_MT2_BLK
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


// ---------------------------------------------------------------------------
// Grouped MoE SIMT kernel (fork addition, v100-skinny): one launch for all
// experts of a layer. Routing is device-side -- `perm` lists the (token,
// topk) slots sorted by expert, `offsets[e]..offsets[e+1]` is expert e's
// range -- so there is no host sync and the launch is CUDA-graph safe.
// Grid (N/8, E): a block whose expert has no rows exits before touching
// its weights, so only selected experts are read. Rows per expert are
// bounded by the token count (<= 8 on the decode/verify path); the inner
// loop is dispatched block-uniformly on that count so a one-row expert
// pays one row's FMAs, not eight.
// ---------------------------------------------------------------------------
template <int M, int KC>
DEV_INLINE void moe_simt_rows(const uint8_t *__restrict__ crow,
                              const uint8_t *__restrict__ srow,
                              const half *const *__restrict__ xrows,
                              half2 *__restrict__ xs, int cnt, int K,
                              half2 gm2, float *__restrict__ accf) {
  constexpr int P2 = KC / 2;
  const int lane = threadIdx.x & 31;
  int k0 = 0;
  for (; k0 + KC <= K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < cnt * (KC / 8); idx += blockDim.x) {
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      const uint4 v = *reinterpret_cast<const uint4 *>(xrows[m] + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < KC / 512; i++) {
      const int s = lane + 32 * i;
      const uint2 q2 = *reinterpret_cast<const uint2 *>(crow + (k0 >> 1) + s * 8);
      const half2 sc2 = __hmul2(fp8e4m3_to_half2(srow[(k0 >> 4) + s]), gm2);
      half2 acch[M];
#pragma unroll
      for (int m = 0; m < M; m++) acch[m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[4];
        const unsigned qw = w == 0 ? q2.x : q2.y;
#ifndef SKINNY_LUT_CVT
        dequant8_tm(qw, sc2, w4);
#else
#pragma unroll
        for (int pi = 0; pi < 4; pi++) w4[pi] = dequant_pair(qw, pi, sc2);
#endif
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++)
            acch[m] = __hfma2(w4[pi], xs[m * P2 + psw], acch[m]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++) {
        const float2 f = __half22float2(acch[m]);
        accf[m] += f.x + f.y;
      }
    }
  }
  const int tail = K - k0;
  if (tail > 0) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < cnt * (tail / 8); idx += blockDim.x) {
      const int m = idx / (tail / 8), j4 = idx % (tail / 8);
      const uint4 v = *reinterpret_cast<const uint4 *>(xrows[m] + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();
    const int nseg = tail >> 4;
    for (int s = lane; s < nseg; s += 32) {
      const uint2 q2 = *reinterpret_cast<const uint2 *>(crow + (k0 >> 1) + s * 8);
      const half2 sc2 = __hmul2(fp8e4m3_to_half2(srow[(k0 >> 4) + s]), gm2);
      half2 acch[M];
#pragma unroll
      for (int m = 0; m < M; m++) acch[m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[4];
        const unsigned qw = w == 0 ? q2.x : q2.y;
#ifndef SKINNY_LUT_CVT
        dequant8_tm(qw, sc2, w4);
#else
#pragma unroll
        for (int pi = 0; pi < 4; pi++) w4[pi] = dequant_pair(qw, pi, sc2);
#endif
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++)
            acch[m] = __hfma2(w4[pi], xs[m * P2 + psw], acch[m]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++) {
        const float2 f = __half22float2(acch[m]);
        accf[m] += f.x + f.y;
      }
    }
  }
}

template <int KC>
__global__ void skinny_nvfp4_moe_simt(const uint8_t *__restrict__ codes,
                                      const uint8_t *__restrict__ scales,
                                      const float *__restrict__ gscales,
                                      const half *__restrict__ x,
                                      half *__restrict__ y_slots,
                                      const int *__restrict__ perm,
                                      const int *__restrict__ offsets,
                                      int N, int K, int topk,
                                      int x_slot_major) {
  constexpr int MMAX = 8;
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [MMAX][KC/2]
  __shared__ const half *xrows[MMAX];
  __shared__ int slots[MMAX];

  const int e = blockIdx.y;
  const int beg = offsets[e];
  const int cnt = offsets[e + 1] - beg;
  if (cnt <= 0) return;  // block-uniform: inactive expert, no weight read

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + warp;
  const uint8_t *crow = codes + ((size_t)e * N + n) * (K >> 1);
  const uint8_t *srow = scales + ((size_t)e * N + n) * (K >> 4);
#ifndef SKINNY_LUT_CVT
  const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);
#else
  const half2 gm2 = __float2half2_rn(gscales[e]);
#endif

  // Rows per expert are usually <= the token count, but hash routing can
  // hand one expert the same token twice, so an expert may exceed MMAX
  // rows. Process MMAX rows per pass; the rare extra pass re-reads this
  // expert's weights once more.
  for (int base = 0; base < cnt; base += MMAX) {
    const int rows = min(MMAX, cnt - base);
    __syncthreads();  // previous pass done with slots/xrows/xs
    if (threadIdx.x < MMAX) {
      const int m = threadIdx.x;
      const int slot = m < rows ? perm[beg + base + m] : 0;
      slots[m] = slot;
      // x is either token-major [T,K] (w13: gather by token) or
      // slot-major [S,K] (w2: the activation already lives per slot).
      xrows[m] = x + (size_t)(x_slot_major ? slot : slot / topk) * K;
    }
    __syncthreads();

    float accf[MMAX];
#pragma unroll
    for (int m = 0; m < MMAX; m++) accf[m] = 0.f;

    if (rows == 1)      moe_simt_rows<1, KC>(crow, srow, xrows, xs, rows, K, gm2, accf);
    else if (rows == 2) moe_simt_rows<2, KC>(crow, srow, xrows, xs, rows, K, gm2, accf);
    else if (rows <= 4) moe_simt_rows<4, KC>(crow, srow, xrows, xs, rows, K, gm2, accf);
    else                moe_simt_rows<8, KC>(crow, srow, xrows, xs, rows, K, gm2, accf);

#pragma unroll
    for (int m = 0; m < MMAX; m++) {
      float v = accf[m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0 && m < rows) y_slots[(size_t)slots[m] * N + n] = __float2half(v);
    }
  }
}

// y_slots[S, N] (S = tokens*topk, slot-major) = x[token(slot)] @ W[expert(slot)]
void skinny_moe_simt(torch::Tensor x, torch::Tensor codes, torch::Tensor scales,
                     torch::Tensor gscales, torch::Tensor perm,
                     torch::Tensor offsets, int64_t topk, torch::Tensor y_slots,
                     bool x_slot_major, int64_t num_tokens) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 && scales.is_contiguous());
  TORCH_CHECK(gscales.is_cuda() && gscales.dtype() == torch::kFloat && gscales.is_contiguous());
  TORCH_CHECK(perm.is_cuda() && perm.dtype() == torch::kInt && perm.is_contiguous());
  TORCH_CHECK(offsets.is_cuda() && offsets.dtype() == torch::kInt && offsets.is_contiguous());
  TORCH_CHECK(y_slots.is_cuda() && y_slots.dtype() == torch::kHalf && y_slots.is_contiguous());
  const int64_t T = num_tokens, K = x.size(1);
  TORCH_CHECK(x.size(0) == (x_slot_major ? T * topk : T), "x rows mismatch");
  const int64_t E = codes.size(0), N = codes.size(1);
  TORCH_CHECK(codes.size(2) * 2 == K, "codes/x K mismatch");
  TORCH_CHECK(scales.size(0) == E && scales.size(1) == N && scales.size(2) * 16 == K);
  TORCH_CHECK(gscales.size(0) == E && offsets.size(0) == E + 1);
  TORCH_CHECK(perm.size(0) == T * topk && y_slots.size(0) == T * topk && y_slots.size(1) == N);
  TORCH_CHECK(T <= 8, "grouped simt MoE serves <= 8 tokens (decode/verify); got ", T);
  TORCH_CHECK(K % 128 == 0 && K >= 128 && N % 8 == 0);
  constexpr int KC = 1024;
  const dim3 grid((unsigned)(N / 8), (unsigned)E), block(256);
  const int smem = 8 * (KC / 2) * sizeof(half2);
  auto stream = at::cuda::getCurrentCUDAStream();
  skinny_nvfp4_moe_simt<KC><<<grid, block, smem, stream>>>(
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
      gscales.data_ptr<float>(),
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
      reinterpret_cast<half *>(y_slots.data_ptr<at::Half>()),
      perm.data_ptr<int>(), offsets.data_ptr<int>(), (int)N, (int)K, (int)topk,
      x_slot_major ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Grouped MoE QPN kernel (fork addition, v100-skinny): the moe_simt routing
// skeleton (device-side perm/offsets, inactive experts exit before touching
// their weights, CUDA-graph safe) driving the QPN2 tensor-core
// dataflow (mma.m8n8k4 on fragment-order prepacked weights, SPLITK warps
// splitting K on one N=32 tile, NACC independent accumulator fragments).
// Weights must be prepacked per expert with the dense shim's _qpn_prepack
// permutation: [E][tile N/32][group K/16][lane 32] x 8B codes + 1B scales.
// A rows come through the slot indirection (token-major x for w13,
// slot-major for w2) instead of qpn2's contiguous x.
// Routing arrives COMPACT: grid.y spans slot-count-many group slots (a
// static bound, so it captures into CUDA graphs), `gids[grp]` names the
// group's expert and `goff[grp]..goff[grp+1]` its slot range; padding
// groups carry an empty range and exit. This keeps the launch independent
// of the expert count -- with a per-expert grid, many-expert models
// (E=512, top-k 10, T=1) schedule thousands of empty blocks that cost
// more than the actual work.
// ---------------------------------------------------------------------------
// RB row blocks share one weight load: each group's codes are fetched and
// dequantized once and feed the MMAs of up to RB x 8 rows, instead of one
// re-read per 8-row pass. Every row keeps its own accumulators and the same
// K order, so the result does not depend on RB.
template <int SPLITK, int NACC, int SCALE_MODE = SCALE_NVFP4_FP8_16, int RB = 1>
__global__ void __launch_bounds__(32 * SPLITK)
skinny_nvfp4_moe_qpn(const uint8_t *__restrict__ qcodes,
                                     const uint8_t *__restrict__ qscales,
                                     const float *__restrict__ gscales,
                                     const half *__restrict__ x,
                                     half *__restrict__ y_slots,
                                     const int *__restrict__ perm,
                                     const int *__restrict__ gids,
                                     const int *__restrict__ goff,
                                     int N, int K, int topk,
                                     int x_slot_major) {
  constexpr int MMAX = 8;
  constexpr int ROWS = RB * MMAX;
  // Activations are staged per warp, XCH groups at a time: the four
  // quadpairs of a warp read the same A rows, so loading them straight from
  // global memory issued every load four times. Rows are padded to 40 halves
  // (80 B) so the eight rows of a fragment fall into distinct banks.
  constexpr int XCH = 2;
  constexpr int XROW = XCH * 16 + 8;
  __shared__ const half *xrows[ROWS];
  __shared__ int slots[ROWS];
  // The activation stage (used while computing; decode, RB = 1, reads A
  // directly and needs none) and the split-K reduction buffer (used after)
  // share one allocation, which keeps two 512-thread blocks resident on a
  // 64 KiB-shared Turing SM.
  constexpr int XS_BYTES = (RB == 1 ? 1 : SPLITK) * RB * MMAX * XROW * 2;
  constexpr int CS_BYTES = SPLITK * 256 * 4;
  __shared__ __align__(16) unsigned char smem[XS_BYTES > CS_BYTES ? XS_BYTES : CS_BYTES];
  auto xs = reinterpret_cast<half (*)[RB][MMAX][XROW]>(smem);
  auto cs = reinterpret_cast<float (*)[256]>(smem);

  const int grp = blockIdx.y;
  const int beg = goff[grp];
  const int cnt = goff[grp + 1] - beg;
  if (cnt <= 0) return;  // block-uniform: padding group, no weight read
  const int e = gids[grp];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const uint2 *cb = reinterpret_cast<const uint2 *>(qcodes) +
                    ((size_t)e * (N >> 5) + tile) * G * 32 + lane;
  // MXFP4 keeps one scale per 32 codes, so its table is half as long.
  const int SG = (SCALE_MODE == SCALE_MXFP4_E8M0_32) ? (G >> 1) : G;
  const uint8_t *sb =
      qscales + ((size_t)e * (N >> 5) + tile) * SG * 32 + lane;
  const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);

  // An expert with more than ROWS slots (hash routing at decode, large
  // prefill chunks) takes one pass per ROWS rows, each re-reading this
  // tile's weights; within a pass the weights are read once.
  for (int base = 0; base < cnt; base += ROWS) {
    const int rows = min(ROWS, cnt - base);
    const int blocks = (rows + MMAX - 1) / MMAX;
    __syncthreads();  // previous pass done with slots/xrows/cs
    if (threadIdx.x < ROWS) {
      const int m = threadIdx.x;
      const int slot = m < rows ? perm[beg + base + m] : 0;
      slots[m] = slot;
      xrows[m] = x + (size_t)(x_slot_major ? slot : slot / topk) * K;
    }
    __syncthreads();
    float c[RB][NACC][8];
#pragma unroll
    for (int bl = 0; bl < RB; bl++)
#pragma unroll
      for (int a = 0; a < NACC; a++)
#pragma unroll
        for (int i = 0; i < 8; i++) c[bl][a][i] = 0.f;

    if constexpr (RB == 1) {
      // Decode (at most 8 rows per expert): staging costs more than the
      // repeated loads it saves, so A comes straight from global memory.
      const half *xr = r < rows ? xrows[r] : nullptr;
#pragma unroll 4
      for (int g = g0; g < g0 + Gq; g++) {
        const uint2 q2 = __ldcs(cb + (size_t)g * 32);
        const half2 sc2 = __hmul2(group_scale<SCALE_MODE>(sb, g), gm2);
        half2 bq[8];
        dequant8_tm(q2.x, sc2, bq + 0);
        dequant8_tm(q2.y, sc2, bq + 4);
        const unsigned *B = reinterpret_cast<const unsigned *>(bq);
        uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
        if (xr) {
          a01 = *reinterpret_cast<const uint4 *>(xr + g * 16);
          a23 = *reinterpret_cast<const uint4 *>(xr + g * 16 + 8);
        }
        const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
        const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
        MMA_8N8K4(c[0][0], A0[0], A0[1], B[0], B[1]);
        MMA_8N8K4(c[0][1 % NACC], A0[2], A0[3], B[2], B[3]);
        MMA_8N8K4(c[0][2 % NACC], A1[0], A1[1], B[4], B[5]);
        MMA_8N8K4(c[0][3 % NACC], A1[2], A1[3], B[6], B[7]);
      }
    } else {
      // Lane l stages 16 B of row (l >> 2) of each row block per chunk. The
      // next chunk's codes and scales -- streamed from DRAM, unlike the
      // activation rows every tile shares through L2 -- are loaded while the
      // current one computes, hiding the latency that dominated the prefill
      // stalls.
      const int srow = lane >> 2, scol = (lane & 3) * 8;
      const int gend = g0 + Gq;
      // A warp's slice may end on an odd group (K/16 = splitk, for instance),
      // so every read of a chunk's second group is bounded by gend. The guards
      // are uniform across the warp and fall away when Gq is even.
      uint2 qv[XCH];
      uint8_t sv[XCH];
      auto load_weights = [&](int gc) {
  #pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          if (gi > 0 && gc + gi >= gend) break;
          qv[gi] = __ldcs(cb + (size_t)(gc + gi) * 32);
          sv[gi] = group_scale_byte<SCALE_MODE>(sb, gc + gi);
        }
      };
      load_weights(g0);
      for (int gc = g0; gc < gend; gc += XCH) {
        const int gcnt = min(XCH, gend - gc);
  #pragma unroll
        for (int bl = 0; bl < RB; bl++) {
          if (bl > 0 && bl >= blocks) break;  // block-uniform
          const int row = bl * MMAX + srow;
          uint4 v = make_uint4(0, 0, 0, 0);
          // scol spans 8 halves inside the chunk, so it belongs to group
          // gc + scol / 16; skip it when that group is past the slice.
          if (row < rows && scol / 16 < gcnt)
            v = *reinterpret_cast<const uint4 *>(xrows[row] + gc * 16 + scol);
          *reinterpret_cast<uint4 *>(&xs[warp][bl][srow][scol]) = v;
        }
        uint2 q[XCH];
        uint8_t s[XCH];
  #pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          if (gi > 0 && gi >= gcnt) break;
          q[gi] = qv[gi];
          s[gi] = sv[gi];
        }
        __syncwarp();
        if (gc + XCH < gend) load_weights(gc + XCH);
  #pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          if (gi > 0 && gi >= gcnt) break;
          const uint2 q2 = q[gi];
          const half2 sc2 = __hmul2(decode_scale<SCALE_MODE>(s[gi]), gm2);
          half2 bq[8];
          dequant8_tm(q2.x, sc2, bq + 0);
          dequant8_tm(q2.y, sc2, bq + 4);
          const unsigned *B = reinterpret_cast<const unsigned *>(bq);
  #pragma unroll
          for (int bl = 0; bl < RB; bl++) {
            // Block 0 always has rows; only later blocks can be empty (block-
            // uniform), and keeping block 0 unguarded leaves RB = 1 branch-free.
            if (bl > 0 && bl >= blocks) break;
            const uint4 a01 =
                *reinterpret_cast<const uint4 *>(&xs[warp][bl][r][gi * 16]);
            const uint4 a23 =
                *reinterpret_cast<const uint4 *>(&xs[warp][bl][r][gi * 16 + 8]);
            const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
            const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
            MMA_8N8K4(c[bl][0], A0[0], A0[1], B[0], B[1]);
            MMA_8N8K4(c[bl][1 % NACC], A0[2], A0[3], B[2], B[3]);
            MMA_8N8K4(c[bl][2 % NACC], A1[0], A1[1], B[4], B[5]);
            MMA_8N8K4(c[bl][3 % NACC], A1[2], A1[3], B[6], B[7]);
          }
        }
        __syncwarp();  // chunk consumed before the next one overwrites xs
      }
    }

    // Other warps may still read their stage, which the reduction buffer
    // overlaps.
    if constexpr (RB > 1) __syncthreads();
#pragma unroll
    for (int bl = 0; bl < RB; bl++) {
      if (bl > 0 && bl >= blocks) break;  // block-uniform: barriers match
#pragma unroll
      for (int a = 1; a < NACC; a++)
#pragma unroll
        for (int i = 0; i < 8; i++) c[bl][0][i] += c[bl][a][i];
      if (bl > 0) __syncthreads();  // previous block done reading cs
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        cs[warp][row * 32 + qp * 8 + col] = c[bl][0][i];
      }
      __syncthreads();
      for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < SPLITK; w++) v += cs[w][t];
        const int row = bl * MMAX + (t >> 5), col = t & 31;
        if (row < rows)
          y_slots[(size_t)slots[row] * N + (size_t)tile * 32 + col] =
              __float2half(v);
      }
    }
  }
}

// y_slots[S, N] (S = tokens*topk, slot-major) = x[row(slot)] @ W[expert(slot)]
// with W in per-expert QPN fragment order (see _qpn_prepack).
void skinny_moe_qpn(torch::Tensor x, torch::Tensor qcodes,
                    torch::Tensor qscales, torch::Tensor gscales,
                    torch::Tensor perm, torch::Tensor gids,
                    torch::Tensor goff, int64_t topk,
                    torch::Tensor y_slots, bool x_slot_major,
                    int64_t num_tokens, int64_t splitk, int64_t nacc,
                    int64_t scale_mode) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(gscales.is_cuda() && gscales.dtype() == torch::kFloat &&
              gscales.is_contiguous());
  TORCH_CHECK(perm.is_cuda() && perm.dtype() == torch::kInt && perm.is_contiguous());
  TORCH_CHECK(gids.is_cuda() && gids.dtype() == torch::kInt && gids.is_contiguous());
  TORCH_CHECK(goff.is_cuda() && goff.dtype() == torch::kInt &&
              goff.is_contiguous());
  TORCH_CHECK(y_slots.is_cuda() && y_slots.dtype() == torch::kHalf &&
              y_slots.is_contiguous());
  const int64_t T = num_tokens, K = x.size(1);
  const int64_t E = gscales.size(0), N = y_slots.size(1);
  const int64_t S = T * topk;
  TORCH_CHECK(x.size(0) == (x_slot_major ? S : T), "x rows mismatch");
  TORCH_CHECK(qcodes.numel() == E * N * (K >> 1), "qpn codes size");
  TORCH_CHECK(scale_mode == SCALE_NVFP4_FP8_16 ||
                  scale_mode == SCALE_MXFP4_E8M0_32,
              "scale_mode 0 (NVFP4 fp8/16) or 1 (MXFP4 e8m0/32)");
  // One scale per 16 codes for NVFP4, per 32 for MXFP4.
  const int64_t sh = 4 + scale_mode;
  TORCH_CHECK(qscales.numel() == E * N * (K >> sh), "qpn scales size");
  TORCH_CHECK(gids.size(0) == S && goff.size(0) == S + 1, "compact routing size");
  TORCH_CHECK(perm.size(0) == S && y_slots.size(0) == S);
  TORCH_CHECK(S <= 65535, "grouped qpn MoE: tokens * topk = ", S,
              " exceeds the CUDA grid y limit");
  TORCH_CHECK(K % 64 == 0 && (K / 16) % splitk == 0,
              "K/16 must split into splitk equal slices");
  TORCH_CHECK(N % 32 == 0, "N % 32");
  const dim3 grid((unsigned)(N / 32), (unsigned)S);
  auto stream = at::cuda::getCurrentCUDAStream();
  // Row blocks per weight load, from the mean rows per expert: one block
  // while experts see at most 8 rows (decode), two for prefill chunks. Four
  // need ~80 registers (halving the resident blocks per SM) and more shared
  // memory than the static 48 KiB for SPLITK 16.
  const int64_t mean_rows = (S + E - 1) / E;
  const int rb = mean_rows > 8 ? 2 : 1;

#define LAUNCH_MOE_QPN_R(SPv, NAv, SMv, RBv)                                \
  skinny_nvfp4_moe_qpn<SPv, NAv, SMv, RBv>                                  \
      <<<grid, dim3(32 * SPv), 0, stream>>>(                                \
          qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),          \
          gscales.data_ptr<float>(),                                        \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),           \
          reinterpret_cast<half *>(y_slots.data_ptr<at::Half>()),           \
          perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(), \
          (int)N, (int)K, (int)topk, x_slot_major ? 1 : 0)

#define LAUNCH_MOE_QPN_S(SPv, NAv, SMv)                                     \
  do {                                                                      \
    if (rb == 2)                                                            \
      LAUNCH_MOE_QPN_R(SPv, NAv, SMv, 2);                                   \
    else                                                                    \
      LAUNCH_MOE_QPN_R(SPv, NAv, SMv, 1);                                   \
  } while (0)

#define LAUNCH_MOE_QPN(SPv, NAv)                                            \
  do {                                                                      \
    if (scale_mode == SCALE_MXFP4_E8M0_32)                                  \
      LAUNCH_MOE_QPN_S(SPv, NAv, SCALE_MXFP4_E8M0_32);                      \
    else                                                                    \
      LAUNCH_MOE_QPN_S(SPv, NAv, SCALE_NVFP4_FP8_16);                       \
  } while (0)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 81: LAUNCH_MOE_QPN(8, 1); break;
    case 82: LAUNCH_MOE_QPN(8, 2); break;
    // SPLITK 10 serves K = 320 (K/16 = 20 = 10 slices of one 2-group chunk),
    // e.g. a 640-wide expert intermediate split over TP2, which 8 and 16
    // cannot divide.
    case 101: LAUNCH_MOE_QPN(10, 1); break;
    case 161: LAUNCH_MOE_QPN(16, 1); break;
    case 162: LAUNCH_MOE_QPN(16, 2); break;
    // SPLITK 32 would need more than the 48 KiB of static shared memory for
    // the per-warp activation stage plus the split-K reduction.
    default: TORCH_CHECK(false, "moe_qpn (splitk, nacc) in {(8,1), (8,2), (10,1), (16,1), (16,2)}");
  }
#undef LAUNCH_MOE_QPN
#undef LAUNCH_MOE_QPN_S
#undef LAUNCH_MOE_QPN_R
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_simt", &skinny_moe_simt,
        "grouped NVFP4 MoE GEMM (SIMT, device-side routing, tokens<=8)");
  m.def("moe_qpn", &skinny_moe_qpn,
        "grouped NVFP4/MXFP4 MoE GEMM (mma.m8n8k4 on prepacked fragments, "
        "compact device-side routing, 8 rows per weight read); scale_mode "
        "0 = NVFP4 fp8-e4m3 per 16 codes, 1 = MXFP4 e8m0 per 32",
        py::arg("x"), py::arg("qcodes"), py::arg("qscales"),
        py::arg("gscales"), py::arg("perm"), py::arg("gids"),
        py::arg("goff"), py::arg("topk"), py::arg("y_slots"),
        py::arg("x_slot_major"), py::arg("num_tokens"), py::arg("splitk"),
        py::arg("nacc"), py::arg("scale_mode") = 0);
  m.def("gemm_qpn8", &skinny_gemm_qpn8,
        "skinny FP8 E4M3 GEMM (QPN8, M<=8)");
  m.def("qpn8_blk_dequant", &skinny_qpn8_blk_dequant,
        "packed QPN8 bytes + scale raster -> fp16 [N,K] (transient, for the "
        "cuBLAS prefill band)");
  m.def("gemm_qpn8_blk_wmma", &skinny_gemm_qpn8_blk_wmma,
        "skinny block-scaled FP8 GEMM (tensor-core tiles from the packed "
        "layout, prefill band)");
  m.def("gemm_qpn8_blk", &skinny_gemm_qpn8_blk,
        "skinny block-scaled FP8 GEMM (QPN8 + [bn,bk] scale raster, M<=8)");
  m.def("gemm_qpn8_blk_mt2", &skinny_gemm_qpn8_blk_mt2,
        "skinny block-scaled FP8 GEMM (QPN8 MT=2 + [bn,bk] scale raster, "
        "M<=16)");
  m.def("gemm_qpn8_mt2", &skinny_gemm_qpn8_mt2,
        "skinny FP8 E4M3 GEMM (QPN8 MT=2, two row-tiles, one weight stream, "
        "M<=16)");
  m.def("gemm_qpn2", &skinny_gemm_qpn2,
        "skinny NVFP4 GEMM (QP-N geometry winner, M<=8)");
  m.def("pack_x8", &skinny_pack_x8_op,
        "block-pack activations to xb[k/16][8][16] (diagnostic entry: the "
        "QP-N wrappers call the kernel themselves)");
  m.def("gemm_qpn", &skinny_gemm_qpn,
        "skinny NVFP4 GEMM (QP-N mma.m8n8k4, prepacked weights, M<=16)");
  m.def("gemm_qpn_simt", &skinny_gemm_qpn_simt,
        "skinny NVFP4 GEMM (qpn-layout SIMT, M<=3, CT-stash-free)");
  m.def("gemm_dp4a", &skinny_gemm_dp4a,
        "skinny NVFP4 GEMM (int8 dp4a SIMT, W lossless / A8 per-16)");
  m.def("gemm_simt", &skinny_gemm_simt, "skinny NVFP4 GEMM (SIMT, M<=8)");
  m.def("gemm_simt_argmax", &skinny_gemm_simt_argmax,
        "fused NVFP4 lm_head GEMM + greedy argmax (M=1; block val/idx)");
  m.def("gemm_wmma", &skinny_gemm_wmma, "skinny NVFP4 GEMM (WMMA, M<=64)");
  m.def("gemm_mma8", &skinny_gemm_mma8,
        "skinny NVFP4 GEMM (mma.m8n8k4, M<=8)");
  m.def("gemm_wmma_cfg", &skinny_gemm_wmma_cfg,
        "skinny NVFP4 GEMM (WMMA, config-selectable tile sweep)");
  m.def("gemm_wmma_splitk", &skinny_gemm_wmma_splitk,
        "skinny NVFP4 GEMM (WMMA split-K, fp32 partials)");
}
