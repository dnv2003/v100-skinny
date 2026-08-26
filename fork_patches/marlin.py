# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Patched: adds an optional custom SM70 "skinny" NVFP4 kernel for M<=64
# (decode / speculative-verification widths), env-gated by
# VLLM_SKINNY_NVFP4=1. Weights are kept in the checkpoint's native
# layout (codes [N][K/2], fp8-e4m3 group-16 scales [N][K/16]) alongside
# the marlin-repacked copy, which serves M>64 (prefill).
#
# All M-dependent dispatch lives INSIDE one torch custom op whose eager
# body sees the true runtime M — torch.compile treats the op as opaque,
# so no shape-dependent branch can be constant-folded into a compiled
# graph. Eager-only self-check vs marlin auto-disables on mismatch.
# Original file: marlin.py.orig.

import os

import torch
from torch.library import custom_op

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

_SKINNY_ENABLED = os.environ.get("VLLM_SKINNY_NVFP4", "0") == "1"
_SKINNY_SRC = os.environ.get(
    "VLLM_SKINNY_NVFP4_SRC",
    # Default: the kernel source shipped alongside this checkout. Set
    # VLLM_SKINNY_NVFP4_SRC to override. Never a machine-specific path.
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "kernels", "skinny_kernels.cu"),
)
import os as _os

# Upper M bound for the skinny WMMA path; above it, marlin serves the
# GEMM. Env-tunable for dispatch experiments (e.g., 7 sends the whole
# batch regime to marlin, keeping only the SIMT decode band).
_SKINNY_MAX_M = int(_os.environ.get("VLLM_SKINNY_MAX_M", "64"))
# QPN (Volta-native four-quadpair mma.m8n8k4) owns M 4..16 when enabled:
# frontier from qpn_sweep_20260810 — 1.28x M=5 / 1.69x M=8 / 1.29x M=11 /
# 1.22x M=16 vs prior best. Costs one extra weight-sized copy per layer
# (fragment-order prepack, built once at load); disable with
# VLLM_SKINNY_QPN=0 if the memory envelope needs it back.
_QPN_ENABLED = os.environ.get("VLLM_SKINNY_QPN", "1") == "1"
# Drop the checkpoint-native stash once the qpn prepack exists: the
# prepack is a byte-equal permutation, so serving returns to the pre-QPN
# weight footprint (fp32 GDN state then fits). M1-3 route to
# gemm_qpn_simt (qpn-layout SIMT), M17+ to marlin (conceded band).
_QPN_DROP_CT = os.environ.get("VLLM_SKINNY_DROP_CT", "0") == "1"
# QPN2: geometry-winner kernel for M 4..8 (qpn_matrix/qpn_msweep
# 2026-08-17: weighted 637 GB/s vs 441; -30.7% QPN time). Same prepack,
# same bytes — kernel and launch geometry only. VLLM_SKINNY_QPN2=0
# restores the fixed-4-warp kernel.
_QPN2_ENABLED = os.environ.get("VLLM_SKINNY_QPN2", "1") == "1"
# (K, N) -> (splitk, nacc), measured winners; heuristic covers the rest.
# graph-mode winners (qpn_graphmatrix 2026-08-17): captured-replay is
# the serving regime and reshuffles two cells vs the eager matrix.
_QPN2_TABLE = {(1536, 5120): (16, 2), (4352, 5120): (16, 2),
               (5120, 8704): (8, 2), (5120, 4096): (16, 2),
               (5120, 2048): (32, 2), (5120, 62080): (8, 1),
               (5120, 3584): (16, 2)}


def _qpn2_cfg(k: int, n: int):
    cfg = _QPN2_TABLE.get((k, n))
    if cfg is not None and (k // 16) % cfg[0] == 0:
        return cfg
    # smallest split that puts ~640+ warps in flight (80 SMs x 8)
    for s in (8, 16, 32):
        if (k // 16) % s == 0 and (n // 32) * s >= 640:
            return (s, 2 if s >= 16 else 1)
    for s in (16, 8):
        if (k // 16) % s == 0:
            return (s, 2 if s >= 16 else 1)
    return None
_SELF_CHECK_CALLS = 3
_SELF_CHECK_TOL = 3e-2


def _qpn_prepack(codes: torch.Tensor, scales: torch.Tensor):
    """Fragment-order prepack for gemm_qpn: [tile N/32][group K/16]
    [lane 32] x 8B, nibbles pre-interleaved so the TM decoder's (i, i+4)
    output is exactly the adjacent-k B-fragment register pair. Pure
    permutation of the checkpoint bytes."""
    n, k2 = codes.shape
    k = k2 * 2
    if n % 32 or k % 64:
        return None, None
    dev = codes.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=dev)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7,
                           8, 10, 12, 14, 9, 11, 13, 15], device=dev)
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    g = torch.arange(groups, device=dev)
    kidx = g.view(groups, 1) * 16 + korder.view(1, 16)
    qc = torch.empty(tiles, groups, 32, 8, dtype=torch.uint8, device=dev)
    qs = torch.empty(tiles, groups, 32, dtype=torch.uint8, device=dev)
    # Chunk the gather: the int64 broadcast-index intermediates are 16x
    # the payload — one-shot on lm_head (37984x5120) spikes 2.4 GiB and
    # OOM'd the memory-profiler forward. Cap transients at ~300 MB.
    chunk = max(1, 36864 // groups)
    for t0 in range(0, tiles, chunk):
        t1 = min(t0 + chunk, tiles)
        tt = t1 - t0
        ncol = (torch.arange(t0, t1, device=dev).view(tt, 1) * 32
                + col.view(1, 32))
        nb = nib[ncol.view(tt, 1, 32, 1).expand(tt, groups, 32, 16),
                 kidx.view(1, groups, 1, 16).expand(tt, groups, 32, 16)]
        qc[t0:t1] = nb[..., 0::2] | (nb[..., 1::2] << 4)
        qs[t0:t1] = scales[ncol.view(tt, 1, 32).expand(tt, groups, 32),
                           g.view(1, groups, 1).expand(tt, groups, 32)]
    del nib
    return qc.view(-1).contiguous(), qs.view(-1).contiguous()


def _qpn_stash(layer) -> None:
    """Attach skinny_qpn_codes/scales (empty tensors when disabled or
    shape-ineligible, so the custom-op signature stays uniform)."""
    qc = qs = None
    if _QPN_ENABLED:
        qc, qs = _qpn_prepack(layer.skinny_codes.data,
                              layer.skinny_scales.data)
    empty = layer.skinny_codes.data.new_empty(0)
    layer.skinny_qpn_codes = torch.nn.Parameter(
        qc if qc is not None else empty, requires_grad=False)
    layer.skinny_qpn_scales = torch.nn.Parameter(
        qs if qs is not None else empty, requires_grad=False)
    if _QPN_DROP_CT and qc is not None:
        layer.skinny_codes.data = empty
        layer.skinny_scales.data = empty

_skinny_ext = None
_skinny_ok = True
_self_checks_done = 0


def _get_skinny_ext():
    global _skinny_ext
    if _skinny_ext is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _skinny_ext = load(
            name="skinny_nvfp4_v11",
            sources=[_SKINNY_SRC],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                               "-gencode=arch=compute_70,code=sm_70"],
            verbose=False,
        )
        logger.info_once("SM70 skinny NVFP4 kernel loaded from %s", _SKINNY_SRC)
    return _skinny_ext


_route_log_seen: set = set()

# Per-(route, M) dispatch counter. The op body runs at capture/compile/
# eager time (replays skip it), so these tally dispatch DECISIONS — proof
# that e.g. every M=16 call routes to qpn and zero to wmma/marlin. When
# VLLM_SKINNY_ROUTE_COUNT_FILE is set, counts are flushed (throttled) to
# "<file>.<pid>" per rank for the campaign to read back.
import collections as _collections
import json as _json

_route_counts: "_collections.Counter" = _collections.Counter()
_ROUTE_COUNT_FILE = _os.environ.get("VLLM_SKINNY_ROUTE_COUNT_FILE")
_route_flush_n = [0]


def _bump_route_count(route: str, m: int) -> None:
    _route_counts[(route, int(m))] += 1
    if not _ROUTE_COUNT_FILE:
        return
    _route_flush_n[0] += 1
    if _route_flush_n[0] % 50 != 0:
        return
    try:
        payload = {f"{r}:M{mm}": c for (r, mm), c in _route_counts.items()}
        with open(f"{_ROUTE_COUNT_FILE}.{_os.getpid()}", "w") as f:
            _json.dump(payload, f)
    except Exception:
        pass


@custom_op("skinny_nvfp4::linear", mutates_args=())
def _skinny_linear(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                   qpn_codes: torch.Tensor, qpn_scales: torch.Tensor,
                   marlin_w: torch.Tensor, marlin_s: torch.Tensor,
                   marlin_gs: torch.Tensor, workspace: torch.Tensor,
                   gscale: float, n: int, k: int) -> torch.Tensor:
    m = x.shape[0]
    # Frontier (qpn_sweep_20260810): simt M<=3, qpn M4..16 (Volta-native
    # four-quadpair m8n8k4, prepacked weights), wmma 17..64. With QPN
    # disabled the legacy seam applies: simt through M=7 (seam_bench),
    # wmma above.
    has_ct = codes.numel() > 0
    use_qpn = qpn_codes.numel() > 0 and 4 <= m <= 16 \
        and k % 64 == 0 and n % 32 == 0
    # qpn2 owns M 1..8 (msweep: 550-800 GB/s at M=1 vs qpn1/simt);
    # eligibility mirrors use_qpn's shape checks without its m >= 4.
    qpn2_ok = (_QPN2_ENABLED and qpn_codes.numel() > 0 and 1 <= m <= 8
               and k % 64 == 0 and n % 32 == 0)
    qpn2_cfg = _qpn2_cfg(k, n) if qpn2_ok else None
    use_qpn1 = (not has_ct) and qpn_codes.numel() > 0 and 1 <= m <= 3 \
        and k % 64 == 0 and n % 32 == 0
    use_simt = has_ct and (not use_qpn) and 1 <= m <= 7 \
        and k % 128 == 0 and n % 8 == 0
    use_wmma = has_ct and (not use_qpn) and (not use_simt) \
        and m <= _SKINNY_MAX_M and k % 128 == 0 and n % 64 == 0
    route = "qpn2" if qpn2_cfg is not None else "qpn" if use_qpn \
        else "qpn1" if use_qpn1 \
        else "simt" if use_simt else "wmma" if use_wmma else "marlin"
    # Route map: the op body runs at capture/compile/eager time, so one
    # line per unique (route, M) documents what each graph replays.
    key = (route, m, n, k)
    _bump_route_count(route, m)
    if key not in _route_log_seen:
        _route_log_seen.add(key)
        logger.info("Skinny route map: M=%d N=%d K=%d -> %s", m, n, k, route)
    if _skinny_ok and qpn2_cfg is not None:
        return _get_skinny_ext().gemm_qpn2(
            x, qpn_codes, qpn_scales, gscale, n, qpn2_cfg[0], qpn2_cfg[1])
    if _skinny_ok and use_qpn:
        return _get_skinny_ext().gemm_qpn(x, qpn_codes, qpn_scales, gscale, n)
    if _skinny_ok and use_qpn1:
        return _get_skinny_ext().gemm_qpn_simt(x, qpn_codes, qpn_scales,
                                               gscale, n)
    if _skinny_ok and (use_simt or use_wmma):
        ext = _get_skinny_ext()
        fn = ext.gemm_simt if use_simt else ext.gemm_wmma
        return fn(x, codes, scales, gscale)
    return apply_fp4_marlin_linear(
        input=x, weight=marlin_w, weight_scale=marlin_s,
        weight_global_scale=marlin_gs, workspace=workspace,
        size_n=n, size_k=k, bias=None,
    )


@_skinny_linear.register_fake
def _skinny_linear_fake(x, codes, scales, qpn_codes, qpn_scales, marlin_w,
                        marlin_s, marlin_gs, workspace, gscale, n, k):
    return x.new_empty((x.shape[0], n))


def _eager_context() -> bool:
    """True only when neither compiling/tracing nor capturing a graph."""
    if torch.compiler.is_compiling():
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        return False
    return True


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16), with an optional
    custom skinny kernel for M<=64."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression "
            "will be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
        if _SKINNY_ENABLED:
            # Native CT layout is exactly what the skinny kernel consumes;
            # stash it (and the already-multiplicative global) before the
            # marlin repack rewrites the layer's tensors.
            layer.skinny_codes = torch.nn.Parameter(
                layer.weight.data.clone(), requires_grad=False
            )
            layer.skinny_scales = torch.nn.Parameter(
                layer.weight_scale.data.view(torch.uint8).clone(),
                requires_grad=False,
            )
            layer.skinny_gscale = float(
                layer.weight_global_scale.data.float().item()
            )
            _qpn_stash(layer)
            logger.info_once(
                "SM70 skinny NVFP4 path enabled for M<=%d (QPN %s).",
                _SKINNY_MAX_M, "on" if _QPN_ENABLED else "off"
            )
        prepare_fp4_layer_for_marlin(layer)

    def _marlin_apply(self, layer, x, bias):
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _skinny_ok, _self_checks_done
        if not (_SKINNY_ENABLED and hasattr(layer, "skinny_codes")
                and x.dtype == torch.half):
            return self._marlin_apply(layer, x, bias)

        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        xc = x.reshape(-1, k).contiguous()
        y = torch.ops.skinny_nvfp4.linear(
            xc, layer.skinny_codes, layer.skinny_scales,
            layer.skinny_qpn_codes, layer.skinny_qpn_scales,
            layer.weight, layer.weight_scale, layer.weight_global_scale,
            layer.workspace, layer.skinny_gscale, n, k,
        )
        if bias is not None:
            y = y + bias

        if _self_checks_done < _SELF_CHECK_CALLS and _eager_context() \
                and _skinny_ok and xc.shape[0] <= _SKINNY_MAX_M:
            _self_checks_done += 1
            y_ref = self._marlin_apply(layer, xc, bias)
            denom = y_ref.float().abs().max().clamp(min=1e-6)
            rel = ((y.float() - y_ref.float()).abs().max() / denom).item()
            if rel > _SELF_CHECK_TOL:
                _skinny_ok = False
                logger.error(
                    "Skinny NVFP4 self-check FAILED (rel err %.3e, M=%d, "
                    "N=%d, K=%d); falling back to marlin permanently.",
                    rel, xc.shape[0], n, k,
                )
                return y_ref
            logger.info(
                "Skinny NVFP4 self-check ok (rel err %.3e, M=%d, N=%d, K=%d).",
                rel, xc.shape[0], n, k,
            )
        return y.reshape(x.shape[:-1] + (n,))


# ---------------------------------------------------------------------------
# Optional 4-bit lm_head (VLLM_SKINNY_LMHEAD=1): lazily NVFP4-pack the
# fp16 logits weight and route small-batch logits GEMMs through the
# skinny kernel. fp16 path preserved as fallback and for large batches.
#
# POLICY (2026-08-06): NOT the serving default, by decision — logits
# quantization measurably perturbs the output distribution (kernel-level
# top-1 agreement 92% on worst-case hidden states; ~0.005 nats KLD).
# Saves ~0.69 ms/token when enabled; treat as an opt-in for
# throughput-over-fidelity deployments only.
# ---------------------------------------------------------------------------
_LMHEAD_ENABLED = os.environ.get("VLLM_SKINNY_LMHEAD", "0") == "1"

# Native NVFP4 lm_head (VLLM_SKINNY_LMHEAD_NATIVE=1 or =<path>): load
# NVIDIA's ORIGINAL lm_head codes/scales from the modelopt source
# checkpoint — zero requantization. Verified 2026-08-11: lo-first
# nibbles, value = e2m1 * fp8_scale * weight_scale_2, bf16-exact vs the
# CT rendering on all sampled rows. Requires VLLM_SKINNY_LMHEAD=1.
_LMHEAD_NATIVE = os.environ.get("VLLM_SKINNY_LMHEAD_NATIVE", "")
# BF16-LMHead checkpoint (VLLM_SKINNY_LMHEAD_BF16=1): the checkpoint's
# lm_head is a dense BF16 tensor, not quantized. Serve it dense via the
# original fp16 linear — never repack, never pull native codes.
_LMHEAD_BF16 = os.environ.get("VLLM_SKINNY_LMHEAD_BF16", "0") == "1"


def _lmhead_resolve_native_path():
    """Resolve native lm_head codes from the SERVED checkpoint.

    NEVER falls back to another model's checkpoint: vocab sizes commonly
    match across a family, so a hardcoded path loads a foreign head with no
    shape error. Returns None when the served checkpoint has no native
    NVFP4 lm_head, and the caller then packs from the model's own weights.
    """
    import glob
    import json
    try:
        from vllm.config import get_current_vllm_config
        model_dir = get_current_vllm_config().model_config.model
    except Exception as exc:
        logger.warning("Native lm_head: cannot resolve served model dir (%s); "
                       "packing from the model's own weights instead", exc)
        return None
    if not os.path.isdir(model_dir):
        return None
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        try:
            wm = json.load(open(idx))["weight_map"]
        except Exception:
            wm = {}
        for key in wm:
            if key.endswith("lm_head.weight_scale"):
                return os.path.join(model_dir, wm[key])
        return None
    for cand in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        try:
            from safetensors import safe_open
            with safe_open(cand, "pt") as f:
                if any(k.endswith("lm_head.weight_scale") for k in f.keys()):
                    return cand
        except Exception:
            continue
    return None

if _LMHEAD_ENABLED:
    _E2M1_MAGS = None
    _E2M1_MIDS = None

    def _pack_rows(w):
        global _E2M1_MAGS, _E2M1_MIDS
        if _E2M1_MAGS is None:
            _E2M1_MAGS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6],
                                      device=w.device)
            _E2M1_MIDS = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0],
                                      device=w.device)
        n, k = w.shape
        wf = w.float().view(n, k // 16, 16)
        scale = wf.abs().amax(-1) / 6.0
        gmax = scale.max()
        q8 = (scale / (gmax / 448.0)).to(torch.float8_e4m3fn)
        eff = (q8.float() * (gmax / 448.0)).clamp(min=1e-12).unsqueeze(-1)
        idx = torch.bucketize((wf / eff).abs(), _E2M1_MIDS)
        sgn = (wf / eff) < 0
        codes = (idx | (sgn.long() << 3)).view(n, k).to(torch.uint8)
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        return packed, q8.view(torch.uint8).contiguous(), float(gmax / 448.0)

    def _lmhead_pack(layer):
        w = layer.weight.data
        n, k = w.shape
        chunks_c, chunks_s, gmaxes = [], [], []
        # two passes: global scale must be tensor-wide
        step = 8192
        for i in range(0, n, step):
            wf = w[i:i + step].float().view(-1, k // 16, 16)
            gmaxes.append((wf.abs().amax(-1) / 6.0).max())
        g = float(torch.stack(gmaxes).max().item() / 448.0)
        for i in range(0, n, step):
            wf = w[i:i + step].float().view(-1, k // 16, 16)
            scale = wf.abs().amax(-1) / 6.0
            q8 = (scale / g).to(torch.float8_e4m3fn)
            eff = (q8.float() * g).clamp(min=1e-12).unsqueeze(-1)
            idx = torch.bucketize((wf / eff).abs(), _E2M1_MIDS if
                                  _E2M1_MIDS is not None else
                                  torch.tensor([.25, .75, 1.25, 1.75, 2.5,
                                                3.5, 5.0], device=w.device))
            sgn = (wf / eff) < 0
            codes = (idx | (sgn.long() << 3)).view(-1, k).to(torch.uint8)
            chunks_c.append((codes[:, 0::2] | (codes[:, 1::2] << 4)))
            chunks_s.append(q8.view(torch.uint8).view(-1, k // 16))
            del wf, scale, q8, eff, idx, sgn, codes
        layer.skinny_lm_codes = torch.cat(chunks_c).contiguous()
        layer.skinny_lm_scales = torch.cat(chunks_s).contiguous()
        layer.skinny_lm_gscale = g
        logger.info("Skinny lm_head packed: [%d, %d] -> %.0f MB (4-bit)",
                    n, k, layer.skinny_lm_codes.numel() / 2**20)

    def _lmhead_load_native(layer):
        """Load NVIDIA's original NVFP4 lm_head codes/scales for this
        rank's vocab shard. No requantization — these ARE the published
        model's head bits."""
        from safetensors import safe_open
        from vllm.distributed import get_tensor_model_parallel_rank
        path = (_LMHEAD_NATIVE if _LMHEAD_NATIVE != "1"
                else _lmhead_resolve_native_path())
        if not path or not os.path.exists(path):
            logger.warning(
                "Native lm_head: served checkpoint has no native NVFP4 "
                "lm_head; packing from the model's own weights (no foreign "
                "checkpoint is ever borrowed).")
            _lmhead_pack(layer)
            return
        logger.info("Native lm_head source: %s", path)
        n, k = layer.weight.shape
        off = get_tensor_model_parallel_rank() * n
        with safe_open(path, "pt") as f:
            _lmk = "lm_head.weight"
            if _lmk not in f.keys():
                _cand = [k for k in f.keys() if k.endswith("lm_head.weight")]
                if not _cand:
                    logger.warning("Native lm_head: no lm_head.weight in %s; "
                                   "packing from the model's own weights", path)
                    _lmhead_pack(layer)
                    return
                _lmk = _cand[0]
            total = f.get_slice(_lmk).get_shape()[0]
            if off + n > total:
                logger.warning(
                    "Native lm_head shard [%d:%d] exceeds source rows %d "
                    "(padded vocab?) — falling back to requant pack.",
                    off, off + n, total)
                _lmhead_pack(layer)
                return
            codes = f.get_slice(_lmk)[off:off + n]
            scales = f.get_slice(_lmk + "_scale")[off:off + n]
            g = float(f.get_tensor(_lmk + "_scale_2").float().item())
        dev = layer.weight.device
        layer.skinny_lm_codes = codes.to(torch.uint8).to(dev).contiguous()
        layer.skinny_lm_scales = (scales.view(torch.uint8).to(dev)
                                  .contiguous())
        layer.skinny_lm_gscale = g
        logger.info(
            "Skinny lm_head: NATIVE NVFP4 codes loaded (zero requant), "
            "rows %d..%d of %d, %.0f MB", off, off + n, total,
            layer.skinny_lm_codes.numel() / 2**20)

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    _orig_embed_apply = UnquantizedEmbeddingMethod.apply

    def _skinny_embed_apply(self, layer, x, bias=None):
        w = getattr(layer, "weight", None)
        if (w is None or w.dim() != 2 or w.shape[0] < 32768
                or w.shape[1] % 128 != 0 or w.shape[0] % 64 != 0
                or x.dtype != torch.half or bias is not None):
            return _orig_embed_apply(self, layer, x, bias)
        if _LMHEAD_BF16:
            logger.info_once(
                "Skinny lm_head: native BF16 head served dense (N=%d K=%d, no repack)",
                w.shape[0], w.shape[1])
            return _orig_embed_apply(self, layer, x, bias)
        if not hasattr(layer, "skinny_lm_codes"):
            if _LMHEAD_NATIVE:
                _lmhead_load_native(layer)
            else:
                _lmhead_pack(layer)
        x2 = x.reshape(-1, w.shape[1])
        m = x2.shape[0]
        if m > 64:
            return _orig_embed_apply(self, layer, x, bias)
        ext = _get_skinny_ext()
        # OFF by default: routing the logits head through a different
        # kernel than the drafter's M=1 path flips 4-bit-head near-ties —
        # measured 2026-08-10: extraction acceptance 82% (vs 97-100) and
        # 2/4 losslessness diffs failed. Trunk QPN is unaffected.
        # 2026-08-17: with qpn2 the historical blocker inverts. The
        # 2026-08-10 failure (extraction acceptance 82%) came from the
        # drafter (M=1, simt) and verifier (M=8, qpn) computing logits
        # through DIFFERENT kernels whose near-ties disagree. qpn2's
        # reduction order is M-independent, so routing M 1..8 through it
        # gives both sides bit-identical logits per row.
        lm_cfg = None
        if (os.environ.get("VLLM_SKINNY_QPN_LMHEAD", "1") == "1"
                and _QPN2_ENABLED and 1 <= m <= 8
                and w.shape[1] % 64 == 0 and w.shape[0] % 32 == 0):
            lm_cfg = _qpn2_cfg(w.shape[1], w.shape[0])
        if lm_cfg is not None:
            if not hasattr(layer, "skinny_lm_qpn_codes"):
                qc, qs = _qpn_prepack(layer.skinny_lm_codes,
                                      layer.skinny_lm_scales)
                layer.skinny_lm_qpn_codes = qc
                layer.skinny_lm_qpn_scales = qs
            if layer.skinny_lm_qpn_codes is None:
                lm_cfg = None
        if lm_cfg is not None:
            y = ext.gemm_qpn2(x2.contiguous(), layer.skinny_lm_qpn_codes,
                              layer.skinny_lm_qpn_scales,
                              layer.skinny_lm_gscale, w.shape[0],
                              lm_cfg[0], lm_cfg[1])
        else:
            fn = ext.gemm_simt if m <= 7 else ext.gemm_wmma
            y = fn(x2.contiguous(), layer.skinny_lm_codes,
                   layer.skinny_lm_scales, layer.skinny_lm_gscale)
        return y.reshape(x.shape[:-1] + (w.shape[0],))

    UnquantizedEmbeddingMethod.apply = _skinny_embed_apply

    # EAGER native-lm_head load: the lazy first-logits-call load fired
    # AFTER the memory profiler committed the pool and cost three boots
    # to a ~20 MiB OOM on the tightest rank. Wrap process_weights_after_
    # loading so the codes/scales are resident (and profiler-accounted)
    # at weight-load time. Requant path stays lazy (legacy only).
    _orig_embed_pw = getattr(UnquantizedEmbeddingMethod,
                             "process_weights_after_loading", None)

    def _skinny_embed_pw(self, layer):
        if _orig_embed_pw is not None:
            _orig_embed_pw(self, layer)
        if _LMHEAD_BF16:
            return
        w = getattr(layer, "weight", None)
        # ParallelLMHead only — embed_tokens shares the vocab shape and
        # would eat ~180 MB/rank of head codes it never reads.
        if (_LMHEAD_NATIVE and w is not None and w.dim() == 2
                and type(layer).__name__ == "ParallelLMHead"
                and w.shape[0] >= 32768 and w.shape[1] % 128 == 0
                and w.shape[0] % 64 == 0
                and not hasattr(layer, "skinny_lm_codes")):
            _lmhead_load_native(layer)

    UnquantizedEmbeddingMethod.process_weights_after_loading = \
        _skinny_embed_pw

    # BF16-LMHead: ModelOpt hands the unquantized head an
    # UnquantizedLinearMethod (never None), so the
    # UnquantizedEmbeddingMethod hook above never fires for it. Wrap
    # UnquantizedLinearMethod.apply to witness the dense serve; the
    # original apply is dense-correct — no pack, no repack, ever.
    if _LMHEAD_BF16:
        from vllm.model_executor.layers.linear import (
            UnquantizedLinearMethod,
        )

        _orig_lin_apply = UnquantizedLinearMethod.apply

        def _skinny_lin_apply(self, layer, x, bias=None):
            if layer.__class__.__name__ == "ParallelLMHead":
                w = getattr(layer, "weight", None)
                if w is not None and w.dim() == 2:
                    logger.info_once(
                        "Skinny lm_head: native BF16 head served dense "
                        "(N=%d K=%d, no repack)",
                        w.shape[0], w.shape[1])
            return _orig_lin_apply(self, layer, x, bias)

        UnquantizedLinearMethod.apply = _skinny_lin_apply
    # Deterministic BF16-native witness at LOAD time: the runtime forward path
    # does not reliably log on every boot, so also emit it at weight-load.
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod as _ULM
    _orig_ulm_pw = _ULM.process_weights_after_loading if hasattr(_ULM, "process_weights_after_loading") else None
    def _bf16_ulm_pw(self, layer):
        if _orig_ulm_pw is not None:
            _orig_ulm_pw(self, layer)
        if (_LMHEAD_BF16 and getattr(layer, "weight", None) is not None
                and type(layer).__name__ == "ParallelLMHead"
                and layer.weight.dim() == 2):
            logger.info_once(
                "Skinny lm_head: native BF16 head served dense (N=%d K=%d, no repack)",
                layer.weight.shape[0], layer.weight.shape[1],
            )
    _ULM.process_weights_after_loading = _bf16_ulm_pw
    logger.info_once("Skinny 4-bit lm_head path enabled (M<=64).")

    # Fused greedy argmax (VLLM_SKINNY_FUSED_ARGMAX=1): plug the fused
    # GEMM+argmax kernel into the fork's top1 socket. The drafter's
    # greedy sampling (and temp-0 target argmax) then never materializes
    # logits. Exact-equivalence gated (benchmarks/argmax_equiv.py):
    # identical half rounding, lowest-index ties. Falls through to the
    # original hook whenever any precondition fails.
    _FUSED_ARGMAX = os.environ.get("VLLM_SKINNY_FUSED_ARGMAX", "0") == "1"
    if _FUSED_ARGMAX:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding as _VPE,
        )

        _orig_top1 = _VPE.maybe_get_sm70_lm_head_top1

        def _skinny_top1(self, hidden_states, bias=None):
            codes = getattr(self, "skinny_lm_codes", None)
            if (codes is not None and bias is None
                    and hidden_states.dim() == 2
                    and hidden_states.shape[0] == 1
                    and hidden_states.dtype == torch.half
                    and self.shard_indices.num_org_vocab_padding == 0):
                ext = _get_skinny_ext()
                bv, bi = ext.gemm_simt_argmax(
                    hidden_states.contiguous(), codes,
                    self.skinny_lm_scales, self.skinny_lm_gscale)
                mx = bv.max()
                blk = (bv == mx).int().argmax()
                # Consumer uses these as GLOBAL indices (logits_processor
                # unpacks the hook return directly into global_indices).
                idx = (bi[blk].to(torch.long)
                       + self.shard_indices.org_vocab_start_index)
                return mx.view(1), idx.view(1)
            return _orig_top1(self, hidden_states, bias)

        _VPE.maybe_get_sm70_lm_head_top1 = _skinny_top1
        logger.info_once("Skinny FUSED lm_head argmax enabled (M=1).")


# ---------------------------------------------------------------------------
# Hook the W4A16 NVFP4 scheme too (lazily, to avoid circular imports at
# package init): wrap prepare_fp4_layer_for_marlin to stash the native
# tensors for ANY scheme that reaches marlin prepare, and install the
# W4A16 apply_weights patch on first use.
# ---------------------------------------------------------------------------
if _SKINNY_ENABLED:
    import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 \
        as _mu_fp4

    _w4a16_patched = [False]
    _orig_prepare = _mu_fp4.prepare_fp4_layer_for_marlin

    def _skinny_w4a16_apply(self, layer, x, bias=None):
        if (not hasattr(layer, "skinny_codes") or x.dtype != torch.half
                or not hasattr(layer, "workspace")):
            return self._skinny_orig_apply(layer, x, bias)
        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        xc = x.reshape(-1, k).contiguous()
        y = torch.ops.skinny_nvfp4.linear(
            xc, layer.skinny_codes, layer.skinny_scales,
            layer.skinny_qpn_codes, layer.skinny_qpn_scales,
            layer.weight, layer.weight_scale, layer.weight_global_scale,
            layer.workspace, layer.skinny_gscale, n, k)
        if bias is not None:
            y = y + bias
        return y.reshape(x.shape[:-1] + (n,))

    def _stashing_prepare(layer, input_dtype=None):
        # At this point both W4A4 and W4A16 schemes have layer.weight =
        # CT-packed codes and a multiplicative weight_global_scale.
        if (not hasattr(layer, "skinny_codes")
                and getattr(layer, "weight", None) is not None
                and layer.weight.dtype == torch.uint8):
            layer.skinny_codes = torch.nn.Parameter(
                layer.weight.data.clone(), requires_grad=False)
            layer.skinny_scales = torch.nn.Parameter(
                layer.weight_scale.data.view(torch.uint8).clone(),
                requires_grad=False)
            layer.skinny_gscale = float(
                layer.weight_global_scale.data.float().max().item())
            _qpn_stash(layer)
        if not _w4a16_patched[0]:
            _w4a16_patched[0] = True
            from vllm.model_executor.layers.quantization.compressed_tensors \
                .schemes.compressed_tensors_w4a16_nvfp4 import (
                    CompressedTensorsW4A16Fp4 as _S)
            _S._skinny_orig_apply = _S.apply_weights
            _S.apply_weights = _skinny_w4a16_apply
            logger.info_once("Skinny path hooked into W4A16 NVFP4 scheme.")
        return _orig_prepare(layer, input_dtype)

    _mu_fp4.prepare_fp4_layer_for_marlin = _stashing_prepare
    globals()["prepare_fp4_layer_for_marlin"] = _stashing_prepare
