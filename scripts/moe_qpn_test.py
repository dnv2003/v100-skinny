"""Grouped NVFP4/MXFP4 MoE kernels: correctness and timing against the per-expert loop.

`moe_qpn` runs every expert of a layer in one launch on the QPN tensor-core
path (mma.m8n8k4 over per-expert prepacked fragments); `moe_simt` is the SIMT
variant on the checkpoint layout. Both take device-side routing, so there is
no host sync and the launch captures into CUDA graphs.

Checked here, on synthetic weights so the script needs no checkpoint:
  - moe_qpn against a loop of `gemm_qpn` calls per expert (same decoder, tight
    tolerance) and against an fp32 dequantised reference,
  - moe_simt against the same reference (it serves up to 8 tokens),
  - an expert that fills the 8-row tile exactly, and at 64 and 128 tokens
    experts with more than 8 rows, which take the row-block path,
  - both scale rasters: NVFP4 (one fp8-e4m3 scale per 16 codes) and MXFP4
    (one E8M0 scale per 32 codes), selected by the kernel's `scale_mode`,
  - K = 320 with split-K 10 and K = 256 with split-K 16, where a warp's slice
    ends on an odd group,
  - decode, verify and prefill-chunk sizes, with per-layer timing.

Run per card: CUDA_VISIBLE_DEVICES=<n> python scripts/moe_qpn_test.py
"""
import os
import time

import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.environ.get("SKINNY_KERNELS_SRC",
                      os.path.join(_REPO, "kernels", "skinny_kernels.cu"))

ext = load(name="skinny_nvfp4_v11",
           sources=[_SRC],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                              "-gencode=arch=compute_70,code=sm_70"],
           verbose=False)
print("device:", torch.cuda.get_device_name(),
      "| moe_qpn:", hasattr(ext, "moe_qpn"), "| moe_simt:", hasattr(ext, "moe_simt"))

DEV = torch.device("cuda:0")
E, TOPK, HIDDEN, INTER = 64, 6, 2048, 1024
SCALE_NVFP4, SCALE_MXFP4 = 0, 1
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                    device=DEV)
KORDER = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]


def qpn_prepack(codes: torch.Tensor, scales: torch.Tensor, scale_group: int = 16):
    """The shim's _qpn_prepack: (N,K/2) codes + (N,K/scale_group) scale bytes ->
    [tile N/32][group][lane 32] x (8 code bytes, 1 scale byte). The code layout
    does not depend on scale_group; only the scale table is shorter for MXFP4,
    where a 32-code block is two adjacent 16-code groups sharing one scale."""
    n, k = codes.shape[0], codes.shape[1] * 2
    tiles, cgroups, sgroups = n // 32, k // 16, k // scale_group
    lane = torch.arange(32, device=DEV)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    kidx = (torch.arange(cgroups, device=DEV).view(cgroups, 1) * 16
            + torch.tensor(KORDER, device=DEV).view(1, 16))
    ncol = torch.arange(tiles, device=DEV).view(tiles, 1) * 32 + col.view(1, 32)
    nb = nib[ncol.view(tiles, 1, 32, 1).expand(tiles, cgroups, 32, 16),
             kidx.view(1, cgroups, 1, 16).expand(tiles, cgroups, 32, 16)]
    qc = nb[..., 0::2] | (nb[..., 1::2] << 4)
    qs = scales[ncol.view(tiles, 1, 32).expand(tiles, sgroups, 32),
                torch.arange(sgroups, device=DEV).view(1, sgroups, 1).expand(tiles, sgroups, 32)]
    return qc.contiguous(), qs.contiguous()


def make_experts(n: int, k: int, scale_mode: int):
    """Synthetic experts plus their fp32 truth, in the raster the mode selects.

    NVFP4 keeps one fp8-e4m3 scale per 16 codes. MXFP4 keeps one E8M0 exponent
    per 32 codes; the kernel decodes it as 2^(b - 112) and folds the remaining
    bias into the caller's global scale, which is what
    `rebase_e8m0_for_fp16` does on real checkpoints. The bytes here are drawn
    inside that window, so the script stays self-contained.
    """
    codes = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=DEV)
    group = 16 if scale_mode == SCALE_NVFP4 else 32
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(E, n, k)
    if scale_mode == SCALE_NVFP4:
        # E4M3 bytes in a sane band, away from the NaN encodings.
        scales = torch.randint(0x28, 0x48, (E, n, k // group), dtype=torch.uint8, device=DEV)
        factor = scales.view(torch.float8_e4m3fn).float()
        gscale = (torch.rand(E, device=DEV) * 0.5 + 0.75) * 1e-2
    else:
        # E8M0 is a bare exponent, value 2^(b - 127). The kernel writes
        # (b - 112) into the fp16 exponent field, whose bias is 15, so the
        # byte has to sit in 113..142 for the result to stay normal; real
        # checkpoints are shifted into that window by rebase_e8m0_for_fp16.
        scales = torch.randint(120, 135, (E, n, k // group), dtype=torch.uint8, device=DEV)
        factor = torch.pow(2.0, scales.float() - 127.0)
        # The kernel forms the group scale as decode(b) * (gscale * 2^14) in
        # fp16, so that product has to stay representable: keep the exponent
        # span and the global scale within about 2^16 of each other.
        gscale = (torch.rand(E, device=DEV) * 0.5 + 0.75) * 1e-3
    dense = E2M1[nib.long()] * factor.repeat_interleave(group, dim=2)
    dense = dense * gscale.view(E, 1, 1)
    packed = [qpn_prepack(codes[e], scales[e], group) for e in range(E)]
    qc = torch.stack([p[0] for p in packed]).contiguous()
    qs = torch.stack([p[1] for p in packed]).contiguous()
    return codes, scales, gscale.float().contiguous(), qc, qs, dense


def routing(ids: torch.Tensor):
    flat = ids.reshape(-1).to(torch.int64)
    slots = flat.numel()
    perm = torch.argsort(flat, stable=True).to(torch.int32)
    sorted_e = flat[perm.long()]
    counts = torch.bincount(flat, minlength=E).to(torch.int32)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=DEV)
    offsets[1:] = torch.cumsum(counts, 0)
    new_group = torch.ones(slots, dtype=torch.bool, device=DEV)
    new_group[1:] = sorted_e[1:] != sorted_e[:-1]
    gidx = torch.cumsum(new_group, 0) - 1
    goff = torch.full((slots + 1,), slots, dtype=torch.int64, device=DEV)
    goff.scatter_reduce_(0, gidx, torch.arange(slots, device=DEV), reduce="amin")
    gids = torch.zeros(slots, dtype=torch.int64, device=DEV)
    gids.scatter_(0, gidx, sorted_e)
    return perm, offsets, gids.to(torch.int32), goff.to(torch.int32)


def timed(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def run(name: str, n: int, k: int, slot_major: bool, cfg: tuple[int, int],
        scale_mode: int = SCALE_NVFP4, sizes=((1, False), (6, False), (8, False),
                                              (8, True), (64, False), (128, False))) -> bool:
    codes, scales, gscale, qc, qs, dense = make_experts(n, k, scale_mode)
    gl = gscale.tolist()
    ok = True
    raster = "NVFP4" if scale_mode == SCALE_NVFP4 else "MXFP4"
    print(f"\n-- {name}: N={n} K={k}, {raster}, moe_qpn splitk/nacc {cfg} --")
    for tokens, forced in sizes:
        torch.manual_seed(tokens + forced)
        ids = torch.topk(torch.randn(tokens, E, device=DEV), TOPK, dim=-1)[1].to(torch.int32)
        if forced:
            ids[:, 0] = 3
            ids[:, 1:] = torch.topk(torch.randn(tokens, E - 1, device=DEV), TOPK - 1, dim=-1)[1].to(torch.int32)
            ids[:, 1:] += (ids[:, 1:] >= 3).to(torch.int32)
        slots = tokens * TOPK
        rows = slots if slot_major else tokens
        x = (torch.randn(rows, k, device=DEV, dtype=torch.float16) / 8).contiguous()
        perm, offsets, gids, goff = routing(ids)
        flat = ids.reshape(-1).long()
        src = torch.arange(slots, device=DEV) if slot_major else torch.arange(slots, device=DEV) // TOPK
        ref = torch.einsum("sk,snk->sn", x[src].float(), dense[flat])

        def grouped():
            out = torch.empty(slots, n, dtype=torch.float16, device=DEV)
            ext.moe_qpn(x, qc, qs, gscale, perm, gids, goff, TOPK, out, slot_major,
                        tokens, *cfg, scale_mode)
            return out

        b = grouped()
        scale = ref.abs().max().item()
        d_ref = (ref - b.float()).abs().max().item()
        good = d_ref <= 5e-3 * scale + 1e-3
        line = (f"T={tokens:3d}{' one expert x' + str(tokens) if forced else '':16s} "
                f"experts={flat.unique().numel():3d}  |moe_qpn - fp32|={d_ref:.1e}  (max {scale:.2f})")
        if scale_mode == SCALE_NVFP4:
            # gemm_qpn and moe_simt read the NVFP4 raster only.
            def loop():
                out = torch.empty(slots, n, dtype=torch.float16, device=DEV)
                for e in flat.unique().tolist():
                    sel = (flat == e).nonzero().squeeze(1)
                    xe = x[src[sel]]
                    parts = [ext.gemm_qpn(xe[i:i + 16].contiguous(), qc[e], qs[e], gl[e], n)
                             for i in range(0, xe.shape[0], 16)]
                    out[sel] = torch.cat(parts)
                return out

            a = loop()
            d_loop = (a.float() - b.float()).abs().max().item()
            good &= d_loop <= 2e-3 * scale + 1e-3
            line += f"  |moe_qpn - loop|={d_loop:.1e}"
            if tokens <= 8 and k % 128 == 0:
                c = torch.empty(slots, n, dtype=torch.float16, device=DEV)
                ext.moe_simt(x, codes, scales, gscale, perm, offsets, TOPK, c, slot_major, tokens)
                d_simt = (ref - c.float()).abs().max().item()
                good &= d_simt <= 5e-3 * scale + 1e-3
                line += f"  |moe_simt - fp32|={d_simt:.1e}"
                line += f"  simt {timed(lambda: ext.moe_simt(x, codes, scales, gscale, perm, offsets, TOPK, c, slot_major, tokens)):.3f} ms"
            line += f"  loop {timed(loop, 8):.2f} ms"
        line += f"  qpn {timed(grouped):.3f} ms  {'OK' if good else 'FAIL'}"
        print(line)
        ok &= good
    return ok


results = [
    run("w13 (token-major x)", 2 * INTER, HIDDEN, False, (16, 1)),
    run("w2 (slot-major x)", HIDDEN, INTER, True, (8, 1)),
    run("w13 MXFP4", 2 * INTER, HIDDEN, False, (16, 1), SCALE_MXFP4),
    run("w2 MXFP4", HIDDEN, INTER, True, (8, 1), SCALE_MXFP4),
    # A warp slice of one group: K/16 == splitk, so a chunk has no second group.
    run("w2 K=256, split-K 16 (odd slice end)", HIDDEN, 256, True, (16, 1)),
    # Qwen3.8-Flash-Next's expert intermediate over two ranks: K/16 = 20.
    run("w2 K=320, split-K 10", HIDDEN, 320, True, (10, 1)),
]
print("\nALL OK" if all(results) else "\nFAILURES")
