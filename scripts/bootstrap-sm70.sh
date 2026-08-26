#!/usr/bin/env bash
# Bootstrap the SM70 NVFP4 serving stack from a clean checkout.
#
# Installs the pinned 1Cat-vLLM wheel into a fresh environment, deploys the
# fork patches over the installed package, and warms the skinny-kernel JIT
# build so the first served request does not pay for it.
#
# Nothing here is specific to our machines: every path is derived from the
# checkout or overridable by environment variable.
#
#   VLLM_WHEEL     OPTIONAL. Defaults to the pinned 1Cat-vLLM 1.3.0 release
#                  URL, whose SHA256 is pinned below and verified before
#                  installation. If you override it you MUST also set
#                  VLLM_WHEEL_SHA256 -- an unverified wheel is refused.
#   VLLM_WHEEL_SHA256  required alongside a custom VLLM_WHEEL
#                  1cat-vllm 1.3.0 — 1Cat AI's SM70/Volta fork of vLLM,
#                  Apache-2.0: https://github.com/1CatAI/1Cat-vLLM
#                  This project is a set of NVFP4/FP8 kernels and patches
#                  ON TOP of that engine; it is not an engine itself.
#   ENV_PREFIX     conda env prefix to create/use   (default: ./.venv-sm70)
#   PYTHON_VERSION default 3.12
#
# Usage:  bash scripts/bootstrap-sm70.sh          # pinned + digest-verified
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.venv-sm70}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "==> $*"; }

# ---------------------------------------------------------------- preflight
say "checking prerequisites"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; need an NVIDIA driver"

CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ' ')
echo "    compute capability: $CAPS"
case "$CAPS" in
  *7.0*) ;;
  *) die "no SM70 (compute 7.0 / Volta) GPU found — this stack targets V100.
       Found: $CAPS" ;;
esac

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "    GPUs visible: $NGPU"
# The serving config is TP4 and the published measurements are 4x V100-16GB.
# Discover this now, not two steps into a build.
[ "$NGPU" -ge "${REQUIRE_GPUS:-4}" ] || die "this configuration needs ${REQUIRE_GPUS:-4} GPUs; $NGPU visible
       (set REQUIRE_GPUS=n only if you intend a different topology --
        the published numbers are TP4 on 4x V100-SXM2-16GB)"

# The skinny kernels are built by torch's JIT extension loader, which shells
# out to nvcc. It is frequently absent from PATH even where CUDA is installed.
NVCC="${NVCC:-$(command -v nvcc || true)}"
[ -n "$NVCC" ] || for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.8/bin/nvcc; do
  [ -x "$c" ] && NVCC="$c" && break
done
[ -n "$NVCC" ] || die "nvcc not found. Install the CUDA toolkit or set NVCC=/path/to/nvcc"
echo "    nvcc: $NVCC ($("$NVCC" --version | tail -1))"
export PATH="$(dirname "$NVCC"):$PATH"

# The exact wheel every published number was measured on. Digest verified
# against the installed artifact on the reference machine.
PINNED_WHEEL_URL="https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.3.0/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl"
PINNED_WHEEL_SHA256="2bdb14a9c44f83ee6a766d88ed0d85b11390d6f5d65747e8dbe80a8e2d5d63e0"
VLLM_WHEEL="${VLLM_WHEEL:-$PINNED_WHEEL_URL}"
VLLM_WHEEL_SHA256="${VLLM_WHEEL_SHA256:-}"
[ -n "$VLLM_WHEEL_SHA256" ] || \
  { [ "$VLLM_WHEEL" = "$PINNED_WHEEL_URL" ] && VLLM_WHEEL_SHA256="$PINNED_WHEEL_SHA256"; } || true

# The checkpoint is pinned too: a mutable ref silently changes the weights
# under the published results.
PINNED_MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
PINNED_MODEL_REVISION="554ebba9b5f1b79dc11246341960360e6ef05ef4"
echo "    checkpoint: $PINNED_MODEL_REPO @ $PINNED_MODEL_REVISION"
echo "      hf download $PINNED_MODEL_REPO --revision $PINNED_MODEL_REVISION"

# ------------------------------------------------------------------- python
if command -v conda >/dev/null; then
  say "creating environment at $ENV_PREFIX (python $PYTHON_VERSION)"
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" >/dev/null
  PY="$ENV_PREFIX/bin/python"
else
  say "conda not found; using python -m venv at $ENV_PREFIX"
  python3 -m venv "$ENV_PREFIX"
  PY="$ENV_PREFIX/bin/python"
fi
"$PY" -m pip install --upgrade pip >/dev/null

say "installing pinned vLLM wheel"
# The README pins the engine and every published number was measured on it;
# this branch upgrades the pin to 1.3.0 (long-context D=256 prefill). Verify
# the version, and the digest when one is supplied.
# Fetch a remote wheel to a local file FIRST, so the digest is checked against
# the bytes that get installed. Hashing only local paths let a URL install
# unverified, which is the weakest link in a "reproducible" bootstrap.
WHEEL_LOCAL="$VLLM_WHEEL"
case "$VLLM_WHEEL" in
  http://*|https://*)
    WHEEL_LOCAL="$(mktemp -d)/$(basename "${VLLM_WHEEL%%\?*}")"
    say "downloading wheel"
    curl -fL --retry 3 -o "$WHEEL_LOCAL" "$VLLM_WHEEL" || die "wheel download failed: $VLLM_WHEEL" ;;
esac
[ -f "$WHEEL_LOCAL" ] || die "wheel not found: $WHEEL_LOCAL"

# Fail CLOSED: no digest means no install, unless explicitly waived.
if [ -z "${VLLM_WHEEL_SHA256:-}" ]; then
  [ "${ALLOW_UNVERIFIED_WHEEL:-0}" = 1 ] || die \
    "no SHA256 for $VLLM_WHEEL
   Pin one:   VLLM_WHEEL_SHA256=<digest> ...
   Or waive:  ALLOW_UNVERIFIED_WHEEL=1 ...  (not reproducible)"
  echo "    WARNING: installing an unverified wheel" >&2
else
  got=$(sha256sum "$WHEEL_LOCAL" 2>/dev/null | cut -d" " -f1) \
    || got=$(shasum -a 256 "$WHEEL_LOCAL" | cut -d" " -f1)
  [ "$got" = "$VLLM_WHEEL_SHA256" ] || die \
    "wheel digest mismatch
   expected $VLLM_WHEEL_SHA256
   got      $got"
  echo "    wheel sha256 verified"
fi
"$PY" -m pip install "$WHEEL_LOCAL"

# Trust installed metadata, not the filename: a renamed wheel passes any
# filename check ever written.
INSTALLED=$("$PY" - <<'EOF'
import importlib.metadata as md
for n in ("1cat-vllm", "1cat_vllm"):
    try:
        print(md.distribution(n).version); break
    except Exception:
        pass
else:
    print("")
EOF
)
[ "$INSTALLED" = "1.3.0" ] || { [ "${ALLOW_WHEEL_MISMATCH:-0}" = 1 ] && \
    echo "    WARNING: installed 1cat-vllm $INSTALLED, not 1.3.0" >&2; } || die \
  "installed 1cat-vllm is '${INSTALLED:-<none>}', not 1.3.0
   This branch targets 1Cat-vLLM 1.3.0. Override with
   ALLOW_WHEEL_MISMATCH=1 if you intend a different engine."
echo "    1cat-vllm $INSTALLED (from installed metadata)"

# tilelang must be pinned. The SM70 TileLang compile is fixed only in 0.1.10
# (upstream commit 0bb5cc4132); whatever pip resolves by default does not
# build on Volta, and the failure surfaces later inside GDN attention where it
# reads like a kernel bug rather than a dependency problem.
# apache-tvm-ffi MUST be pinned with it. tilelang 0.1.10 does not pin its own
# tvm-ffi, and current releases (0.1.13.post3 as of 2026-08) abort on import:
#   terminate called after throwing an instance of 'tvm::ffi::Error'
#     what(): TypeAttr `__ffi_repr__` is already registered for type index 132
# That is a hard crash, not a warning, and it only appears on a machine built
# today -- reference machines resolved 0.1.10 back when it was current. Found
# by the clean-clone run; without this pin the published command does not work.
say "pinning tilelang 0.1.10 + apache-tvm-ffi 0.1.10 (required for SM70)"
"$PY" -m pip install "tilelang==0.1.10" "apache-tvm-ffi==0.1.10"

# pip WILL print a red dependency-conflict block here: the 1cat-vllm wheel
# declares tilelang==0.1.9 and apache-tvm-ffi==0.1.9. Overriding it is
# deliberate -- 0.1.9 does not build on Volta. Say so, because otherwise every
# user reasonably assumes the install broke.
echo "    NOTE: pip reported a dependency conflict against the wheel's declared"
echo "          tilelang/apache-tvm-ffi 0.1.9 pins. That is EXPECTED: 0.1.9 does"
echo "          not build on SM70. The versions installed above are correct."

# Verify the import actually works. This previously ran unchecked, so the
# tvm-ffi abort above scrolled past and the bootstrap carried on with a broken
# environment.
"$PY" -c "import tilelang; print('    tilelang', tilelang.__version__)" \
  || die "tilelang failed to import -- the environment is broken.
   This is usually an apache-tvm-ffi version mismatch; expected 0.1.10.
   Check: $PY -m pip show apache-tvm-ffi"

SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -d "$SP/vllm" ] || die "vllm did not install into $SP"
say "site-packages: $SP"

# ------------------------------------------------------------- fork patches
# Copy tracked files over the installed package, keeping a .pre_bootstrap
# backup of whatever was there. Never hand-edit the installed file: the
# tracked copy in fork_patches/ is the reviewable source of truth.
say "deploying fork patches"
deploy() {  # $1 = tracked file, $2 = path under site-packages
  local src="$REPO_ROOT/fork_patches/$1" dst="$SP/$2"
  [ -f "$src" ] || die "missing tracked patch: $src"
  [ -f "$dst" ] || die "install path not found (wheel version mismatch?): $dst"
  [ -f "$dst.pre_bootstrap" ] || cp -p "$dst" "$dst.pre_bootstrap"
  cp -p "$src" "$dst"
  echo "    $1 -> $2"
}
deploy gdn_attn.py          vllm/v1/attention/backends/gdn_attn.py
deploy gpu_model_runner.py  vllm/v1/worker/gpu_model_runner.py
deploy marlin.py            vllm/model_executor/kernels/linear/nvfp4/marlin.py
deploy modelopt.py          vllm/model_executor/layers/quantization/modelopt.py
deploy torch_utils.py       vllm/utils/torch_utils.py
deploy attention.py         vllm/model_executor/layers/attention/attention.py
deploy custom_all_reduce.py vllm/distributed/device_communicators/custom_all_reduce.py
# sm70_native_round.py is deliberately NOT installed — experimental, inert.

# ------------------------------------------------------------------ kernels
KERNEL_SRC="$REPO_ROOT/kernels/skinny_kernels.cu"
[ -f "$KERNEL_SRC" ] || die "kernel source missing: $KERNEL_SRC"
say "warming the skinny-kernel JIT build (first build takes a few minutes)"
# Build through the DEPLOYED shim, not a hand-rolled ext.load(). The server
# loads name="skinny_nvfp4_v11" with -O3 --use_fast_math -lineinfo
# (fork_patches/marlin.py:150). torch.utils.cpp_extension keys its build
# directory on the name, and differing flags force a rebuild anyway, so
# building a differently-named extension here warms nothing and validates
# nothing -- the real nvcc build would then land inside the server's boot
# wait. Calling the shim also proves the fork-patch deploy above landed.
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}" \
VLLM_SKINNY_NVFP4_SRC="$KERNEL_SRC" TORCH_CUDA_ARCH_LIST=7.0 "$PY" - <<'PYEOF'
import sys
from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext
mod = _get_skinny_ext()
if mod is None:
    sys.exit("skinny extension failed to build (see nvcc output above)")
missing = [f for f in ("gemm_qpn2", "gemm_qpn8", "gemm_qpn8_mt2")
           if not hasattr(mod, f)]
if missing:
    sys.exit(f"kernel built but missing entry points: {missing}")
print("    kernels built and all entry points present")
PYEOF

cat <<EOF

Bootstrap complete.

  environment : $ENV_PREFIX
  kernels     : $KERNEL_SRC
  next        : bash scripts/serve-qwen38-native.sh <checkpoint-dir>

Set VLLM_SKINNY_NVFP4_SRC=$KERNEL_SRC in the serving environment
(serve-qwen38-native.sh does this for you).
EOF
