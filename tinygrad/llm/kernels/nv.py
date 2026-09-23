# ******** quant linear: NV PQ2 gemv kernels over packed ggml weights ********

import functools
from typing import Callable, Any, cast
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, getenv, DEBUG
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.amd import Linear as AMDLinear

PQ2_0 = 142
WARP_SIZE = 32

@functools.cache
def nv_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  # opt-in until the gemv beats the scheduler's fused dequant+matmul on the real remote path
  if getenv("DISABLE_NV_KERNELS") or not getenv("NV_PQ2_GEMV"): return False
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "NV": return False
  with Context(ALLOW_DEVICE_USAGE=1): return Device["NV"].device == "NV"

def _shuffle_fmt(device:str|tuple[str, ...]|None) -> str:
  if isinstance(device, tuple): device = device[0]
  # Metal shading language and CUDA both provide xor-shuffles with different names
  return "simd_shuffle_xor({0}, %d)" if device is not None and device.split(":")[0] == "METAL" else "__shfl_xor_sync(0xffffffffu, {0}, %d)"

def nv_warp_reduce(val:UOp, fmt:str, maximum:bool=False) -> UOp:
  for offset in (16, 8, 4, 2, 1):
    other = UOp(Ops.CUSTOM, src=(val,), arg=(fmt % offset, dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _nv_pq2_gemv_kernel(out:UOp, codes:UOp, scales:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int, shuffle_fmt:str) -> UOp:
  # one warp per (token, output row): lanes own strided u32 words of the packed row.
  # PQ2_0 layout per 128-elem block: 8 u32 words, 16 two-bit codes each, values {0,1,2,3} -> {-1,0,1,2} * fp16 scale.
  nb = in_features // 128
  nwords = nb * 8
  assert nwords % WARP_SIZE == 0, f"in_features {in_features} gives {nwords} words not divisible by warp size"
  per = nwords // WARP_SIZE
  codes, scales = codes.reshape((out_features, nwords)), scales.reshape((out_features, nb))
  x = x.reshape((tokens, in_features))
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    wi = i*WARP_SIZE + lane
    b, wsub = wi // 8, wi % 8
    word = codes[out_row, wi].load()
    scale = scales[out_row, b].load().cast(dtypes.uint16).bitcast(dtypes.float16).float()
    inner = UOp.const(0, dtypes.float32)
    base = b*128 + wsub*16
    for j in range(16):
      code = (word >> (j*2)) & 0x3
      xv = x[token, base+j].load()
      inner = inner + (code.cast(dtypes.float32) - 1) * (xv.cast(dtypes.float32) if xv.dtype != dtypes.float32 else xv)
    acc = acc + scale * inner
  total = nv_warp_reduce(acc, shuffle_fmt)
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="pq2_gemv", opts_to_apply=()))

def pq2_gemv(layer:'Linear', x:Tensor) -> Tensor:
  out_shape, tokens = (*x.shape[:-1], layer.out_features), prod(x.shape[:-1])
  assert isinstance(tokens, int)
  x = x.contiguous().reshape(tokens, layer.in_features)
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn:Callable = functools.partial(_nv_pq2_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features,
                                   tokens=tokens, shuffle_fmt=_shuffle_fmt(x.device))
  result = Tensor.custom_kernel(out, layer._pq2_codes, layer._pq2_scales, x, fxn=fxn)[0].reshape(out_shape)
  return result if layer.bias is None else result + layer.bias

def _nv_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int, shuffle_fmt:str) -> UOp:
  # one warp per (token, output row): lanes stride the dense fp16 row
  per = in_features // WARP_SIZE
  assert per * WARP_SIZE == in_features, f"in_features {in_features} not divisible by warp size"
  w, x = w.reshape((out_features, in_features)), x.reshape((tokens, in_features))
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    e = i*WARP_SIZE + lane
    acc = acc + w[out_row, e].load().float() * x[token, e].load().float()
  total = nv_warp_reduce(acc, shuffle_fmt)
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="f16_gemv", opts_to_apply=()))

def f16_gemv(layer:AMDLinear, x:Tensor) -> Tensor:
  out_shape, tokens = (*x.shape[:-1], layer.out_features), prod(x.shape[:-1])
  assert isinstance(tokens, int)
  x = x.contiguous().reshape(tokens, layer.in_features)
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn:Callable = functools.partial(_nv_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features,
                                   tokens=tokens, shuffle_fmt=_shuffle_fmt(x.device))
  result = Tensor.custom_kernel(out, layer.weight.reshape(-1), x, fxn=fxn)[0].reshape(out_shape)
  return result if layer.bias is None else result + layer.bias

def _nv_pq2_gemv_raw_kernel(out:UOp, codes:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int, shuffle_fmt:str) -> UOp:
  # warp-per-row on the *packed* 34B-block layout: u16 scale at block start, then 8 u32 code
  # words at a 2-byte offset. words are assembled from two aligned u32 loads + parity select -
  # no repack needed, so the raw weight buffer is the only device footprint.
  nb = in_features // 128
  assert nb * 128 == in_features and (nb * 34) % 4 == 0, f"in_features {in_features} gives unaligned row"
  nwords, rowb32 = nb * 8, nb * 34 // 4
  codes, x = codes.reshape((out_features, rowb32)), x.reshape((tokens, in_features))
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  acc = UOp.const(0, dtypes.float32)
  per = nwords // WARP_SIZE
  assert per * WARP_SIZE == nwords, f"{nwords} words not divisible by warp size"
  for i in range(per):
    wi = i*WARP_SIZE + lane
    b, wsub = wi // 8, wi % 8
    # scale: the fp16 u16 sits at byte offset 34b - grab it from the covering aligned u32
    # (34b mod 4 is 0 or 2 -> shift 0 or 16, both branchless)
    su32 = codes[out_row, (34*b) >> 2].load()
    scale = ((su32 >> (8*(34*b & 3))) & 0xFFFF).cast(dtypes.uint16).bitcast(dtypes.float16).float()
    # word: bytes [off, off+4) where off = 34b+2+4*wsub -> assemble from the covering aligned u32s.
    # off mod 4 is 0 (odd block) or 2 (even block); select instead of an illegal <<32.
    off, w32 = 34*b + 2 + 4*wsub, (34*b + 2 + 4*wsub) >> 2
    rem = off & 3
    w0 = codes[out_row, w32].load()
    w1 = codes[out_row, w32 + 1].load()
    word = rem.eq(0).where(w0, (w0 >> (8*rem)) | (w1 << (8*(4-rem))))
    inner = UOp.const(0, dtypes.float32)
    base = b*128 + wsub*16
    for j in range(16):
      code = (word >> (j*2)) & 0x3
      xv = x[token, base+j].load()
      inner = inner + (code.cast(dtypes.float32) - 1) * (xv.cast(dtypes.float32) if xv.dtype != dtypes.float32 else xv)
    acc = acc + scale * inner
  total = nv_warp_reduce(acc, shuffle_fmt)
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="pq2_gemv", opts_to_apply=()))

def pq2_gemv_raw(layer:'Linear', x:Tensor) -> Tensor:
  out_shape, tokens = (*x.shape[:-1], layer.out_features), prod(x.shape[:-1])
  assert isinstance(tokens, int)
  x = x.contiguous().reshape(tokens, layer.in_features)
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn:Callable = functools.partial(_nv_pq2_gemv_raw_kernel, in_features=layer.in_features, out_features=layer.out_features,
                                   tokens=tokens, shuffle_fmt=_shuffle_fmt(x.device))
  result = Tensor.custom_kernel(out, layer._pq2_raw, x, fxn=fxn)[0].reshape(out_shape)
  return result if layer.bias is None else result + layer.bias

def _nv_pq2_gemv_moe_kernel(out:UOp, codes:UOp, sel:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int,
                            tokens:int, k:int, x_per:bool, shuffle_fmt:str) -> UOp:
  # MoE variant: one warp per (token*expert-slot, output row). sel[token] picks the expert's
  # packed rows; x is indexed per (b,t,k) or shared per (b,t) when the router broadcast it.
  nb = in_features // 128
  assert nb * 128 == in_features and (nb * 34) % 4 == 0, f"in_features {in_features} gives unaligned row"
  nwords, rowb32 = nb * 8, nb * 34 // 4
  n_exp = codes.shape[0]
  codes, x = codes.reshape((n_exp, out_features, rowb32)), x.reshape((tokens if x_per else tokens // k, in_features))
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  e, xi = sel[token].load(), (token if x_per else token // k)
  acc = UOp.const(0, dtypes.float32)
  per = nwords // WARP_SIZE
  for i in range(per):
    wi = i*WARP_SIZE + lane
    b, wsub = wi // 8, wi % 8
    su32 = codes[e, out_row, (34*b) >> 2].load()
    scale = ((su32 >> (8*(34*b & 3))) & 0xFFFF).cast(dtypes.uint16).bitcast(dtypes.float16).float()
    off, w32 = 34*b + 2 + 4*wsub, (34*b + 2 + 4*wsub) >> 2
    rem = off & 3
    w0 = codes[e, out_row, w32].load()
    w1 = codes[e, out_row, w32 + 1].load()
    word = rem.eq(0).where(w0, (w0 >> (8*rem)) | (w1 << (8*(4-rem))))
    inner = UOp.const(0, dtypes.float32)
    base = b*128 + wsub*16
    for j in range(16):
      code = (word >> (j*2)) & 0x3
      xv = x[xi, base+j].load()
      inner = inner + (code.cast(dtypes.float32) - 1) * (xv.cast(dtypes.float32) if xv.dtype != dtypes.float32 else xv)
    acc = acc + scale * inner
  total = nv_warp_reduce(acc, shuffle_fmt)
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="pq2_gemv_moe", opts_to_apply=()))

def pq2_gemv_moe(ew:Any, sel:Tensor, x:Tensor) -> Tensor:
  B, T, k = sel.shape
  in_f, out_f = ew._pq2_in, ew._pq2_out
  shared = x.shape[2] == 1
  xs = x.contiguous().reshape(B*T if shared else B*T*k, in_f)
  out = Tensor.empty(B*T*k, out_f, dtype=dtypes.float32, device=x.device)
  fxn:Callable = functools.partial(_nv_pq2_gemv_moe_kernel, in_features=in_f, out_features=out_f,
                                   tokens=cast(int, B*T*k), k=cast(int, k), x_per=not shared, shuffle_fmt=_shuffle_fmt(x.device))
  return Tensor.custom_kernel(out, ew._pq2_raw3, sel.contiguous().reshape(-1), xs, fxn=fxn)[0].reshape(B, T, k, out_f)

def tag_pq2_expert_raw(ew:Any, raw:Tensor, name:str="") -> None:
  """Tag a lazy PQ2_0 ExpertWeights to run the expert-indexed gemv on packed 34B-block rows."""
  E, out_f, in_f = ew.weight.shape
  nb = in_f // 128
  if in_f % 128 or (nb * 34) % 4 or (nb * 8) % WARP_SIZE: return
  ew._pq2_E, ew._pq2_in, ew._pq2_out = E, in_f, out_f
  ew._pq2_raw3 = raw.reshape(E * out_f, nb * 34).to(ew.weight.device).contiguous().bitcast(dtypes.uint32).realize()
  if nv_custom_kernels_supported(ew.weight.device):
    ew.weight = Tensor.zeros(1, dtype=dtypes.float16, device=ew.weight.device)

def tag_pq2_linear_raw(lin:'Linear', raw:Tensor, name:str="", dest:str|None=None) -> None:
  """Tag a lazy PQ2_0 linear to run the fused gemv directly on its packed 34B-block buffer:
  uploads the raw bytes once (u32 view) and drops the lazy weight - no repack.
  dest='CPU:APL' keeps the packed data in GPU-mapped sysmem instead of VRAM - slower per byte
  over the link, but it still beats the fused lazy-dequant kernel and frees VRAM."""
  nb = lin.in_features // 128
  if lin.in_features % 128 or (nb * 34) % 4 or (nb * 8) % WARP_SIZE: return
  dest = dest or lin.weight.device
  lin._pq2_raw = raw.reshape(lin.out_features, nb * 34).to(dest).contiguous().bitcast(dtypes.uint32).realize()
  lin.ggml_type = PQ2_0
  if nv_custom_kernels_supported(lin.weight.device):
    lin.weight = Tensor.zeros(1, dtype=dtypes.float16, device=lin.weight.device)

def tag_pq2_linear(lin:'Linear', raw:Tensor, name:str="") -> None:
  """Repack a PQ2_0 raw-block weight into aligned (scales u16, codes u32) tensors and tag
  the Linear so __call__ routes to the fused gemv instead of the lazy-dequant matmul.
  The repack runs on the host: the 34-byte block layout puts code words at a 2-byte
  offset, and realizing the bitcast view on-device emits misaligned u32 loads that
  fault the remote GPU."""
  if lin.in_features % 128: return
  import numpy as np
  nb = lin.in_features // 128
  rn = raw.reshape(lin.out_features, nb, 34).numpy()
  dev = lin.weight.device
  lin._pq2_scales = Tensor(rn[:, :, :2].copy().view(np.uint16), device=dev)
  lin._pq2_codes = Tensor(rn[:, :, 2:].copy().view('<u4'), device=dev)
  # upload once: unrealized numpy-backed tensors would re-copy every jit call
  if DEBUG >= 2:
    from tinygrad.device import GlobalCounters
    pd = {d: round(v / (1 << 20)) for d, v in GlobalCounters.mem_used_per_device.items()}
    print(f"  tag_pq2 {name} {lin.out_features}x{lin.in_features}: scales {lin._pq2_scales.nbytes()>>20}MB "
          f"codes {lin._pq2_codes.nbytes()>>20}MB per-device {pd}", flush=True)
  lin._pq2_scales.realize()
  lin._pq2_codes.realize()
  lin.ggml_type = PQ2_0
  # the repack supersedes the packed storage; drop the lazy weight so its buffer frees
  if nv_custom_kernels_supported(lin.weight.device):
    lin.weight = Tensor.zeros(1, dtype=dtypes.float16, device=lin.weight.device)

def nv_forward(lin:'Linear', x:Tensor) -> Tensor|None:
  """Dispatch decode matvecs to the custom NV kernels when the device supports them."""
  if not nv_custom_kernels_supported(lin.weight.device): return None
  if getattr(lin, 'ggml_type', None) == PQ2_0:
    gemv = pq2_gemv_raw if hasattr(lin, '_pq2_raw') else pq2_gemv
    if isinstance(x.numel(), int): return gemv(lin, x)
    return gemv(lin, x.pad_to(x.max_shape)).shrink(tuple((0, s) for s in (*x.shape[:-1], lin.out_features)))
  if lin.in_features % WARP_SIZE: return None
  w = lin.weight
  if w.uop.base.op is Ops.BUFFER and w.dtype == dtypes.float16 and isinstance(x.numel(), int) and x.numel() == lin.in_features:
    return f16_gemv(lin, x)
  return None

pq2_forward = nv_forward # back-compat name

class Linear(AMDLinear):
  """AMD custom-kernel Linear extended with an NV PQ2_0 gemv path."""
  _pq2_scales: Tensor
  _pq2_codes: Tensor
  _pq2_raw: Tensor
  def __call__(self, x:Tensor) -> Tensor:
    if (out := nv_forward(self, x)) is not None: return out
    return super().__call__(x)
