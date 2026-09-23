# ******** quant linear: NV PQ2 gemv kernels over packed ggml weights ********

import functools
from typing import Callable
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.gguf import dequant_blocks
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

def _shuffle_fmt(device:str|tuple[str, ...]) -> str:
  if isinstance(device, tuple): device = device[0]
  # Metal shading language and CUDA both provide xor-shuffles with different names
  return "simd_shuffle_xor({0}, %d)" if device.split(":")[0] == "METAL" else "__shfl_xor_sync(0xffffffffu, {0}, %d)"

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
  tokens = prod(x.shape[:-1])
  assert isinstance(tokens, int)
  x = x.contiguous().reshape(tokens, layer.in_features)
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn:Callable = functools.partial(_nv_pq2_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features,
                                   tokens=tokens, shuffle_fmt=_shuffle_fmt(x.device))
  result = Tensor.custom_kernel(out, layer._pq2_codes, layer._pq2_scales, x, fxn=fxn)[0]
  result = result.reshape(*x.shape[:-1], layer.out_features)
  return result if layer.bias is None else result + layer.bias

class Linear(AMDLinear):
  """AMD custom-kernel Linear extended with an NV PQ2_0 gemv path."""
  _pq2_scales: Tensor
  _pq2_codes: Tensor
  def set_quantized(self, decoded:Tensor):
    if self.ggml_type is not None or self.in_features % 128: return
    graph = decoded.uop.toposort()
    raw = next((u for u in graph if u.op is Ops.SHRINK and u.dtype == dtypes.uint8
                and prod(u.shape) == self.out_features * (self.in_features // 128) * 34), None)
    if raw is None: return super().set_quantized(decoded)
    def unwrapped(u:UOp) -> UOp:
      while u.op in (Ops.RESHAPE, Ops.STAGE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)):
        u = u.src[0]
      return u
    expected = dequant_blocks(Tensor(raw), PQ2_0, self.out_features, self.in_features)
    if unwrapped(decoded.uop).key != unwrapped(expected.uop).key: return super().set_quantized(decoded)
    nb = self.in_features // 128
    blocks = Tensor(raw).reshape(self.out_features, nb, 34)
    self._pq2_scales = blocks[:, :, :2].bitcast(dtypes.uint16).contiguous()
    self._pq2_codes = blocks[:, :, 2:].bitcast(dtypes.uint32).contiguous()
    self.ggml_type = PQ2_0
  def __call__(self, x:Tensor) -> Tensor:
    if getattr(self, 'ggml_type', None) is None and nv_custom_kernels_supported(self.weight.device):
      self.set_quantized(self.weight)
    if getattr(self, 'ggml_type', None) == PQ2_0 and nv_custom_kernels_supported(self.weight.device):
      return pq2_gemv(self, x)
    return super().__call__(x)
