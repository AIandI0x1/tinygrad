"""Fast NV PQ2 kernel checks - no model load. Run: DEV=NV python3 extra/nv_quicktest.py

Covers pq2_gemv_raw correctness vs a numpy reference and per-call timing on a
few representative shapes. Use for kernel edits; full-model changes still need
a real benchmark run.
"""
import time, sys
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.nv import Linear, pq2_gemv_raw, PQ2_0

def make_raw(rows:int, cols:int, seed:int=0):
  rng = np.random.default_rng(seed)
  nb = cols // 128
  raw = np.zeros((rows, nb, 34), np.uint8)
  raw[:, :, 0:2] = (rng.standard_normal((rows, nb, 1)) * 0.01).astype(np.float16).view(np.uint8)[:, :, :2]
  raw[:, :, 2:] = rng.integers(0, 256, (rows, nb, 32), np.uint8)
  return raw.reshape(rows, nb * 34)

def ref_pq2(raw:np.ndarray, x:np.ndarray) -> np.ndarray:
  """numpy reference: per 34B block, u16 scale then 8 u32 code words, 16 x 2-bit codes per word, val-1."""
  rows, rowb = raw.shape
  nb = rowb // 34
  r = raw.reshape(rows, nb, 34)
  scale = r[:, :, :2].copy().view(np.float16).astype(np.float32)          # (rows, nb)
  words = r[:, :, 2:].astype(np.uint8).reshape(rows, nb, 8, 4).copy().view('<u4')[:, :, :, 0]  # (rows, nb, 8) u32
  out = np.zeros((rows,), np.float32)
  for b in range(nb):
    for j in range(8):
      v = ((words[:, b, j][:, None] >> (np.arange(16)[None, :] * 2)) & 3).astype(np.float32) - 1
      out += scale[:, b] * v @ x[b * 128 + j * 16:b * 128 + j * 16 + 16].astype(np.float32)
  return out

def main():
  dev = 'NV'
  for rows, cols in ((6144, 5120), (5120, 17408), (64, 512)):
    if cols % 128 or (cols // 128 * 34) % 4: continue
    raw = make_raw(rows, cols, seed=rows)
    lin = Linear(cols, rows, bias=False)
    lin._pq2_raw = Tensor(raw, device=dev).contiguous().bitcast(dtypes.uint32).realize()
    lin.ggml_type = PQ2_0
    x = np.random.default_rng(1).standard_normal(cols).astype(np.float16)
    xt = Tensor(x, device=dev)
    got = pq2_gemv_raw(lin, xt).numpy()
    want = ref_pq2(raw, x)
    err = float(np.abs(got - want).max())
    rel = err / float(np.abs(want).max() + 1e-9)
    ts = []
    for _ in range(3):
      t = time.time()
      pq2_gemv_raw(lin, xt).numpy()
      ts.append(time.time() - t)
    status = 'OK' if rel < 1e-4 else 'FAIL'
    print(f'{status} {rows}x{cols}: max_err={err:.2e} rel={rel:.2e} per-call={min(ts)*1e3:.1f}ms')
    if rel >= 1e-4: sys.exit(1)
  print('all pass')

if __name__ == '__main__': main()
