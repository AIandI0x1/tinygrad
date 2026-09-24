#!/usr/bin/env python3
"""Reset helper for the wedged TinyGPU remote NV device (macOS eGPU path).

Two wedge severities:
- soft: the single-client TinyGPU server holds a dead client's session. Fixed by
  killing stale python clients + the server process; the next init respawns it.
- hard: the GPU won't answer config reads (dead GSP / dead channels after a
  kill mid-command-buffer). Only a physical replug (or a dext reload, which
  needs admin) recovers it. This script detects and reports that case.

Usage: python3 extra/nv_remote_reset.py
"""
import os, sys, signal, socket, struct, tempfile, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def _server_sock_path() -> str: return os.path.join(tempfile.gettempdir(), os.getenv("APL_REMOTE_SOCK", "tinygpu.sock"))

def kill_stale_clients():
  me = os.getpid()
  out = subprocess.run(["pgrep", "-fl", "python.*tinygrad|python.*-m tinygrad"], capture_output=True, text=True).stdout
  for line in out.splitlines():
    try: pid = int(line.split()[0])
    except ValueError: continue
    if pid != me:
      print(f"killing stale client {pid}: {line.strip()[:80]}")
      try: os.kill(pid, signal.SIGTERM)
      except OSError: pass

def kill_server():
  out = subprocess.run(["pgrep", "-fl", "TinyGPU server"], capture_output=True, text=True).stdout
  for line in out.splitlines():
    print(f"killing server {line.split()[0]}")
    try: os.kill(int(line.split()[0]), signal.SIGTERM)
    except OSError: pass

def dext_spin() -> tuple[int|None, float]:
  """A wedged dext burns ~100% CPU polling dead registers. Returns (pid, cpu%)."""
  out = subprocess.run(["pgrep", "-fl", "tinygpu.driver2"], capture_output=True, text=True).stdout
  worst, wp = 0.0, None
  for line in out.splitlines():
    try: pid = int(line.split()[0])
    except (ValueError, IndexError): continue
    top = subprocess.run(["ps", "-o", "%cpu=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    try: cpu = float(top)
    except ValueError: continue
    if cpu > worst: worst, wp = cpu, pid
  return wp, worst

def probe(timeout_s=5.0) -> bool:
  """True if the device answers a PROBE/CFG_READ within the timeout."""
  sock_path = _server_sock_path()
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.settimeout(timeout_s)
  try:
    sock.connect(sock_path)
    # CFG_READ offset 0 (vendor/device id): requires the device to actually answer,
    # unlike PROBE which can be served from cached state. 0xffffffff means wedged.
    sock.sendall(struct.pack("<BIIQQQ", 3, 0, 0, 0, 4, 0))  # RemoteCmd.CFG_READ
    resp = sock.recv(17)  # REMOTE_RESP <BQQ: status, r0, r1
    if len(resp) >= 9:
      status, r0 = resp[0], struct.unpack("<Q", resp[1:9])[0]
      return status == 0 and r0 not in (0xffffffff, 0)
    return True
  except (socket.timeout, ConnectionError, FileNotFoundError, struct.error): return False
  finally: sock.close()

def try_device_reset() -> bool:
  """Send RemoteCmd.RESET on the live server socket - if the dext maps it to a PCI
  function-level reset this recovers a wedged card without a physical replug."""
  try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect(_server_sock_path())
    sock.sendall(struct.pack("<BIIQQQ", 5, 0, 0, 0, 0, 0))  # RemoteCmd.RESET
    sock.recv(17)
    sock.close()
    return True
  except (socket.timeout, ConnectionError, FileNotFoundError, struct.error): return False

def main():
  if "--check" in sys.argv:
    return 0 if probe() else 1
  # first try a device reset while the server still owns the card - if it answers,
  # an FLR can recover a wedge without touching sessions or hardware
  if not probe():
    print("probe failed - trying remote device reset before killing sessions...")
    if try_device_reset():
      time.sleep(3.0)
      if probe():
        print("device recovered via remote reset")
        return 0
  kill_stale_clients()
  kill_server()
  time.sleep(1.0)
  if probe():
    print("device answers - soft wedge cleared, DEV=NV should work")
    return 0
  dpid, dcpu = dext_spin()
  if dpid is not None and dcpu > 50.0:
    print(f"dext {dpid} is spinning at {dcpu:.0f}% CPU - the device is wedged below the driver")
  # server may not be running yet; try a fresh init in a subprocess (init polls
  # dead registers without a timeout, so it must not hang this script)
  print("no answer from server - attempting fresh init to respawn it...")
  env = dict(os.environ, DEV="NV")
  try:
    out = subprocess.run([sys.executable, "-c",
      "from tinygrad import Tensor; print((Tensor.ones(4,4,device='NV')@Tensor.ones(4,4,device='NV')).numpy()[0,0])"],
      capture_output=True, text=True, timeout=60, env=env)
    if "4.0" in out.stdout:
      print("device recovered")
      return 0
    print(f"hard wedge: {out.stderr.strip().splitlines()[-1] if out.stderr else 'no output'}")
  except subprocess.TimeoutExpired:
    print("hard wedge: init hung (60s timeout)")
  # try the remote RESET rpc: if the dext implements it as a function-level reset it
  # recovers the card without a physical replug. Harmless if unsupported.
  print("attempting remote device reset...")
  try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect(_server_sock_path())
    sock.sendall(struct.pack("<BIIQQQ", 5, 0, 0, 0, 0, 0))  # RemoteCmd.RESET
    sock.recv(17)
    sock.close()
    time.sleep(3.0)
    out = subprocess.run([sys.executable, "-c",
      "from tinygrad import Tensor; print((Tensor.ones(4,4,device='NV')@Tensor.ones(4,4,device='NV')).numpy()[0,0])"],
      capture_output=True, text=True, timeout=60, env=env)
    if "4.0" in out.stdout:
      print("device recovered after remote reset")
      return 0
  except (socket.timeout, ConnectionError, FileNotFoundError, subprocess.TimeoutExpired): pass
  # NOTE: killing the wedged dext is NOT safe - it can hang IOKit/WindowServer and
  # freeze the whole system. A hard wedge really does need a physical replug.
  print("the GPU is not answering config reads - replug the eGPU (or reload the dext with admin)")
  return 1

if __name__ == "__main__": sys.exit(main())
