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

def probe(timeout_s=5.0) -> bool:
  """True if the device answers a PROBE/CFG_READ within the timeout."""
  sock_path = _server_sock_path()
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  sock.settimeout(timeout_s)
  try:
    sock.connect(sock_path)
    # PROBE vendor=0x10de base_class=3 (display): reply (n, bus:func) packed list
    sock.sendall(struct.pack("<BIIQQQ", 0, 0, 0, 0x10de, 0, 3))
    sock.recv(4)  # status + counts; a live device session answers quickly
    return True
  except (socket.timeout, ConnectionError, FileNotFoundError, struct.error): return False
  finally: sock.close()

def main():
  kill_stale_clients()
  kill_server()
  time.sleep(1.0)
  if probe():
    print("device answers - soft wedge cleared, DEV=NV should work")
    return 0
  # server may not be running yet; force a spawn via the driver path
  print("no answer from server - attempting fresh init to respawn it...")
  try:
    from tinygrad import Tensor
    print("probe init:", (Tensor.ones(2,2) @ Tensor.ones(2,2)).to("NV").numpy()[0,0])
    print("device recovered")
    return 0
  except Exception as e:
    print(f"hard wedge: {type(e).__name__}: {e}")
    print("the GPU is not answering config reads - replug the eGPU (or reload the dext with admin)")
    return 1

if __name__ == "__main__": sys.exit(main())
