# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
- tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules.
- Remote NV (macOS eGPU via TinyGPU): never kill a running NV process mid-flight (SIGKILL/SIGTERM during command-buffer execution wedges the GPU - it stops answering config reads and only a physical replug recovers it). The llm CLI exits cleanly on SIGINT between tokens. Run `python3 extra/nv_remote_reset.py` to clear soft wedges or confirm a hard wedge needs a replug.
- Wedge severities: soft = stale client session on the single-client TinyGPU server (reset script fixes it). Hard = dead config reads (mailbox `ffffffff`) - needs a replug; killing the wedged dext (`org.tinygrad.tinygpu.driver2`) can hang IOKit/WindowServer and freeze the whole system - do NOT attempt a dext restart.
- `NV_SKIP_FINI=1` skips the GSP unload RPC at process exit - UNSAFE on the remote eGPU: a parked GSP holds host sysmem mappings that die with the process, and GSP touching freed memory wedges the card. `init_hw` detects a parked GSP and unloads it before re-booting. Wedge matrix (T1-T4 clean exits + idle signals were healthy; crashes/die-mid-flight wedge): the safe rule is all exits drain GPU work first - the llm CLI handles SIGINT+SIGTERM via a stop flag checked between tokens.
- The NV kernel compiler on macOS runs in docker (`ghcr.io/tinygrad/cuda-arm64`) - a cold-booted mac has no docker until `open -a Docker`; compile failures surface as BrokenPipeError in compile_server.
