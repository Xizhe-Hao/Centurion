# Centurion Cross-Device Collaborative Training (DiLoCo) — Step 1

Goal: get **Windows (PyTorch)** and **Mac (MLX)** to connect, exchange training
parameters, and train collaboratively. The paradigm is **DiLoCo**: each device
keeps a full copy of the model, trains K steps locally, then submits its
parameters to a coordinator that averages them in fp32 and broadcasts the new
global model back. Repeat.

Step 1 uses a **tiny two-layer MLP** + a synthetic classification task; the point
is to get the whole collaboration pipeline working. A later milestone upgrades
the model to GPT-2 small and adds a cross-framework numerical parity gate.

## Components

| Module | Role |
|---|---|
| `centurion_worker/model_spec.py` | Single source of truth: layer shapes, param names, fp32. Shared by both backends. |
| `centurion_worker/backend_base.py` | Abstract `Backend` (train / get_params / set_params / forward_logits). |
| `centurion_worker/pytorch_backend.py` | Windows implementation (auto-selects CUDA / CPU / MPS). |
| `centurion_worker/mlx_backend.py` | Mac MLX implementation; same architecture and param names as PyTorch. |
| `centurion_worker/tensor_codec.py` | safetensors (de)serialization + `average_states` (the DiLoCo average). |
| `centurion_worker/data.py` | Synthetic task (fixed teacher; every worker learns the same function). |
| `centurion_worker/diloco_client.py` | Worker main loop: pull global -> train locally -> submit -> wait for barrier. |
| `centurion_coord/server.py` | Coordinator (parameter server): collect world_size submissions -> average -> broadcast. |

The coordinator only handles numpy / safetensors and **never imports torch or
mlx**, so a PyTorch worker and an MLX worker are interchangeable to it. This
mirrors the registry role in the collaborator's `swarm_rebalacing_full` project;
its `FaultToleranceManager` can be plugged in later for fault tolerance.

## Protocol (round-based barrier)

```
global[0] = canonical initial weights
For each round r:
  worker:  GET  /global_params?round=r        # fetch global[r]
           set_params(global[r]); train K local steps
           POST /submit_params?worker_id=..&round=r   # body = its own params
  coord:   once world_size submissions arrive -> fp32 element-wise average
           -> global[r+1] -> current_round = r+1
  worker:  poll GET /round until current_round > r, then pull global[r+1]
```

## Setup

The repo's `.venv` was incomplete (no interpreter). Create a working venv first:

```powershell
# Windows, from the repo root
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch numpy safetensors fastapi uvicorn httpx pydantic pytest
```

## Running

### A) Single-machine local check (Windows)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_demo.ps1
```
Starts the coordinator + 2 PyTorch workers as separate processes; logs land in
`logs\`. Expect both workers' `global_eval_loss` to decrease each round and
converge to the same value.

### B) Real two-machine run: Windows <-> Mac (same LAN)

**Step 1 is achieved once 4a works.** Use PyTorch on both ends first to rule out
framework differences, then switch the Mac to MLX.

1. Find the LAN IP of the machine running the coordinator (Windows: `ipconfig`,
   Mac: `ipconfig getifaddr en0`). Assume `192.168.1.50`.

2. **Coordinator** (pick one machine, here Windows):
   ```powershell
   .venv\Scripts\python.exe -m centurion_coord.server --host 0.0.0.0 --port 9100 --world-size 2 --rounds 20
   ```
   `--host 0.0.0.0` is required so the other machine can reach it. Windows
   Firewall will prompt the first time — allow port 9100.

3. **Windows worker**:
   ```powershell
   .venv\Scripts\python.exe -m centurion_worker.diloco_client --coord-url http://192.168.1.50:9100 --worker-id win --framework pytorch --data-seed 0 --local-steps 20 --rounds 20
   ```

4. **Mac worker** — create a venv and install dependencies (on the Mac):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install numpy safetensors httpx
   ```
   **4a (do this first — verifies connectivity + averaging):** Mac on PyTorch too
   (`pip install torch`):
   ```bash
   python -m centurion_worker.diloco_client --coord-url http://192.168.1.50:9100 --worker-id mac --framework pytorch --data-seed 1 --local-steps 20 --rounds 20
   ```
   **4b (true cross-framework):** switch the Mac to MLX (`pip install mlx`):
   ```bash
   python -m centurion_worker.diloco_client --coord-url http://192.168.1.50:9100 --worker-id mac --framework mlx --data-seed 1 --local-steps 20 --rounds 20
   ```

   When both ends' `global_eval_loss` decrease together, Windows and Mac are
   successfully training collaboratively.

## Tests
```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

## Later milestones
- Upgrade the tiny model to GPT-2 small; add a cross-framework forward-parity
  gate (`max|delta logit| < 1e-3`).
- Integrate `swarm_rebalacing_full`'s registry + `FaultToleranceManager`: workers
  can join/leave mid-run; if a worker drops in a round, average the survivors and
  let the worker catch up to the latest global on restart.
- (stretch) Real pipeline parallelism: replace swarm's fake `stage_ops` with
  actual layer shards.
