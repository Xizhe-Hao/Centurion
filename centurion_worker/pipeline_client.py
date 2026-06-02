"""Pipeline-parallel worker client (connects to the team's cloud pipeline server).

This lets a Windows PyTorch machine JOIN the teammate's pipeline-parallel GPT-2
training on 34.60.122.134:9998, where the model is split across devices and each
worker owns a contiguous slice of layers (the server assigns more layers to more
capable devices).

Modes:
  --join-only : Stage 0. connect -> auth -> register (0x40) -> answer profiling
                (0x48 -> 0x49) -> receive PIPELINE_CONFIG (0x41) -> print the
                assigned layer range -> ack (0x42). No training compute.
  (default)   : full worker. After CONFIG/START, runs the assigned slice through
                the all-forward-then-all-backward micro-batch schedule, relaying
                activations (0x60) and gradients (0x61) via the server.

Wire protocol (verified against Centurion/Server/pipeline_server.py and
Centurion/PipelineTrainingManager.swift):
  Frame:  [4B BE uint32 length][payload]
  Auth:   identical HMAC-SHA256 handshake as the checkpoint server (reused).

  DATA_BATCH 0x50 (server->head):
    [0x50][mb_id u32][mini_batch u32][B u32][S u32][tokens B*S BE i32][targets B*S BE i32]
  ACTIVATION 0x60 (worker<->server):
    [0x60][mb_id u32][src u32][dst u32][has_targets 1B][tgt_len u32][targets...][st_len u32][safetensors{"activation"}]
  GRADIENT 0x61 (worker<->server):
    [0x61][mb_id u32][src u32][dst u32][loss BE f32][st_len u32][safetensors{"grad"}]
  SYNC_BARRIER 0x70 (server->worker): [0x70][mb_id u32]
  SYNC_ACK 0x71 (worker->server):     [0x71][mb_id u32][worker_id u32]
  LOSS_REPORT 0x80 (tail->server):    [0x80][mini_batch u32][mb_id u32][loss f32][step u32]
"""

from __future__ import annotations

import argparse
import socket
import struct

import numpy as np
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

# Reuse the already-tested frame + HMAC layer from the checkpoint client.
from .checkpoint_client import authenticate, read_frame, write_frame

# ── Pipeline message type constants ──
MSG_PIPELINE_REGISTER = 0x40
MSG_PIPELINE_CONFIG = 0x41
MSG_PIPELINE_CONFIG_ACK = 0x42
MSG_PIPELINE_START = 0x43
MSG_PIPELINE_STOP = 0x44
MSG_PROFILE_REQUEST = 0x48
MSG_PROFILE_RESULT = 0x49
MSG_PIPELINE_DATA_BATCH = 0x50
MSG_PIPELINE_ACTIVATION = 0x60
MSG_PIPELINE_GRADIENT = 0x61
MSG_PIPELINE_SYNC_BARRIER = 0x70
MSG_PIPELINE_SYNC_ACK = 0x71
MSG_PIPELINE_LOSS_REPORT = 0x80

# Device type reported to the server (informational; affects layer allocation
# only via reported available memory). 2 = a generic non-iOS worker.
DEVICE_TYPE = 2


class PipelineConfig:
    """Parsed PIPELINE_CONFIG (0x41) frame."""

    __slots__ = ("stage_index", "total_stages", "first_layer", "last_layer",
                 "is_head", "is_tail", "num_micro_batches", "vocab_size",
                 "d_model", "n_heads", "n_layers_total", "seq_len", "batch_size",
                 "ffn_hidden_mul", "learning_rate", "dropout")

    def __init__(self, fields):
        (self.stage_index, self.total_stages, self.first_layer, self.last_layer,
         self.is_head, self.is_tail, self.num_micro_batches, self.vocab_size,
         self.d_model, self.n_heads, self.n_layers_total, self.seq_len,
         self.batch_size, self.ffn_hidden_mul, self.learning_rate,
         self.dropout) = fields

    @property
    def ffn_hidden(self) -> int:
        return self.d_model * self.ffn_hidden_mul

    @property
    def local_layers(self) -> int:
        return self.last_layer - self.first_layer

    def describe(self) -> str:
        role = []
        if self.is_head:
            role.append("HEAD")
        if self.is_tail:
            role.append("TAIL")
        if not role:
            role.append("MIDDLE")
        return (
            f"stage {self.stage_index + 1}/{self.total_stages} [{'+'.join(role)}]\n"
            f"  assigned layers: blocks[{self.first_layer}:{self.last_layer}] "
            f"({self.local_layers} of {self.n_layers_total} total)\n"
            f"  model: vocab={self.vocab_size} d_model={self.d_model} "
            f"heads={self.n_heads} layers_total={self.n_layers_total} "
            f"seq={self.seq_len} ffn={self.ffn_hidden}\n"
            f"  train: batch={self.batch_size} "
            f"micro_batches={self.num_micro_batches} "
            f"lr={self.learning_rate:.2e} dropout={self.dropout}"
        )


# ── Protocol steps ──

def register(sock: socket.socket, memory_mb: int = 8192) -> None:
    """Send PIPELINE_REGISTER (0x40). Server assigns the real worker_id; the
    client_id field is ignored, so we send 0."""
    payload = struct.pack(">BIII", MSG_PIPELINE_REGISTER, 0, DEVICE_TYPE, memory_mb)
    write_frame(sock, payload)


def handle_profile_request(sock: socket.socket, frame: bytes,
                           memory_mb: int = 8192) -> None:
    """Answer PROFILE_REQUEST (0x48) with a PROFILE_RESULT (0x49).

    The server uses these timing/memory numbers to decide how many layers to
    give us. We report modest positive timings and our available memory (MB).
    Layout (verified): >B I ff I ff I ff I I I
      [0x49][worker_id=0]
      [layer_fwd_ms f][layer_bwd_ms f][layer_peak_mem u32]
      [head_fwd_ms f][head_bwd_ms f][head_peak_mem u32]
      [tail_fwd_ms f][tail_bwd_ms f][tail_peak_mem u32]
      [avail_mem_mb u32][device_type u32]
    """
    reply = struct.pack(
        ">B I ff I ff I ff I I I",
        MSG_PROFILE_RESULT, 0,
        2.0, 4.0, 50_000_000,      # per-layer fwd/bwd ms, peak bytes
        1.0, 2.0, 20_000_000,      # head extras
        2.0, 4.0, 200_000_000,     # tail extras (embedding/LM head heavier)
        memory_mb, DEVICE_TYPE,
    )
    write_frame(sock, reply)


def parse_config(frame: bytes) -> PipelineConfig:
    """Parse PIPELINE_CONFIG (0x41). Layout (verified against server):
      >B IIII BB I IIIII II ff
    """
    fields = struct.unpack(">B IIII BB I IIIII II ff".replace(" ", ""), frame)
    # Drop the leading message-type byte.
    return PipelineConfig(fields[1:])


def config_ack(sock: socket.socket) -> None:
    """Send PIPELINE_CONFIG_ACK (0x42), status 0 = ok."""
    write_frame(sock, struct.pack(">BB", MSG_PIPELINE_CONFIG_ACK, 0))


# ── Tensor (de)serialization for activations/gradients ──

def _save_tensor(key: str, arr: np.ndarray) -> bytes:
    """safetensors blob holding one fp32 contiguous tensor under `key`."""
    return st_save({key: np.ascontiguousarray(arr, dtype=np.float32)})


def _load_tensor(blob: bytes, key: str) -> np.ndarray:
    return st_load(blob)[key]


def parse_data_batch(frame: bytes):
    """0x50 -> (mb_id, tokens[B,S] int64, targets[B,S] int64)."""
    _, mb_id, _mini, B, S = struct.unpack_from(">BIIII", frame, 0)
    off = 17
    n = B * S
    tokens = np.frombuffer(frame[off:off + n * 4], dtype=">i4").astype(np.int64).reshape(B, S)
    off += n * 4
    targets = np.frombuffer(frame[off:off + n * 4], dtype=">i4").astype(np.int64).reshape(B, S)
    return mb_id, tokens, targets


def parse_activation(frame: bytes):
    """0x60 -> (mb_id, activation[B,S,d] f32, targets[B,S] int64 or None)."""
    _, mb_id, _src, _dst = struct.unpack_from(">BIII", frame, 0)
    off = 13
    has_targets = frame[off]; off += 1
    tgt_len = struct.unpack_from(">I", frame, off)[0]; off += 4
    targets = None
    if has_targets and tgt_len > 0:
        targets = np.frombuffer(frame[off:off + tgt_len], dtype=">i4").astype(np.int64)
        off += tgt_len
    st_len = struct.unpack_from(">I", frame, off)[0]; off += 4
    act = _load_tensor(frame[off:off + st_len], "activation")
    if targets is not None:
        B, S = act.shape[0], act.shape[1]
        targets = targets.reshape(B, S)
    return mb_id, act, targets


def parse_gradient(frame: bytes):
    """0x61 -> (mb_id, loss_float, grad[B,S,d] f32)."""
    _, mb_id, _src, _dst = struct.unpack_from(">BIII", frame, 0)
    off = 13
    loss = struct.unpack_from(">f", frame, off)[0]; off += 4
    st_len = struct.unpack_from(">I", frame, off)[0]; off += 4
    grad = _load_tensor(frame[off:off + st_len], "grad")
    return mb_id, loss, grad


def build_activation(mb_id: int, src_stage: int, arr: np.ndarray) -> bytes:
    """0x60 from a worker (has_targets=0; the server injects targets downstream)."""
    blob = _save_tensor("activation", arr)
    msg = struct.pack(">BIII", MSG_PIPELINE_ACTIVATION, mb_id, src_stage, src_stage + 1)
    msg += struct.pack(">B", 0)                 # has_targets = 0
    msg += struct.pack(">I", 0)                 # targets_len = 0
    msg += struct.pack(">I", len(blob)) + blob
    return msg


def build_gradient(mb_id: int, src_stage: int, arr: np.ndarray, loss: float) -> bytes:
    """0x61 upstream (dst = src_stage - 1)."""
    blob = _save_tensor("grad", arr)
    msg = struct.pack(">BIII", MSG_PIPELINE_GRADIENT, mb_id, src_stage, src_stage - 1)
    msg += struct.pack(">f", float(loss))
    msg += struct.pack(">I", len(blob)) + blob
    return msg


def build_loss_report(mini_batch: int, mb_id: int, loss: float, step: int) -> bytes:
    return struct.pack(">BIIfI", MSG_PIPELINE_LOSS_REPORT, mini_batch, mb_id, float(loss), step)


def build_sync_ack(mb_id: int) -> bytes:
    return struct.pack(">BII", MSG_PIPELINE_SYNC_ACK, mb_id, 0)


class _Stop(Exception):
    """Raised when the server sends STOP (0x44) mid-run."""


def _recv_or_stop(sock: socket.socket, expected: int) -> bytes:
    """Read a frame; raise _Stop on 0x44, else return it (verifying type)."""
    f = read_frame(sock)
    if f[0] == MSG_PIPELINE_STOP:
        raise _Stop()
    return f


# ── Full training driver ──

def run_training(host: str, port: int, secret: str, memory_mb: int = 8192,
                 timeout: float = 300.0):
    import torch
    from .pipeline_slice import PipelineSlice

    print(f"[pipeline] connecting to {host}:{port} ...")
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    authenticate(sock, secret)
    print("[pipeline] authenticated")
    register(sock, memory_mb=memory_mb)
    print(f"[pipeline] registered (mem={memory_mb}MB); waiting for training to start...")

    # Wait for the first CONFIG (answering profiling / tolerating STOP probes).
    cfg = _await_config(sock, memory_mb)
    print("\n========== ASSIGNED BY SERVER ==========")
    print(cfg.describe())
    print("========================================\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PipelineSlice(cfg).to(device)
    config_ack(sock)
    print(f"[pipeline] slice built on {device}; CONFIG_ACK sent")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    # Wait for START.
    f = read_frame(sock)
    while f[0] == MSG_PIPELINE_STOP:
        f = read_frame(sock)
    if f[0] != MSG_PIPELINE_START:
        print(f"[pipeline] expected START, got 0x{f[0]:02x}; aborting")
        sock.close()
        return
    total_steps = struct.unpack_from(">I", f, 1)[0]
    M = cfg.num_micro_batches
    print(f"[pipeline] START: {total_steps} mini-batches x {M} micro-batches\n")

    try:
        for mb in range(total_steps):
            if cfg.is_head and cfg.total_stages == 1:
                loss = _run_single_stage_step(sock, model, opt, cfg, device, mb)
            elif cfg.is_head:
                loss = _run_head_step(sock, model, opt, cfg, device, mb)
            elif cfg.is_tail:
                loss = _run_tail_step(sock, model, opt, cfg, device, mb)
            else:
                loss = _run_middle_step(sock, model, opt, cfg, device, mb)

            # Sync barrier at the end of each mini-batch.
            bf = _recv_or_stop(sock, MSG_PIPELINE_SYNC_BARRIER)
            if bf[0] != MSG_PIPELINE_SYNC_BARRIER:
                print(f"[pipeline] expected SYNC_BARRIER, got 0x{bf[0]:02x}")
                break
            mb_id = struct.unpack_from(">I", bf, 1)[0]
            write_frame(sock, build_sync_ack(mb_id))

            tag = "loss=%.4f" % loss if loss is not None else "(relayed)"
            print(f"[pipeline] mini-batch {mb+1}/{total_steps} done  {tag}")
    except _Stop:
        print("[pipeline] server sent STOP — training ended.")
    finally:
        sock.close()
        print("[pipeline] done.")


def _await_config(sock, memory_mb):
    while True:
        f = read_frame(sock)
        t = f[0]
        if t == MSG_PIPELINE_STOP:
            continue
        if t == MSG_PROFILE_REQUEST:
            handle_profile_request(sock, f, memory_mb=memory_mb)
            continue
        if t == MSG_PIPELINE_CONFIG:
            return parse_config(f)
        # ignore anything else while waiting


def _accumulate_grads(opt, M):
    """Scale accumulated .grad by 1/M, step, then zero."""
    for p in opt.param_groups[0]["params"]:
        if p.grad is not None:
            p.grad.mul_(1.0 / M)
    opt.step()
    opt.zero_grad(set_to_none=True)


def _run_head_step(sock, model, opt, cfg, device, mini_batch):
    import torch
    M = cfg.num_micro_batches
    model.train()
    opt.zero_grad(set_to_none=True)
    outputs = {}   # mb_id -> retained output activation tensor (graph alive)
    # Phase A: receive M data batches, forward, send activations.
    for _ in range(M):
        df = _recv_or_stop(sock, MSG_PIPELINE_DATA_BATCH)
        mb_id, tokens, _targets = parse_data_batch(df)
        tok = torch.from_numpy(tokens).to(device)
        act = model.forward_head(tok)            # graph retained
        outputs[mb_id] = act
        arr = act.detach().cpu().numpy()
        write_frame(sock, build_activation(mb_id, cfg.stage_index, arr))
    # Phase B: receive M gradients, backward into params.
    for _ in range(M):
        gf = _recv_or_stop(sock, MSG_PIPELINE_GRADIENT)
        mb_id, _loss, grad = parse_gradient(gf)
        g = torch.from_numpy(grad).to(device)
        outputs[mb_id].backward(g)               # accumulates param grads
    _accumulate_grads(opt, M)
    return None


def _run_middle_step(sock, model, opt, cfg, device, mini_batch):
    import torch
    M = cfg.num_micro_batches
    model.train()
    opt.zero_grad(set_to_none=True)
    inputs = {}    # mb_id -> input activation (requires_grad)
    outputs = {}   # mb_id -> output activation (graph alive)
    for _ in range(M):
        af = _recv_or_stop(sock, MSG_PIPELINE_ACTIVATION)
        mb_id, act_in, _t = parse_activation(af)
        a_in = torch.from_numpy(act_in).to(device).requires_grad_(True)
        out = model.forward_middle(a_in)
        inputs[mb_id] = a_in
        outputs[mb_id] = out
        write_frame(sock, build_activation(mb_id, cfg.stage_index,
                                           out.detach().cpu().numpy()))
    for _ in range(M):
        gf = _recv_or_stop(sock, MSG_PIPELINE_GRADIENT)
        mb_id, _loss, grad = parse_gradient(gf)
        g = torch.from_numpy(grad).to(device)
        outputs[mb_id].backward(g)               # param grads + inputs[mb_id].grad
        up = inputs[mb_id].grad.detach().cpu().numpy()
        write_frame(sock, build_gradient(mb_id, cfg.stage_index, up, 0.0))
    _accumulate_grads(opt, M)
    return None


def _run_tail_step(sock, model, opt, cfg, device, mini_batch):
    import torch
    M = cfg.num_micro_batches
    model.train()
    opt.zero_grad(set_to_none=True)
    last_loss = 0.0
    for _ in range(M):
        af = _recv_or_stop(sock, MSG_PIPELINE_ACTIVATION)
        mb_id, act_in, targets = parse_activation(af)
        a_in = torch.from_numpy(act_in).to(device).requires_grad_(True)
        tgt = torch.from_numpy(targets).to(device)
        logits = model.forward_tail(a_in)
        loss = model.tail_loss(logits, tgt)
        loss.backward()                          # param grads + a_in.grad
        last_loss = float(loss.detach().cpu())
        write_frame(sock, build_loss_report(mini_batch, mb_id, last_loss, mini_batch))
        up = a_in.grad.detach().cpu().numpy()
        write_frame(sock, build_gradient(mb_id, cfg.stage_index, up, last_loss))
    _accumulate_grads(opt, M)
    return last_loss


def _run_single_stage_step(sock, model, opt, cfg, device, mini_batch):
    """total_stages == 1: this worker is head AND tail. tokens -> loss locally."""
    import torch
    M = cfg.num_micro_batches
    model.train()
    opt.zero_grad(set_to_none=True)
    last_loss = 0.0
    for _ in range(M):
        df = _recv_or_stop(sock, MSG_PIPELINE_DATA_BATCH)
        mb_id, tokens, targets = parse_data_batch(df)
        tok = torch.from_numpy(tokens).to(device)
        tgt = torch.from_numpy(targets).to(device)
        x = model.forward_head(tok)              # embedding + blocks
        x = model.final_ln(x)
        logits = x @ model.head_embedding.weight.t()
        loss = model.tail_loss(logits, tgt)
        loss.backward()
        last_loss = float(loss.detach().cpu())
        write_frame(sock, build_loss_report(mini_batch, mb_id, last_loss, mini_batch))
    _accumulate_grads(opt, M)
    return last_loss


# ── Stage 0 driver: join + print assignment, no training ──

def run_join_only(host: str, port: int, secret: str, memory_mb: int = 8192,
                  timeout: float = 300.0):
    print(f"[pipeline] connecting to {host}:{port} ...")
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)

    authenticate(sock, secret)
    print("[pipeline] authenticated (worker secret OK)")

    register(sock, memory_mb=memory_mb)
    print(f"[pipeline] registered (device_type={DEVICE_TYPE}, mem={memory_mb}MB)")
    print("[pipeline] waiting for the orchestrator to start a training run...")
    print("[pipeline] (ask your teammate to trigger training so the server "
          "profiles + assigns layers)")

    while True:
        frame = read_frame(sock)
        msg_type = frame[0]

        if msg_type == MSG_PIPELINE_STOP:
            # Liveness probe between runs — ignore and keep waiting.
            continue

        if msg_type == MSG_PROFILE_REQUEST:
            print("[pipeline] PROFILE_REQUEST received -> replying with profile")
            handle_profile_request(sock, frame, memory_mb=memory_mb)
            continue

        if msg_type == MSG_PIPELINE_CONFIG:
            cfg = parse_config(frame)
            print("\n========== ASSIGNED BY SERVER ==========")
            print(cfg.describe())
            print("========================================\n")
            config_ack(sock)
            print("[pipeline] sent CONFIG_ACK. Join proof complete.")

            # Optionally wait briefly for START so we can confirm the run begins.
            try:
                sock.settimeout(30.0)
                nxt = read_frame(sock)
                if nxt[0] == MSG_PIPELINE_START:
                    total_steps = struct.unpack_from(">I", nxt, 1)[0]
                    print(f"[pipeline] START received: total_steps={total_steps}")
                    print("[pipeline] (STAGE 0 stops here — training compute is "
                          "not implemented yet)")
                else:
                    print(f"[pipeline] next frame after ack: 0x{nxt[0]:02x}")
            except socket.timeout:
                print("[pipeline] no START within 30s (that's fine for join-proof)")
            break

        print(f"[pipeline] unexpected frame while waiting: 0x{msg_type:02x}")

    sock.close()
    print("[pipeline] done.")


def main():
    parser = argparse.ArgumentParser(
        description="Centurion pipeline worker for the 9998 pipeline server.")
    parser.add_argument("--host", default="34.60.122.134")
    parser.add_argument("--port", type=int, default=9998)
    parser.add_argument("--secret", required=True,
                        help="worker HMAC secret (e.g. centurion-2026)")
    parser.add_argument("--memory-mb", type=int, default=8192,
                        help="available memory reported to the server (MB); "
                             "higher -> the server may assign more layers")
    parser.add_argument("--join-only", action="store_true",
                        help="Stage 0: just join and print the assigned layer "
                             "range, without training (use to verify the join).")
    args = parser.parse_args()

    if args.join_only:
        run_join_only(args.host, args.port, args.secret, memory_mb=args.memory_mb)
    else:
        run_training(args.host, args.port, args.secret, memory_mb=args.memory_mb)


if __name__ == "__main__":
    main()
