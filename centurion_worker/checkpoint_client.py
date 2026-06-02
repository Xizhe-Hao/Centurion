"""Checkpoint-server worker client (connects to the teammate's cloud aggregator).

This is the TCP adapter that lets a Windows (PyTorch) or Mac (MLX) worker join
the team's already-deployed checkpoint aggregation server at
34.60.122.134:9999 instead of our own HTTP coordinator.

The server (Centurion/Server/checkpoint_server.py in the upstream repo) does
DiLoCo-style EMA averaging of full-model safetensors blobs:
    global = (1 - alpha) * global + alpha * incoming      (alpha = 0.5 default)
It is asynchronous: there are no rounds/barriers. Whoever uploads gets mixed in
immediately; `global_step` / `num_contributors` just count uploads.

Wire protocol (length-prefixed binary frames over TCP):
    Frame: [4B big-endian uint32 payload_length][payload bytes]

Auth (HMAC-SHA256 challenge-response, once per connection):
    server -> [0x30][32B nonce]
    client -> [0x31][32B HMAC-SHA256(secret_utf8, nonce)]
    server -> [0x32][1B status: 0=ok, 1=fail]

Checkpoint messages:
    UPLOAD   client -> [0x20][uint32 worker_id][uint32 local_step][uint32 st_len][safetensors]
    ACK      server -> [0x21][uint32 global_step][uint32 num_contributors][1B status]
    REQUEST  client -> [0x22][uint32 worker_id]
    RESPONSE server -> [0x23][uint32 global_step][uint32 num_contributors][uint32 st_len][safetensors]
             (or an ACK with status=1 meaning "no global yet -> poll again")

Reuses tensor_codec (safetensors fp32) and the Backend interface unchanged.
Only stdlib for transport: socket, struct, hmac, hashlib.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import socket
import struct
import time

from centurion_worker import data, make_backend, tensor_codec

# ── Message type constants (must match checkpoint_server.py) ──
MSG_CHECKPOINT_UPLOAD = 0x20
MSG_CHECKPOINT_ACK = 0x21
MSG_CHECKPOINT_REQUEST = 0x22
MSG_CHECKPOINT_RESPONSE = 0x23
MSG_AUTH_CHALLENGE = 0x30
MSG_AUTH_RESPONSE = 0x31
MSG_AUTH_RESULT = 0x32


# ── Frame I/O ──

def _read_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed by server")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> bytes:
    """Read one length-prefixed frame: [4B BE length][payload]."""
    length = struct.unpack(">I", _read_exactly(sock, 4))[0]
    return _read_exactly(sock, length)


def write_frame(sock: socket.socket, payload: bytes) -> None:
    """Write one length-prefixed frame."""
    sock.sendall(struct.pack(">I", len(payload)) + payload)


# ── Authentication ──

def authenticate(sock: socket.socket, secret: str) -> None:
    """Complete the HMAC-SHA256 challenge-response. Raises on failure."""
    challenge = read_frame(sock)
    if not challenge or challenge[0] != MSG_AUTH_CHALLENGE or len(challenge) != 33:
        raise ConnectionError(
            f"expected AUTH_CHALLENGE (0x30, 33B), got "
            f"0x{challenge[0]:02x} len={len(challenge)}"
        )
    nonce = challenge[1:33]
    mac = hmac.new(secret.encode("utf-8"), nonce, hashlib.sha256).digest()
    write_frame(sock, struct.pack(">B", MSG_AUTH_RESPONSE) + mac)

    result = read_frame(sock)
    if not result or result[0] != MSG_AUTH_RESULT:
        raise ConnectionError(f"expected AUTH_RESULT (0x32), got 0x{result[0]:02x}")
    if result[1] != 0:
        raise PermissionError("authentication failed: wrong secret")


# ── Checkpoint messages ──

def upload_checkpoint(sock: socket.socket, worker_id: int, local_step: int,
                      st_blob: bytes) -> dict:
    """Send a CHECKPOINT_UPLOAD and parse the ACK."""
    header = struct.pack(">BIII", MSG_CHECKPOINT_UPLOAD,
                         worker_id, local_step, len(st_blob))
    write_frame(sock, header + st_blob)

    ack = read_frame(sock)
    if ack[0] != MSG_CHECKPOINT_ACK:
        raise ConnectionError(f"expected ACK (0x21), got 0x{ack[0]:02x}")
    _, global_step, num_contributors, status = struct.unpack(">BIIB", ack[:10])
    return {"global_step": global_step,
            "num_contributors": num_contributors,
            "status": status}


def request_checkpoint(sock: socket.socket, worker_id: int):
    """Send a CHECKPOINT_REQUEST.

    Returns (state_dict, meta) if a global model exists, or (None, meta) if the
    aggregator is empty / not ready (server replies with an ACK, status=1).
    """
    write_frame(sock, struct.pack(">BI", MSG_CHECKPOINT_REQUEST, worker_id))
    resp = read_frame(sock)
    msg_type = resp[0]

    if msg_type == MSG_CHECKPOINT_ACK:
        # No global yet (status byte at offset 9).
        _, global_step, num_contributors, status = struct.unpack(">BIIB", resp[:10])
        return None, {"global_step": global_step,
                      "num_contributors": num_contributors}

    if msg_type == MSG_CHECKPOINT_RESPONSE:
        _, global_step, num_contributors, st_len = struct.unpack_from(">BIII", resp, 0)
        blob = resp[13:13 + st_len]
        state = tensor_codec.state_from_bytes(blob)
        return state, {"global_step": global_step,
                       "num_contributors": num_contributors}

    raise ConnectionError(f"unexpected REQUEST reply: 0x{msg_type:02x}")


# ── Worker loop ──

def run_worker(
    host: str,
    port: int,
    secret: str,
    worker_id: int,
    framework: str,
    data_seed: int,
    local_steps: int,
    rounds: int,
    model: str = "mlp",
    poll_seconds: float = 0.5,
    timeout: float = 60.0,
):
    backend = make_backend(framework, model=model, data_seed=data_seed)
    print(f"[w{worker_id}] framework={backend.framework} device={backend.device_kind}")

    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    authenticate(sock, secret)
    print(f"[w{worker_id}] authenticated to {host}:{port}")

    xe, ye = data.fixed_eval_batch(512)

    try:
        for r in range(rounds):
            # 1. Pull the current global model (if any). If the aggregator is
            #    empty we are the seeding worker and just train from local init.
            global_state, meta = request_checkpoint(sock, worker_id)
            if global_state is not None:
                try:
                    backend.set_params(global_state)
                    pulled = True
                except Exception as exc:
                    # Key mismatch (e.g. server seeded with a different model).
                    print(f"[w{worker_id}] WARN: could not load global "
                          f"(key mismatch?): {exc}")
                    pulled = False
            else:
                pulled = False

            # 2. Local training.
            losses = backend.train_local_steps(local_steps)

            # 3. Upload our updated params.
            blob = tensor_codec.state_to_bytes(backend.get_params())
            ack = upload_checkpoint(sock, worker_id, r, blob)

            eval_loss = backend.eval_loss(xe, ye)
            print(f"[w{worker_id}] round={r} "
                  f"train_loss {losses[0]:.4f}->{losses[-1]:.4f} "
                  f"eval_loss={eval_loss:.4f} "
                  f"global_step={ack['global_step']} "
                  f"contributors={ack['num_contributors']} "
                  f"pulled_global={pulled}")

            time.sleep(poll_seconds)

        print(f"[w{worker_id}] done.")
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="Centurion worker that connects to the team's cloud "
                    "checkpoint aggregation server.")
    parser.add_argument("--host", default="34.60.122.134")
    parser.add_argument("--port", type=int, default=9999,
                        help="checkpoint server port (9999); NOT the pipeline "
                             "server on 9998")
    parser.add_argument("--secret", required=True,
                        help="HMAC shared secret (ask the server owner)")
    parser.add_argument("--worker-id", type=int, required=True,
                        help="a uint32 you pick to identify this worker")
    parser.add_argument("--framework", choices=["pytorch", "mlx"], required=True)
    parser.add_argument("--model", choices=["mlp", "gpt2"], default="mlp",
                        help="mlp = tiny protocol-test model; gpt2 = GPT-2 small "
                             "aligned with the iOS/Mac workers")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()

    run_worker(
        host=args.host,
        port=args.port,
        secret=args.secret,
        worker_id=args.worker_id,
        framework=args.framework,
        data_seed=args.data_seed,
        local_steps=args.local_steps,
        rounds=args.rounds,
        model=args.model,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
