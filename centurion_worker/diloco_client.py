"""DiLoCo worker client.

Drives one worker through the round-based barrier against the coordinator:
  pull global -> train K local steps -> submit -> wait for barrier -> repeat.

Run on Windows with `--framework pytorch`; run on Mac with `--framework mlx`.
"""

from __future__ import annotations

import argparse
import time

import httpx

from centurion_worker import data, make_backend, tensor_codec


def run_worker(
    coord_url: str,
    worker_id: str,
    framework: str,
    data_seed: int,
    local_steps: int,
    rounds: int,
    poll_seconds: float = 0.5,
    timeout: float = 30.0,
):
    coord_url = coord_url.rstrip("/")
    backend = make_backend(framework, data_seed=data_seed)
    print(f"[{worker_id}] framework={backend.framework} device={backend.device_kind}")

    client = httpx.Client(timeout=timeout)
    client.post(f"{coord_url}/register", params={"worker_id": worker_id})

    xe, ye = data.fixed_eval_batch(512)

    for r in range(rounds):
        # 1. pull global[r]
        blob = _get_with_retry(client, f"{coord_url}/global_params",
                               params={"round": r}, poll_seconds=poll_seconds)
        backend.set_params(tensor_codec.state_from_bytes(blob))

        # 2. local training
        losses = backend.train_local_steps(local_steps)

        # 3. submit local params for round r
        out = tensor_codec.state_to_bytes(backend.get_params())
        resp = client.post(
            f"{coord_url}/submit_params",
            params={"worker_id": worker_id, "round": r},
            content=out,
        )
        resp.raise_for_status()

        eval_loss = backend.eval_loss(xe, ye)
        print(f"[{worker_id}] round={r} "
              f"train_loss {losses[0]:.4f}->{losses[-1]:.4f} "
              f"global_eval_loss={eval_loss:.4f}")

        # 4. wait until the coordinator has produced global[r+1]
        _wait_for_round(client, coord_url, r + 1, poll_seconds, rounds)

    # final: pull the last global and report
    final_round = rounds
    try:
        blob = _get_with_retry(client, f"{coord_url}/global_params",
                               params={"round": final_round},
                               poll_seconds=poll_seconds, max_wait=10.0)
        backend.set_params(tensor_codec.state_from_bytes(blob))
        print(f"[{worker_id}] FINAL global_eval_loss={backend.eval_loss(xe, ye):.4f}")
    except Exception as exc:
        print(f"[{worker_id}] final pull skipped: {exc}")

    client.close()


def _get_with_retry(client, url, params, poll_seconds, max_wait=120.0):
    waited = 0.0
    while True:
        resp = client.get(url, params=params)
        if resp.status_code == 200:
            return resp.content
        if resp.status_code != 409:
            resp.raise_for_status()
        time.sleep(poll_seconds)
        waited += poll_seconds
        if waited > max_wait:
            raise TimeoutError(f"timed out waiting for {url} {params}")


def _wait_for_round(client, coord_url, target_round, poll_seconds, total_rounds,
                    max_wait=120.0):
    waited = 0.0
    while True:
        st = client.get(f"{coord_url}/round").json()
        if st["current_round"] >= target_round or st.get("done"):
            return
        time.sleep(poll_seconds)
        waited += poll_seconds
        if waited > max_wait:
            raise TimeoutError(f"timed out waiting for round {target_round}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coord-url", required=True,
                        help="e.g. http://192.168.1.50:9100")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--framework", choices=["pytorch", "mlx"], required=True)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()

    run_worker(
        coord_url=args.coord_url,
        worker_id=args.worker_id,
        framework=args.framework,
        data_seed=args.data_seed,
        local_steps=args.local_steps,
        rounds=args.rounds,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
