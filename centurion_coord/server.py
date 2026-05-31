"""DiLoCo coordinator HTTP service (FastAPI).

Protocol (round-based barrier):

  global[0] = canonical shared init.
  For round r:
    1. worker GET  /global_params?round=r     -> bytes of global[r]
    2. worker      set_params(global[r]); train K steps locally
    3. worker POST /submit_params?worker_id=..&round=r  (body = its params bytes)
    4. when all `world_size` submissions for round r arrive, coordinator
       averages them -> global[r+1], advances current_round to r+1.
    5. worker polls GET /round until current_round > r, then pulls global[r+1].

The coordinator stays framework-agnostic: it only averages numpy/safetensors.
"""

from __future__ import annotations

import argparse
import threading
from typing import Dict

from fastapi import FastAPI, HTTPException, Request, Response
import uvicorn

from centurion_worker import model_spec, tensor_codec


class Coordinator:
    def __init__(self, world_size: int, total_rounds: int):
        self.world_size = world_size
        self.total_rounds = total_rounds
        self._lock = threading.Lock()
        self.current_round = 0
        # global[r] -> serialized params that workers train on during round r.
        self._global: Dict[int, bytes] = {
            0: tensor_codec.state_to_bytes(model_spec.init_params(seed=0))
        }
        # submissions for the in-progress round: worker_id -> params bytes
        self._submissions: Dict[str, bytes] = {}
        self._known_workers = set()

    def register(self, worker_id: str) -> dict:
        with self._lock:
            self._known_workers.add(worker_id)
            return {
                "ok": True,
                "current_round": self.current_round,
                "world_size": self.world_size,
                "known_workers": sorted(self._known_workers),
            }

    def global_params(self, round_id: int) -> bytes:
        with self._lock:
            if round_id not in self._global:
                raise HTTPException(
                    status_code=409,
                    detail=f"global for round {round_id} not ready "
                           f"(current_round={self.current_round})",
                )
            return self._global[round_id]

    def submit(self, worker_id: str, round_id: int, blob: bytes) -> dict:
        with self._lock:
            if round_id != self.current_round:
                raise HTTPException(
                    status_code=409,
                    detail=f"stale submit: worker round {round_id} != "
                           f"current_round {self.current_round}",
                )
            self._known_workers.add(worker_id)
            self._submissions[worker_id] = blob
            received = len(self._submissions)

            advanced = False
            if received >= self.world_size:
                states = [tensor_codec.state_from_bytes(b)
                          for b in self._submissions.values()]
                averaged = tensor_codec.average_states(states)
                next_round = self.current_round + 1
                self._global[next_round] = tensor_codec.state_to_bytes(averaged)
                self.current_round = next_round
                self._submissions = {}
                advanced = True

            return {
                "ok": True,
                "round": round_id,
                "received": received,
                "world_size": self.world_size,
                "advanced": advanced,
                "current_round": self.current_round,
            }

    def status(self) -> dict:
        with self._lock:
            return {
                "current_round": self.current_round,
                "total_rounds": self.total_rounds,
                "world_size": self.world_size,
                "submissions_this_round": len(self._submissions),
                "known_workers": sorted(self._known_workers),
                "done": self.current_round >= self.total_rounds,
            }


def build_app(world_size: int, total_rounds: int) -> FastAPI:
    app = FastAPI(title="Centurion DiLoCo Coordinator")
    coord = Coordinator(world_size=world_size, total_rounds=total_rounds)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/round")
    def get_round():
        return coord.status()

    @app.get("/status")
    def status():
        return coord.status()

    @app.post("/register")
    async def register(worker_id: str):
        return coord.register(worker_id)

    @app.get("/global_params")
    def global_params(round: int):
        blob = coord.global_params(round)
        return Response(content=blob, media_type="application/octet-stream")

    @app.post("/submit_params")
    async def submit_params(worker_id: str, round: int, request: Request):
        blob = await request.body()
        return coord.submit(worker_id, round, blob)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0",
                        help="0.0.0.0 so other machines on the LAN can reach it")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--world-size", type=int, required=True,
                        help="number of workers that must submit each round")
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    app = build_app(world_size=args.world_size, total_rounds=args.rounds)
    print(f"Coordinator on {args.host}:{args.port} "
          f"world_size={args.world_size} rounds={args.rounds}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
