#!/usr/bin/env python3
"""JSONL stdin/stdout bridge from Go test51 to ARC-AGI-3 Arcade.

Protocol (one JSON object per line):
  {"op":"reset"} -> {"ok":true,"frame":{...}}
  {"op":"step","action":{"id":1,"x":0,"y":0}} -> {"ok":true,"frame":{...}}
  {"op":"close"} -> exit

Requires ARC_API_KEY (or anonymous) and the arc-agi package from ARC-AGI3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _frame_to_dict(frame: Any, game_id: str) -> dict[str, Any]:
    layers = []
    raw = getattr(frame, "frame", None) or getattr(frame, "layers", None)
    if raw is None:
        layers = [[[0 for _ in range(64)] for _ in range(64)]]
    else:
        # arcengine FrameData.frame is typically list[list[list[int]]]
        try:
            layers = [list(map(list, plane)) for plane in raw]
        except TypeError:
            layers = [[[0 for _ in range(64)] for _ in range(64)]]
    state = getattr(frame, "state", None)
    state_s = state.name if hasattr(state, "name") else str(state or "NOT_FINISHED")
    avail = list(getattr(frame, "available_actions", None) or [1, 2, 3, 4, 5])
    return {
        "layers": layers,
        "state": state_s,
        "levels_completed": int(getattr(frame, "levels_completed", 0) or 0),
        "win_levels": int(getattr(frame, "win_levels", 1) or 1),
        "available_actions": [int(a) for a in avail],
        "game_id": game_id,
    }


def _action_from_dict(d: dict[str, Any]):
    from arcengine import GameAction

    aid = int(d.get("id", 5))
    try:
        action = GameAction.from_id(aid)
    except Exception:
        action = GameAction.ACTION5
    if action == GameAction.ACTION6:
        action.set_data({"x": int(d.get("x", 0)), "y": int(d.get("y", 0))})
    return action


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    args = ap.parse_args()
    game_id = args.game

    try:
        from arc_agi import Arcade
        from arcengine import GameAction
    except ImportError as e:
        print(json.dumps({"ok": False, "error": f"import arc_agi/arcengine: {e}"}), flush=True)
        return 1

    arcade = Arcade()
    env = arcade.make(game_id)
    # initial reset so observation exists
    try:
        obs = env.reset()
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"reset: {e}"}), flush=True)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
            continue
        op = msg.get("op")
        try:
            if op == "close":
                return 0
            if op == "reset":
                obs = env.reset()
                print(json.dumps({"ok": True, "frame": _frame_to_dict(obs, game_id)}), flush=True)
                continue
            if op == "step":
                action = _action_from_dict(msg.get("action") or {})
                obs = env.step(action)
                print(json.dumps({"ok": True, "frame": _frame_to_dict(obs, game_id)}), flush=True)
                continue
            print(json.dumps({"ok": False, "error": f"unknown op {op}"}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
