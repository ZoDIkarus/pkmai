#!/usr/bin/env python3
import glob
import json
import os
import time
from collections import defaultdict

BASE_DIR = os.path.expanduser("~/pokemon_ai_project")
INSTANCES_DIR = os.path.join(BASE_DIR, "instances_data")
OUTPUT_FILE = os.path.join(BASE_DIR, "skeleton_map.json")

POLL_INTERVAL = 1.0
SAVE_INTERVAL = 5.0
OVERWORLD_BANK = 3
MAX_COORD = 511

# Training agents only. Watcher (99) may fill real tiles, but is not needed
# for skeleton discovery.
TRAIN_AGENT_MIN = 0
TRAIN_AGENT_MAX = 39


def valid_point(bank, map_id, x, y):
    return (
        bank == OVERWORLD_BANK
        and 0 <= map_id <= 255
        and 0 <= x <= MAX_COORD
        and 0 <= y <= MAX_COORD
        and not (bank == 0 and map_id == 0 and x == 0 and y == 0)
    )


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


def load_existing():
    maps = {}
    transitions = {}
    if not os.path.exists(OUTPUT_FILE):
        return maps, transitions

    try:
        with open(OUTPUT_FILE, "r") as f:
            old = json.load(f)

        for m in old.get("maps", []):
            key = (int(m["bank"]), int(m["map_id"]))
            maps[key] = {
                "bank": key[0],
                "map_id": key[1],
                "min_x": int(m["min_x"]),
                "max_x": int(m["max_x"]),
                "min_y": int(m["min_y"]),
                "max_y": int(m["max_y"]),
                "observations": int(m.get("observations", 0)),
                "agents": set(int(a) for a in m.get("agents", [])),
            }

        for t in old.get("transitions", []):
            a = (int(t["from_bank"]), int(t["from_map"]))
            b = (int(t["to_bank"]), int(t["to_map"]))
            key = (a, b)
            transitions[key] = {
                "from_bank": a[0],
                "from_map": a[1],
                "to_bank": b[0],
                "to_map": b[1],
                "count": int(t.get("count", 0)),
                "agents": set(int(aid) for aid in t.get("agents", [])),
            }
    except Exception:
        pass

    return maps, transitions


def main():
    os.makedirs(INSTANCES_DIR, exist_ok=True)

    maps, transitions = load_existing()
    last_agent_pos = {}
    last_save = 0.0
    dirty = False

    print("🧱 Skeleton Map Builder gestartet")
    print("   Quelle: instances_data/inst_XX.json")
    print("   Keine Screenshots / keine Tiles / keine RAM-Reads")
    print("   Poll: 1x/s | Save: max 1x/5s")

    while True:
        loop_started = time.perf_counter()

        for path in glob.glob(os.path.join(INSTANCES_DIR, "inst_*.json")):
            try:
                with open(path, "r") as f:
                    inst = json.load(f)

                agent_id = int(inst.get("id", -1))
                if not (TRAIN_AGENT_MIN <= agent_id <= TRAIN_AGENT_MAX):
                    continue

                bank = int(inst.get("bank", 0))
                map_id = int(inst.get("map", 0))
                x = int(inst.get("x", 0))
                y = int(inst.get("y", 0))

                if not valid_point(bank, map_id, x, y):
                    continue

                key = (bank, map_id)
                if key not in maps:
                    maps[key] = {
                        "bank": bank,
                        "map_id": map_id,
                        "min_x": x,
                        "max_x": x,
                        "min_y": y,
                        "max_y": y,
                        "observations": 0,
                        "agents": set(),
                    }
                    dirty = True

                m = maps[key]
                old_bounds = (m["min_x"], m["max_x"], m["min_y"], m["max_y"])
                m["min_x"] = min(m["min_x"], x)
                m["max_x"] = max(m["max_x"], x)
                m["min_y"] = min(m["min_y"], y)
                m["max_y"] = max(m["max_y"], y)
                m["observations"] += 1
                m["agents"].add(agent_id)

                if old_bounds != (
                    m["min_x"], m["max_x"], m["min_y"], m["max_y"]
                ):
                    dirty = True

                prev = last_agent_pos.get(agent_id)
                cur = (bank, map_id, x, y)

                if prev is not None:
                    p_bank, p_map, p_x, p_y = prev
                    if (
                        p_bank == OVERWORLD_BANK
                        and (p_bank, p_map) != (bank, map_id)
                    ):
                        tkey = ((p_bank, p_map), (bank, map_id))
                        if tkey not in transitions:
                            transitions[tkey] = {
                                "from_bank": p_bank,
                                "from_map": p_map,
                                "to_bank": bank,
                                "to_map": map_id,
                                "count": 0,
                                "agents": set(),
                            }
                        transitions[tkey]["count"] += 1
                        transitions[tkey]["agents"].add(agent_id)
                        dirty = True

                last_agent_pos[agent_id] = cur

            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

        now = time.time()
        if dirty and now - last_save >= SAVE_INTERVAL:
            payload_maps = []
            for key in sorted(maps):
                m = maps[key]
                payload_maps.append({
                    "bank": m["bank"],
                    "map_id": m["map_id"],
                    "min_x": m["min_x"],
                    "max_x": m["max_x"],
                    "min_y": m["min_y"],
                    "max_y": m["max_y"],
                    "width_tiles": m["max_x"] - m["min_x"] + 1,
                    "height_tiles": m["max_y"] - m["min_y"] + 1,
                    "observations": m["observations"],
                    "agents": sorted(m["agents"]),
                })

            payload_transitions = []
            for key in sorted(transitions):
                t = transitions[key]
                payload_transitions.append({
                    "from_bank": t["from_bank"],
                    "from_map": t["from_map"],
                    "to_bank": t["to_bank"],
                    "to_map": t["to_map"],
                    "count": t["count"],
                    "agents": sorted(t["agents"]),
                })

            atomic_write_json(OUTPUT_FILE, {
                "updated_at": now,
                "overworld_bank": OVERWORLD_BANK,
                "maps": payload_maps,
                "transitions": payload_transitions,
            })
            dirty = False
            last_save = now

        elapsed = time.perf_counter() - loop_started
        time.sleep(max(0.05, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    main()
