# test51 speed-run

Pure Go multithreaded twin of test51 + Tide.

- **Pick one layer** → full permute (all modes × all dtypes × all challenges × funny LRs × cams 1–3 × grids 1³–3³)
- **`runtime.NumCPU()` workers** (default) — about **one leaf per core**; Welvet leaf MultiCore off when workers > 1
- **Checkpoint:** `speed_ckpt/<layer>/`
- **Dash :5151** · **Tide :8080** · autostart on by default

## Run

```bash
cd apps/aai/test51/test51_speed_run

go run .                              # menu: 1–4 layer, 5=all, or csv
go run . -layer dense
go run . -layer dense,dense-wide
go run . -layer all                   # all four recipes, one after another
go run . -layer dense-wide -workers 8
go run . -layer dense -modes NormalBP,MeshBP
```

Each selected layer writes **`speed_ckpt/<layer>/`**. Workers stay on for every layer (~1 core per leaf).

## Flags / env

| Flag | Env | Default |
|------|-----|---------|
| `-layer` | `SPEED_LAYER` | ask (`dense` / csv / `all`) |
| `-modes` | `SPEED_MODES` | `all` |
| `-workers` | `SPEED_WORKERS` | `NumCPU` |
| `-full` | `SPEED_FULL` | `true` |
| `-ckpt-root` | `SPEED_CKPT_ROOT` | `speed_ckpt` |
| `-addr` | `SPEED_ADDR` | `0.0.0.0:5151` |
| `-tide-addr` | `SPEED_TIDE_ADDR` | `0.0.0.0:8080` |
| `-autostart` | `SPEED_AUTOSTART` | `true` |

Fedora ports: `../unlock-ports.sh` (parent test51 script).

## Layout

```
speed_ckpt/dense/{progress,history,results}.json
speed_ckpt/dense-wide/…
```
