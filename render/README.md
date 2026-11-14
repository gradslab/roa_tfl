# Animations

Each script writes `<name>.mp4`, `<name>.gif` and `<name>.jpg` to `../docs/assets/media/`. The repository keeps the GIFs shown in the top-level README.

| Script | Content |
|---|---|
| `unicycle_circle.py` | Unicycles started on the circle slice (heading north, speed 1) |
| `unicycle_sinusoid.py` | The same on the sinusoid slice |
| `quadrotor.py` | Planar quadrotors on the vertical loop |
| `level_sweep.py` | Level sweep of seed 0's candidate on the circle slice |

Requires the packages of `../requirements.txt`, Pillow, `ffmpeg` (libx264) and `gifsicle`. Data are read from
`../tfl/`. The unicycle slices use `tfl/results/slice_gt_*_321.pt`, written by `tfl/figs3.py`.

```sh
python tfl_data.py
python unicycle_circle.py
python unicycle_sinusoid.py
python quadrotor.py
python level_sweep.py
```

Seeds are fixed. Simulations and chosen initial states are cached in `_cache/`; delete it to recompute everything.
