"""Shared paths, palette, style and the MP4/GIF/JPG writer. The tfl/ and benchmarks/ code is only read."""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TFL = ROOT / "tfl"
UNI = TFL / "unicycle"
BENCH = ROOT / "benchmarks"
MEDIA = HERE.parent / "docs" / "assets" / "media"
CACHE = HERE / "_cache"
CACHE.mkdir(exist_ok=True)
MEDIA.mkdir(parents=True, exist_ok=True)

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
GIFSICLE = shutil.which("gifsicle") or "/opt/homebrew/bin/gifsicle"

C_OUT = "#f2f1ee"
C_ROA = "#a9e3c9"
C_EST = "#2a78d6"
C_INN = "#eb6834"
C_PATH = "#1b1b1b"
C_BAD = "#e34948"
INK, INK2 = "#1b1b1b", "#5a5955"
C_OBS = "#2b2b2b"
PATH_DASH = (0, (3.5, 2.2))

FPS = 30


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 17, "axes.titlesize": 21, "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.edgecolor": "#b9b7b1", "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


@contextlib.contextmanager
def research(subdir):
    old = os.getcwd()
    d = Path(subdir)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    for p in (UNI, BENCH):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def clean_axes(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


class Video:

    def __init__(self, name, fig, gif_width=720, gif_fps=15, crf=22, lossy=40):
        self.name, self.fig = name, fig
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        assert w % 2 == 0 and h % 2 == 0, (w, h)
        self.w, self.h = w, h
        self.mp4 = MEDIA / f"{name}.mp4"
        self.gif_raw = CACHE / f"{name}_raw.gif"
        self.gif = MEDIA / f"{name}.gif"
        self.jpg = MEDIA / f"{name}.jpg"
        self.lossy = lossy
        flt = (f"[0:v]split=2[a][b];[b]fps={gif_fps},scale={gif_width}:-2:flags=lanczos,split[c][d];"
               f"[c]palettegen=max_colors=128:stats_mode=full[p];[d][p]paletteuse=dither=sierra2_4a[g]")
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "-", "-filter_complex", flt,
               "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-tune", "animation", str(self.mp4),
               "-map", "[g]", "-loop", "0", str(self.gif_raw)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n = 0

    def grab(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        self.proc.stdin.write(np.ascontiguousarray(buf).tobytes())
        self.n += 1
        return buf

    def poster(self, buf=None):
        from PIL import Image
        if buf is None:
            self.fig.canvas.draw()
            buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        Image.fromarray(np.ascontiguousarray(buf)).save(self.jpg, quality=90, optimize=True, progressive=True)

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        subprocess.run([GIFSICLE, "-O3", f"--lossy={self.lossy}", "--no-comments", "--no-names", "--no-extensions",
                        "--loopcount=forever", str(self.gif_raw), "-o", str(self.gif)], check=True)
        self.gif_raw.unlink()
        for p in (self.mp4, self.gif, self.jpg):
            if p.exists():
                print(f"  {p.name:28s} {p.stat().st_size / 1e6:6.2f} MB")
        print(f"  {self.n} frames, {self.n / FPS:.1f} s")


def trail_segments(H, k, L, wrap=None):
    """Segments of the last L steps of the histories H (F, N, 2) up to frame k, and their age weight in (0, 1]."""
    j0 = max(0, k - L)
    if k - j0 < 1:
        return np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0, dtype=int)
    a, b = H[j0:k], H[j0 + 1:k + 1]
    seg = np.stack([a, b], 2).reshape(-1, 2, 2)
    w = np.repeat(((np.arange(j0, k) - (k - L) + 1) / L), H.shape[1])
    who = np.tile(np.arange(H.shape[1]), k - j0)
    keep = np.any(seg[:, 0] != seg[:, 1], axis=1)
    if wrap is not None:
        keep &= np.abs(seg[:, 0, 0] - seg[:, 1, 0]) < wrap
    return seg[keep], np.clip(w[keep], 0, 1), who[keep]


class Bursts:
    def __init__(self, ax, n, rmax, zorder=9):
        from matplotlib.patches import Circle
        self.rmax = rmax
        self.ring = [ax.add_patch(Circle((0, 0), 0, fill=False, ec=C_BAD, lw=2.4, zorder=zorder, visible=False, clip_on=False)) for _ in range(n)]
        self.disc = [ax.add_patch(Circle((0, 0), 0, fc=C_BAD, ec="none", zorder=zorder - 0.1, visible=False, clip_on=False)) for _ in range(n)]

    def update(self, i, xy, age, dur=0.45):
        on = 0 <= age < dur
        self.ring[i].set_visible(on); self.disc[i].set_visible(on)
        if on:
            u = age / dur
            r = self.rmax * (0.25 + 0.75 * (1 - (1 - u) ** 2))
            self.ring[i].center = self.disc[i].center = xy
            self.ring[i].set_radius(r); self.disc[i].set_radius(0.6 * r)
            self.ring[i].set_alpha(1 - u); self.disc[i].set_alpha(0.45 * (1 - u))


def smooth(xs, ys, M, up=3, sigma=1.5):
    """Upsampled, lightly smoothed copy of a grid field, for contours without stair steps."""
    from scipy import ndimage
    Z = ndimage.gaussian_filter(ndimage.zoom(np.asarray(M, float), up, order=1), sigma)
    return np.linspace(xs[0], xs[-1], Z.shape[1]), np.linspace(ys[0], ys[-1], Z.shape[0]), Z


def layout(width, ext, top=0.75, bottom=1.25, side=0.3, dpi=150):
    """Figure and data axes of the given width (inches) with room for a title and a legend."""
    aw = width - 2 * side
    ah = aw * (ext[3] - ext[2]) / (ext[1] - ext[0])
    H = top + ah + bottom
    H = round(H * dpi / 2) * 2 / dpi
    fig = plt.figure(figsize=(width, H), dpi=dpi)
    ax = fig.add_axes([side / width, bottom / H, aw / width, ah / H])
    return fig, ax, (bottom + ah) / H, bottom / H
