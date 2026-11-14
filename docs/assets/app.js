(function () {
  "use strict";
  const C = { paper: "#fffdf8", grid: "#ece5d6", ink: "#16202f", ink2: "#46505f", muted: "#9aa0a8", aqua: "#a9e3c9", aquaD: "#1baf7a",
    blue: "#2a78d6", orange: "#eb6834", red: "#e34948", grey: "#f2f1ee" };
  const $ = (id) => document.getElementById(id);
  const DATA = "assets/data/";
  let CONST = null, RES = null;
  const SLICES = {};

  /* ------------------------------------------------------------ unicycle TFL closed loop */
  const DEF = {
    circle: { Lx: 2, L: 2, x4_lo: -3, x4_hi: 1, eps: [0.01, 0.01, 0.05], periodic_x1: false, gains: { k1: -1, k2: -2, k3: -1, eta2_ref: 1 } },
    sinusoid: { Lx: Math.PI, L: 2, x4_lo: -3, x4_hi: 1, eps: [0.01, 0.01, 0.05], periodic_x1: true, gains: { k1: -1, k2: -2, k3: -1, eta2_ref: 1 } },
  };
  const P = (c) => (CONST && CONST[c]) || DEF[c];

  function terms(c, x) {
    const [x1, x2, th, x4] = x, v = 1 + x4, cs = Math.cos(th), sn = Math.sin(th);
    if (c === "circle") {
      const r2 = x1 * x1 + x2 * x2, N = x1 * sn - x2 * cs, M = x1 * cs + x2 * sn;
      return { a: r2 - 1, La: 2 * v * M, Lp: v * N / r2, L2a: 2 * v * v, L2p: -2 * v * v * N * M / (r2 * r2),
        D: [[N / r2, v * M / r2], [2 * M, -2 * v * N]], reg: [Math.abs(2 * v), Math.abs(4 * v), Math.sqrt(r2)] };
    }
    const c1 = Math.cos(x1);
    return { a: x2 - Math.sin(x1), La: v * (sn - c1 * cs), Lp: v * cs, L2a: v * v * Math.sin(x1) * cs * cs, L2p: 0,
      D: [[cs, -v * sn], [sn - c1 * cs, v * (cs + c1 * sn)]], reg: [Math.abs(v), Math.abs(v), 1] };
  }
  function field(c, x) {
    const p = P(c), g = p.gains, t = terms(c, x), v = 1 + x[3];
    const npar = g.k3 * (t.Lp - g.eta2_ref), nperp = g.k1 * t.a + g.k2 * t.La;
    const b0 = -t.L2p + npar, b1 = -t.L2a + nperp, D = t.D, det = D[0][0] * D[1][1] - D[0][1] * D[1][0];
    const u1 = (D[1][1] * b0 - D[0][1] * b1) / det, u2 = (-D[1][0] * b0 + D[0][0] * b1) / det;
    return [v * Math.cos(x[2]), v * Math.sin(x[2]), u2, u1];
  }
  function dists(c, x) {
    const p = P(c), t = terms(c, x), e = p.eps;
    const d = [t.reg[0] - e[0], t.reg[1] - e[1], t.reg[2] - e[2]];
    if (!p.periodic_x1) d.push(p.L - Math.abs(x[0]));
    d.push(p.L - Math.abs(x[1]), x[3] - p.x4_lo, p.x4_hi - x[3]);
    return d;
  }
  function normField(c, x) {
    const p = P(c), f = field(c, x), s = [1 / p.Lx, 1 / p.L, 1 / Math.PI, 2 / (p.x4_hi - p.x4_lo)];
    let n = 1; for (let i = 0; i < 4; i++) n += (s[i] * f[i]) ** 2;
    const a = Math.sqrt(n); return [f.map((fi) => fi / a), a];
  }
  function rk4(c, x, h) {
    const k1 = normField(c, x)[0], x2 = x.map((xi, i) => xi + 0.5 * h * k1[i]);
    const k2 = normField(c, x2)[0], x3 = x.map((xi, i) => xi + 0.5 * h * k2[i]);
    const k3 = normField(c, x3)[0], x4 = x.map((xi, i) => xi + h * k3[i]);
    const k4 = normField(c, x4)[0];
    const y = x.map((xi, i) => xi + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    if (P(c).periodic_x1) y[0] = ((y[0] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return y;
  }
  function zvec(c, x) { const t = terms(c, x); return [t.a, t.La, t.Lp - P(c).gains.eta2_ref]; }
  function qex(c, x) {
    const p = CONST && CONST[c]; if (!p) return NaN;
    const z = zvec(c, x).map((zi, i) => zi / p.z_scale[i]); let q = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) q += z[i] * p.P[i][j] * z[j];
    return q;
  }
  function hobs(c, x) {
    const p = CONST && CONST[c]; if (!p) return 0;
    const d = dists(c, x); let h = 0;
    for (let i = 0; i < d.length; i++) { if (d[i] <= 0) return Infinity; h += Math.max(0, p.beta * p.scale[i] / d[i] - 1) ** p.h_pow; }
    return h;
  }
  /* simulate one run; returns samples every `every` steps with physical time */
  function simulate(c, x0, opt) {
    const o = Object.assign({ h: 0.025, tmax: 26, every: 2, maxSteps: 40000 }, opt || {});
    let x = x0.slice(), t = 0, hold = 0, out = [], status = "running";
    const g = P(c).gains;
    for (let k = 0; k < o.maxSteps; k++) {
      if (k % o.every === 0) out.push({ x: x.slice(), t });
      const a = normField(c, x)[1], y = rk4(c, x, o.h);
      const fin = y.every((v) => isFinite(v)), dd = fin ? dists(c, y) : [];
      const sing = !fin || Math.sign(1 + y[3]) !== Math.sign(1 + x[3]) || dd.slice(0, 3).some((v) => v <= 0);
      if (sing || dd.some((v) => v <= 0)) { out.push({ x: y.slice(), t: t + o.h / a }); status = sing ? "singular" : "box"; break; }
      x = y; t += o.h / a;
      const z = zvec(c, x);
      if (Math.abs(z[0]) < 2e-3 && Math.abs(z[1]) < 2e-3 && Math.abs(z[2]) < 0.05) { hold += o.h / a; if (hold > 1) { status = "converged"; out.push({ x: x.slice(), t }); break; } } else hold = 0;
      if (t > o.tmax) { status = "timeout"; break; }
    }
    return { pts: out, status };
  }

  /* ------------------------------------------------------------ helpers */
  function fit(cv) { const d = window.devicePixelRatio || 1, r = cv.getBoundingClientRect(); return { w: cv.width, h: cv.height, d, r }; }
  function mapper(cv, ext, pad) {
    const W = cv.width, H = cv.height, p = pad == null ? 40 : pad;
    const sx = (W - 2 * p) / (ext[1] - ext[0]), sy = (H - 2 * p) / (ext[3] - ext[2]), s = Math.min(sx, sy);
    const ox = (W - s * (ext[1] - ext[0])) / 2, oy = (H - s * (ext[3] - ext[2])) / 2;
    return { X: (x) => ox + (x - ext[0]) * s, Y: (y) => H - oy - (y - ext[2]) * s, inv: (px, py) => [ext[0] + (px - ox) / s, ext[2] + (H - oy - py) / s],
      s, ox, oy, w: s * (ext[1] - ext[0]), h: s * (ext[3] - ext[2]) };
  }
  function pathPoly(c, ext) {
    const pts = [];
    if (c === "circle") for (let i = 0; i <= 200; i++) { const a = 2 * Math.PI * i / 200; pts.push([Math.cos(a), Math.sin(a)]); }
    else for (let i = 0; i <= 200; i++) { const u = ext[0] + (ext[1] - ext[0]) * i / 200; pts.push([u, Math.sin(u)]); }
    return pts;
  }
  function drawPath(ctx, m, c, ext, alpha) {
    ctx.save(); ctx.setLineDash([10, 7]); ctx.lineWidth = 2.4; ctx.strokeStyle = `rgba(22,32,47,${alpha == null ? .75 : alpha})`; ctx.beginPath();
    pathPoly(c, ext).forEach((p, i) => (i ? ctx.lineTo(m.X(p[0]), m.Y(p[1])) : ctx.moveTo(m.X(p[0]), m.Y(p[1])))); ctx.stroke(); ctx.restore();
  }
  function axes(ctx, m, ext, labels) {
    ctx.save(); ctx.fillStyle = C.ink2; ctx.font = "500 22px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
    const tx = ext[1] > 3 ? [-3, 0, 3] : [-2, 0, 2];
    tx.forEach((v) => ctx.fillText(String(v).replace("-", "−"), m.X(v), m.Y(ext[2]) + 30));
    ctx.textAlign = "right"; [-2, 0, 2].forEach((v) => ctx.fillText(String(v).replace("-", "−"), m.X(ext[0]) - 10, m.Y(v) + 8));
    ctx.font = "italic 600 26px Inter, sans-serif"; ctx.textAlign = "center"; ctx.fillText(labels[0], m.X(0), m.Y(ext[2]) + 62);
    ctx.save(); ctx.translate(m.X(ext[0]) - 52, m.Y(0)); ctx.fillText(labels[1], 0, 8); ctx.restore(); ctx.restore();
  }
  function vehicle(ctx, px, py, th, col, size) {
    const s = size || 14; ctx.save(); ctx.translate(px, py); ctx.rotate(-th); ctx.fillStyle = col; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(s, 0); ctx.lineTo(-0.7 * s, 0.62 * s); ctx.lineTo(-0.35 * s, 0); ctx.lineTo(-0.7 * s, -0.62 * s); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
  }
  function burst(ctx, px, py, r) {
    ctx.save(); ctx.strokeStyle = C.red; ctx.lineWidth = 3; const s = r || 9;
    ctx.beginPath(); ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s); ctx.moveTo(px + s, py - s); ctx.lineTo(px - s, py + s); ctx.stroke(); ctx.restore();
  }
  const fmtPct = (x) => (100 * x).toFixed(1) + "%";
  function rand(a, b) { return a + Math.random() * (b - a); }

  /* ------------------------------------------------------------ data */
  async function getJSON(f) { const r = await fetch(DATA + f); if (!r.ok) throw new Error(f); return r.json(); }
  function loadSlices(c) {
    if (!SLICES[c]) SLICES[c] = fetchSlices(c).catch((e) => { delete SLICES[c]; throw e; });
    return SLICES[c];
  }
  async function fetchSlices(c) {
    const meta = await getJSON(`slices_${c}.json`);
    const r = await fetch(DATA + `slices_${c}.bin`); if (!r.ok) throw new Error(c);
    const buf = await r.arrayBuffer();
    const n = meta.n, S = meta.slices.length, bits = new Uint8Array(buf, 0, S * n * n), V = new Uint16Array(buf.slice(S * n * n));
    return { meta, bits, V, n };
  }
  function sliceIndex(meta, hdg, spd) { return meta.headings.indexOf(hdg) * meta.speeds.length + meta.speeds.indexOf(spd); }
  function sliceImage(sl, idx, colorFn) {
    const n = sl.n, img = new ImageData(n, n), off = idx * n * n;
    for (let r = 0; r < n; r++) for (let col = 0; col < n; col++) {
      const k = off + r * n + col, rgb = colorFn(sl.bits[k], k);
      const o = ((n - 1 - r) * n + col) * 4; img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
    }
    const cv = document.createElement("canvas"); cv.width = n; cv.height = n; cv.getContext("2d").putImageData(img, 0, 0); return cv;
  }
  const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const RGB = { aqua: hex(C.aqua), blue: hex(C.blue), grey: hex(C.grey), red: hex(C.red) };
  function innerOutline(ctx, m, sl, idx, ext) {
    const n = sl.n, off = idx * n * n, cw = m.w / n, ch = m.h / n;
    ctx.save(); ctx.strokeStyle = C.orange; ctx.lineWidth = 4; ctx.lineCap = "round"; ctx.beginPath();
    const In = (r, c) => r >= 0 && c >= 0 && r < n && c < n && (sl.bits[off + r * n + c] & 4);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      if (!In(r, c)) continue;
      const x0 = m.ox + c * cw, y0 = m.oy + (n - 1 - r) * ch;
      if (!In(r + 1, c)) { ctx.moveTo(x0, y0); ctx.lineTo(x0 + cw, y0); }
      if (!In(r - 1, c)) { ctx.moveTo(x0, y0 + ch); ctx.lineTo(x0 + cw, y0 + ch); }
      if (!In(r, c - 1)) { ctx.moveTo(x0, y0); ctx.lineTo(x0, y0 + ch); }
      if (!In(r, c + 1)) { ctx.moveTo(x0 + cw, y0); ctx.lineTo(x0 + cw, y0 + ch); }
    }
    ctx.stroke(); ctx.restore();
  }

  /* ------------------------------------------------------------ hero: live swarm */
  function hero() {
    const cv = $("hero"), ctx = cv.getContext("2d"), ext = [-2, 2, -2, 2], m = mapper(cv, ext, 70);
    const cars = []; let ok = 0, bad = 0;
    function spawn() {
      for (let tries = 0; tries < 50; tries++) {
        const x = [rand(-1.9, 1.9), rand(-1.9, 1.9), rand(0, 2 * Math.PI), rand(-0.85, 0.7)];
        if (dists("circle", x).some((d) => d <= 0.02)) continue;
        const run = simulate("circle", x, { every: 2, tmax: 24 });
        cars.push({ run, i: 0, born: performance.now(), done: false, fade: 1 }); return;
      }
    }
    for (let i = 0; i < 26; i++) spawn();
    function bg() {
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      for (let g = -2; g <= 2.001; g += 0.25) { ctx.beginPath(); ctx.moveTo(m.X(g), m.Y(-2)); ctx.lineTo(m.X(g), m.Y(2)); ctx.moveTo(m.X(-2), m.Y(g)); ctx.lineTo(m.X(2), m.Y(g)); ctx.stroke(); }
      ctx.strokeStyle = "#cfc5ae"; ctx.lineWidth = 2; ctx.strokeRect(m.X(-2), m.Y(2), m.w, m.h);
      for (let r = 0.25; r < 2.9; r += 0.25) { ctx.beginPath(); ctx.strokeStyle = `rgba(42,120,214,${0.05 + 0.04 * Math.exp(-Math.abs(r - 1) * 2)})`; ctx.lineWidth = 2; ctx.arc(m.X(0), m.Y(0), r * m.s, 0, 2 * Math.PI); ctx.stroke(); }
      ctx.fillStyle = "rgba(227,73,72,.18)"; ctx.beginPath(); ctx.arc(m.X(0), m.Y(0), 0.05 * m.s + 3, 0, 2 * Math.PI); ctx.fill();
      drawPath(ctx, m, "circle", ext, .8);
    }
    let last = performance.now();
    function frame(now) {
      const dt = Math.min(60, now - last); last = now;
      bg();
      for (const car of cars) {
        const pts = car.run.pts;
        if (!car.done) { car.i = Math.min(pts.length - 1, car.i + Math.max(1, Math.round(dt / 16))); if (car.i >= pts.length - 1) { car.done = true; car.doneAt = now; if (car.run.status === "converged") ok++; else if (car.run.status !== "timeout") bad++; } }
        const fate = car.run.status, col = fate === "converged" ? C.blue : fate === "timeout" ? C.ink2 : C.red;
        const alpha = car.done ? Math.max(0, 1 - (now - car.doneAt) / 2600) : 1; car.fade = alpha;
        ctx.save(); ctx.globalAlpha = alpha; ctx.lineWidth = 4; ctx.lineJoin = "round";
        const from = Math.max(0, car.i - 160);
        for (let k = from + 1; k <= car.i; k++) {
          const a = pts[k - 1].x, b = pts[k].x; if (Math.abs(a[0] - b[0]) > 1) continue;
          ctx.strokeStyle = !car.done ? "#5f6b7d" : fate === "converged" ? C.blue : fate === "timeout" ? "#5f6b7d" : C.red;
          ctx.globalAlpha = alpha * (0.15 + 0.85 * (k - from) / (car.i - from + 1)); ctx.beginPath(); ctx.moveTo(m.X(a[0]), m.Y(a[1])); ctx.lineTo(m.X(b[0]), m.Y(b[1])); ctx.stroke();
        }
        ctx.globalAlpha = alpha; const p = pts[car.i].x;
        if (car.done && fate !== "converged" && fate !== "timeout") burst(ctx, m.X(p[0]), m.Y(p[1]), 15); else vehicle(ctx, m.X(p[0]), m.Y(p[1]), (1 + p[3]) >= 0 ? p[2] : p[2] + Math.PI, car.done ? col : "#16202f", 22);
        ctx.restore();
      }
      for (let i = cars.length - 1; i >= 0; i--) if (cars[i].done && cars[i].fade <= 0) { cars.splice(i, 1); spawn(); }
      $("hOk").textContent = ok; $("hBad").textContent = bad;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  /* ------------------------------------------------------------ media cards */
  function media() {
    document.querySelectorAll("video[data-src]").forEach((v) => {
      const b = v.dataset.src; v.poster = b + ".jpg"; v.src = b + ".mp4";
      v.addEventListener("error", () => { const ph = document.createElement("div"); ph.style.cssText = "aspect-ratio:4/3;display:grid;place-items:center;background:#f2f1ee;color:#7a7f88;font:600 13px JetBrains Mono, monospace"; ph.textContent = "animation unavailable"; v.replaceWith(ph); }, { once: true });
    });
  }

  /* ------------------------------------------------------------ launch pad */
  function launchPad() {
    const cv = $("pad"), ctx = cv.getContext("2d");
    let cse = "circle", sl = null, img = null, run = null, play = 0, raf = null, launchedAt = null;
    const spdVals = [0.5, 1.0, 1.5];
    const state = () => ({ hdg: +$("hdg").value, spd: spdVals[+$("spd").value] });
    async function refresh() {
      sl = await loadSlices(cse).catch(() => null);
      cv.height = cse === "sinusoid" ? 700 : 900;
      $("hdgOut").textContent = state().hdg + "°"; $("spdOut").textContent = state().spd.toFixed(1);
      img = null;
      if (sl) { const idx = sliceIndex(sl.meta, state().hdg, state().spd); img = { cv: sliceImage(sl, idx, (b) => (b & 2) && (b & 1) ? RGB.blue : (b & 2) ? RGB.red : (b & 1) ? RGB.aqua : RGB.grey), idx }; }
      draw();
    }
    function ext() { return sl ? sl.meta.extent : (cse === "circle" ? [-2, 2, -2, 2] : [-Math.PI, Math.PI, -2, 2]); }
    function draw() {
      const e = ext(), m = mapper(cv, e, 90); ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cv.width, cv.height);
      if (img) { ctx.imageSmoothingEnabled = false; ctx.drawImage(img.cv, m.ox, m.oy, m.w, m.h); innerOutline(ctx, m, sl, img.idx, e); }
      else { ctx.fillStyle = C.grey; ctx.fillRect(m.ox, m.oy, m.w, m.h); ctx.fillStyle = C.muted; ctx.font = "600 22px JetBrains Mono"; ctx.textAlign = "center"; ctx.fillText("map data loading…", cv.width / 2, cv.height / 2); }
      drawPath(ctx, m, cse, e); axes(ctx, m, e, ["x₁", "x₂"]);
      if (run) {
        const pts = run.pts, n = Math.min(pts.length - 1, Math.floor(play));
        ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 3.5; ctx.beginPath();
        for (let k = 0; k <= n; k++) { const p = pts[k].x; if (k && Math.abs(p[0] - pts[k - 1].x[0]) > 1) { ctx.moveTo(m.X(p[0]), m.Y(p[1])); continue; } k ? ctx.lineTo(m.X(p[0]), m.Y(p[1])) : ctx.moveTo(m.X(p[0]), m.Y(p[1])); }
        ctx.stroke(); ctx.restore();
        const p = pts[n].x;
        if (n === pts.length - 1 && run.status !== "converged" && run.status !== "timeout") burst(ctx, m.X(p[0]), m.Y(p[1]), 14);
        else vehicle(ctx, m.X(p[0]), m.Y(p[1]), (1 + p[3]) >= 0 ? p[2] : p[2] + Math.PI, run.status === "converged" && n === pts.length - 1 ? C.blue : C.ink, 20);
        const s0 = pts[0].x; ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(m.X(s0[0]), m.Y(s0[1]), 7, 0, 7); ctx.fill();
      }
      minis();
    }
    function minis() {
      const n = run ? Math.min(run.pts.length - 1, Math.floor(play)) : -1;
      const T = run ? run.pts[run.pts.length - 1].t : 1;
      plot($("mXi"), run, n, T, (q) => Math.log10(Math.max(1e-6, Math.abs(zvec(cse, q.x)[0]))), [-6, 1], C.ink, null);
      plot($("mSpd"), run, n, T, (q) => 1 + q.x[3], [-0.5, 2], C.ink, 0);
      vplot($("mV"), n, T);
    }
    function plot(cvm, r, n, T, f, yr, col, zero) {
      const c2 = cvm.getContext("2d"), W = cvm.width, H = cvm.height, p = 10;
      c2.clearRect(0, 0, W, H);
      const X = (t) => p + (W - 2 * p) * t / T, Y = (y) => H - p - (H - 2 * p) * (y - yr[0]) / (yr[1] - yr[0]);
      if (zero != null) { c2.fillStyle = "rgba(227,73,72,.13)"; c2.fillRect(p, Y(0.02), W - 2 * p, Y(-0.02) - Y(0.02) + 1); c2.strokeStyle = C.red; c2.setLineDash([5, 4]); c2.beginPath(); c2.moveTo(p, Y(0)); c2.lineTo(W - p, Y(0)); c2.stroke(); c2.setLineDash([]); }
      if (!r) return;
      c2.strokeStyle = col; c2.lineWidth = 2.5; c2.beginPath();
      for (let k = 0; k <= n; k++) { const q = r.pts[k]; const y = Math.min(yr[1], Math.max(yr[0], f(q))); k ? c2.lineTo(X(q.t), Y(y)) : c2.moveTo(X(q.t), Y(y)); }
      c2.stroke();
    }
    let Vtr = null;
    function computeV() {
      if (!run || !CONST) { Vtr = null; return; }
      const g = CONST[cse].gamma, pts = run.pts, q = pts.map((s) => g * qex(cse, s.x)), psi = pts.map((s, i) => Math.max(0, hobs(cse, s.x) - q[i]));
      const failed = run.status === "singular" || run.status === "box", Phi = new Array(pts.length); let mx = failed ? Infinity : 0;
      for (let i = pts.length - 1; i >= 0; i--) { mx = Math.max(mx, psi[i]); Phi[i] = mx; }
      Vtr = { q, Phi, failed };
    }
    function vplot(cvm, n, T) {
      const c2 = cvm.getContext("2d"), W = cvm.width, H = cvm.height, p = 10; c2.clearRect(0, 0, W, H);
      if (!run || !Vtr) return;
      if (Vtr.failed) { c2.fillStyle = C.red; c2.font = "600 17px Inter"; c2.fillText("V = ∞ : this run leaves the admissible set", p + 4, H / 2 + 6); return; }
      let ymax = 0.5; for (let k = 0; k < Vtr.q.length; k++) ymax = Math.max(ymax, Vtr.q[k] + Vtr.Phi[k]); ymax = Math.min(ymax * 1.08, 60);
      const X = (t) => p + (W - 2 * p) * t / T, Y = (y) => H - p - (H - 2 * p) * Math.min(1, y / ymax);
      c2.globalAlpha = .9; c2.fillStyle = C.orange; c2.beginPath(); c2.moveTo(X(0), Y(0));
      for (let k = 0; k <= n; k++) c2.lineTo(X(run.pts[k].t), Y(Vtr.q[k])); c2.lineTo(X(run.pts[n].t), Y(0)); c2.fill();
      c2.fillStyle = C.blue; c2.beginPath(); for (let k = 0; k <= n; k++) c2.lineTo(X(run.pts[k].t), Y(Vtr.q[k] + Vtr.Phi[k]));
      for (let k = n; k >= 0; k--) c2.lineTo(X(run.pts[k].t), Y(Vtr.q[k])); c2.fill(); c2.globalAlpha = 1;
    }
    function verdict() {
      const el = $("verdict"); el.innerHTML = "";
      const add = (cls, t) => { const s = document.createElement("span"); s.className = "chip " + cls; s.textContent = t; el.appendChild(s); };
      if (!run) return add("neutral", "click the map to launch");
      if (sl && img) {
        const e = ext(), x = run.pts[0].x, n = sl.n, col = Math.round((x[0] - e[0]) / (e[1] - e[0]) * (n - 1)), row = Math.round((x[1] - e[2]) / (e[3] - e[2]) * (n - 1));
        const b = sl.bits[img.idx * n * n + row * n + col];
        add(b & 2 ? "ok" : "neutral", b & 4 ? "starts in the verified inner layer" : b & 2 ? "starts in our estimate" : "starts outside our estimate");
      }
      const done = play >= run.pts.length - 1;
      if (!done) return add("neutral", "running…");
      add(run.status === "converged" ? "ok" : run.status === "timeout" ? "neutral" : "bad",
        run.status === "converged" ? "converged to the path" : run.status === "singular" ? "reached the singular set" : run.status === "box" ? "left the state constraints" : "still running at the horizon");
    }
    cv.addEventListener("click", (ev) => {
      const r = cv.getBoundingClientRect(), px = (ev.clientX - r.left) * cv.width / r.width, py = (ev.clientY - r.top) * cv.height / r.height;
      const e = ext(), m = mapper(cv, e, 90), xy = m.inv(px, py);
      if (xy[0] < e[0] || xy[0] > e[1] || xy[1] < e[2] || xy[1] > e[3]) return;
      const x0 = [xy[0], xy[1], state().hdg * Math.PI / 180, state().spd - 1];
      if (dists(cse, x0).some((d) => d <= 0)) return;
      run = simulate(cse, x0, { every: 1, tmax: 30 }); computeV(); play = 0; launchedAt = performance.now();
      cancelAnimationFrame(raf);
      const step = () => { play += Math.max(2, run.pts.length / 260); draw(); verdict(); if (play < run.pts.length - 1) raf = requestAnimationFrame(step); else { play = run.pts.length - 1; draw(); verdict(); } };
      raf = requestAnimationFrame(step);
    });
    $("hdg").addEventListener("input", refresh); $("spd").addEventListener("input", refresh);
    $("padCase").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; cse = b.dataset.case; [...$("padCase").children].forEach((x) => x.classList.toggle("on", x === b)); run = null; verdict(); refresh(); });
    refresh(); verdict();
  }

  /* ------------------------------------------------------------ decomposition panel */
  function splitPanel() {
    const cv = $("split"), ctx = cv.getContext("2d"), c = "circle";
    let best = null;
    function pick() {
      if (!CONST) return;
      let score = -1;
      for (let a = 0; a < 24; a++) for (const r of [1.4, 1.7]) for (const hd of [0, 0.5, 1]) {
        const x0 = [r * Math.cos(a * Math.PI / 12), r * Math.sin(a * Math.PI / 12), a * Math.PI / 12 + hd, 0.2];
        const run = simulate(c, x0, { every: 1, tmax: 22 }); if (run.status !== "converged") continue;
        const g = CONST[c].gamma; let mx = 0; for (const s of run.pts) mx = Math.max(mx, hobs(c, s.x) - g * qex(c, s.x));
        if (mx > score && mx < 40) { score = mx; best = run; }
      }
      if (!best) return;
      const g = CONST[c].gamma, pts = best.pts, q = pts.map((s) => g * qex(c, s.x)), psi = pts.map((s, i) => Math.max(0, hobs(c, s.x) - q[i]));
      const Phi = new Array(pts.length); let mx = 0; for (let i = pts.length - 1; i >= 0; i--) { mx = Math.max(mx, psi[i]); Phi[i] = mx; }
      best.q = q; best.Phi = Phi;
    }
    function draw(f) {
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cv.width, cv.height);
      if (!best) { ctx.fillStyle = C.muted; ctx.font = "600 20px JetBrains Mono"; ctx.fillText("loading…", 30, 50); return; }
      const pts = best.pts, k = Math.min(pts.length - 1, Math.round(f * (pts.length - 1)));
      const ext = [-2, 2, -2, 2], m = { X: (x) => 30 + (x + 2) * 100, Y: (y) => 450 - (y + 2) * 100, s: 100 };
      ctx.strokeStyle = C.grid; ctx.strokeRect(30, 50, 400, 400); drawPath(ctx, m, c, ext, .55);
      ctx.fillStyle = "rgba(227,73,72,.2)"; ctx.beginPath(); ctx.arc(m.X(0), m.Y(0), 7, 0, 7); ctx.fill();
      ctx.strokeStyle = "#b8c2d0"; ctx.lineWidth = 2.5; ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(m.X(p.x[0]), m.Y(p.x[1])) : ctx.moveTo(m.X(p.x[0]), m.Y(p.x[1])))); ctx.stroke();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 3; ctx.beginPath(); for (let i = 0; i <= k; i++) { const p = pts[i].x; i ? ctx.lineTo(m.X(p[0]), m.Y(p[1])) : ctx.moveTo(m.X(p[0]), m.Y(p[1])); } ctx.stroke();
      const x = pts[k].x; vehicle(ctx, m.X(x[0]), m.Y(x[1]), (1 + x[3]) >= 0 ? x[2] : x[2] + Math.PI, C.ink, 17);
      let vmax = 0; for (let i = 0; i < pts.length; i++) vmax = Math.max(vmax, best.q[i] + best.Phi[i]);
      const bx = 500, by0 = 450, bh = 390, sc = bh / vmax, qv = best.q[k], pv = best.Phi[k];
      ctx.fillStyle = C.orange; ctx.fillRect(bx, by0 - qv * sc, 70, qv * sc);
      ctx.fillStyle = C.blue; ctx.fillRect(bx, by0 - (qv + pv) * sc, 70, pv * sc);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bx - 10, by0); ctx.lineTo(bx + 80, by0); ctx.stroke();
      ctx.font = "600 19px Inter"; ctx.fillStyle = C.ink; ctx.textAlign = "left";
      ctx.fillText("V = " + (qv + pv).toFixed(2), bx + 84, by0 - (qv + pv) * sc + 6);
      ctx.fillStyle = C.orange; ctx.fillText("γ qₑₓ " + qv.toFixed(2), bx + 84, Math.min(by0 - 8, by0 - qv * sc / 2 + 6));
      ctx.fillStyle = C.blue; if (pv > 0.02) ctx.fillText("Φ " + pv.toFixed(2), bx + 84, by0 - qv * sc - pv * sc / 2 + 6);
      ctx.fillStyle = C.ink2; ctx.font = "600 15px JetBrains Mono"; ctx.fillText("speed " + (1 + x[3]).toFixed(2), 40, 80);
      $("tOut").textContent = pts[k].t.toFixed(1);
    }
    const sl = $("tSlide"); let auto = true, f = 0;
    sl.addEventListener("input", () => { auto = false; f = sl.value / 1000; draw(f); });
    const io = new IntersectionObserver((es) => { if (es[0].isIntersecting && !best) { pick(); draw(0); } });
    io.observe(cv);
    (function loop() { if (auto && best) { f = (f + 0.0025) % 1.15; sl.value = Math.min(1000, f * 1000); draw(Math.min(1, f)); } requestAnimationFrame(loop); })();
    draw(0);
  }

  /* ------------------------------------------------------------ level explorer */
  function levelPanel() {
    const cv = $("lvl"), ctx = cv.getContext("2d"); let cse = "circle", sl = null;
    const spdVals = [0.5, 1.0, 1.5], VMAX = 8;
    const lvlV = () => VMAX * $("lv").value / 1000;
    let snapped = false;
    async function refresh() { sl = await loadSlices(cse).catch(() => null); cv.height = cse === "sinusoid" ? 700 : 900; if (sl && !snapped) { snapped = true; $("lv").value = Math.round(-Math.log(1 - sl.meta.c2_seed0) / VMAX * 1000); } draw(); }
    function draw() {
      const hdg = +$("lh").value, spd = spdVals[+$("ls").value];
      $("lhOut").textContent = hdg + "°"; $("lsOut").textContent = spd.toFixed(1);
      const Vc = lvlV(); $("lvOut").textContent = Vc.toFixed(2) + "  (c = " + (1 - Math.exp(-Vc)).toFixed(4) + ")";
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cv.width, cv.height);
      const e = sl ? sl.meta.extent : [-2, 2, -2, 2], m = mapper(cv, e, 90);
      if (!sl) { ctx.fillStyle = C.muted; ctx.font = "600 22px JetBrains Mono"; ctx.textAlign = "center"; ctx.fillText("map data loading…", cv.width / 2, cv.height / 2); return; }
      const idx = sliceIndex(sl.meta, hdg, spd), n = sl.n, off = idx * n * n, thr = Vc * 1000;
      let roaN = 0, cov = 0, fp = 0;
      const im = sliceImage(sl, idx, (b, k) => {
        const inR = (b & 4) || sl.V[k] < thr, roa = b & 1;
        if (roa) roaN++; if (inR && roa) cov++; if (inR && !roa) fp++;
        return inR ? (roa ? RGB.blue : RGB.red) : roa ? RGB.aqua : RGB.grey;
      });
      ctx.imageSmoothingEnabled = false; ctx.drawImage(im, m.ox, m.oy, m.w, m.h); innerOutline(ctx, m, sl, idx, e);
      drawPath(ctx, m, cse, e); axes(ctx, m, e, ["x₁", "x₂"]);
      $("rCov").textContent = fmtPct(cov / Math.max(1, roaN)); $("rFp").textContent = fp; $("rFp").style.color = fp ? C.red : C.ink;
      const c2 = sl.meta.c2_seed0; $("rC2").textContent = c2.toFixed(4);
    }
    $("snap").addEventListener("click", () => { if (!sl) return; const V2 = -Math.log(1 - sl.meta.c2_seed0); $("lv").value = Math.round(V2 / VMAX * 1000); draw(); });
    ["lv", "lh", "ls"].forEach((id) => $(id).addEventListener("input", draw));
    $("lvlCase").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; cse = b.dataset.case; snapped = false; [...$("lvlCase").children].forEach((x) => x.classList.toggle("on", x === b)); refresh(); });
    refresh();
  }

  /* ------------------------------------------------------------ scenario / Clopper-Pearson */
  function lgamma(z) { const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z); z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i); const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x); }
  function binCdf(k, N, p) { let s = 0; for (let i = 0; i <= k; i++) s += Math.exp(lgamma(N + 1) - lgamma(i + 1) - lgamma(N - i + 1) + i * Math.log(p) + (N - i) * Math.log1p(-p)); return s; }
  function cpUpper(k, N, delta) { let lo = 1e-12, hi = 1; for (let it = 0; it < 200; it++) { const mid = Math.sqrt(lo * hi); if (binCdf(k, N, mid) > delta) lo = mid; else hi = mid; } return hi; }
  function odds() {
    const cv = $("ruler"), ctx = cv.getContext("2d");
    function draw() {
      const N = Math.round(10 ** +$("nS").value), k = +$("kS").value, e = cpUpper(k, N, 1e-6);
      $("nOut").textContent = N.toLocaleString("en-US"); $("kOut").textContent = k;
      $("eps").innerHTML = e < 1e-3 ? `1 in ${Math.round(1 / e).toLocaleString("en-US")}` : (100 * e).toFixed(2) + "%";
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cv.width, cv.height);
      const x0 = 40, x1 = 560, y = 150, L = (v) => x0 + (x1 - x0) * (-Math.log10(v)) / 7;
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.font = "600 15px JetBrains Mono"; ctx.textAlign = "center"; ctx.fillStyle = C.ink2;
      for (let d = 0; d <= 7; d++) { const X = L(10 ** -d); ctx.beginPath(); ctx.moveTo(X, y - 8); ctx.lineTo(X, y + 8); ctx.stroke(); ctx.fillText(d === 0 ? "1" : "10⁻" + "⁰¹²³⁴⁵⁶⁷"[d], X, y + 32); }
      const X = L(Math.max(1e-7, e)); ctx.fillStyle = C.red; ctx.beginPath(); ctx.moveTo(X, y - 12); ctx.lineTo(X - 11, y - 34); ctx.lineTo(X + 11, y - 34); ctx.fill();
      ctx.fillText("ε", X, y - 44);
      ctx.fillStyle = "rgba(227,73,72,.15)"; ctx.fillRect(X, y - 6, x1 - X, 12);
      ctx.fillStyle = C.ink2; ctx.font = "500 14px Inter"; ctx.fillText("share of the box (log scale)", (x0 + x1) / 2, y + 70);
    }
    ["nS", "kS"].forEach((id) => $(id).addEventListener("input", draw)); draw();
  }

  /* ------------------------------------------------------------ results dot plot */
  function results() {
    if (!RES) return;
    const cv = $("dots"), ctx = cv.getContext("2d"), tip = $("tip");
    const M = [
      { key: "ours", label: "Ours", col: C.blue, shape: "circle" },
      { key: "exact", label: "Exact value function at the same level", col: C.blue, shape: "ring" },
      { key: "nodecomp", label: "No decomposition", col: C.aquaD, shape: "tri" },
      { key: "pinn", label: "Zubov PINN", col: "#c98500", shape: "square" },
      { key: "dz", label: "Dirichlet-Zubov PINN", col: "#d55181", shape: "diamond" },
      { key: "quad", label: "Quadratic", col: "#8a8f98", shape: "x" },
    ];
    const on = Object.fromEntries(M.map((m) => [m.key, true]));
    const tools = $("resTools"); tools.innerHTML = "";
    M.forEach((m) => { const b = document.createElement("button"); b.type = "button"; b.className = "toggle"; b.setAttribute("aria-pressed", "true"); b.innerHTML = `<i style="background:${m.col}"></i>${m.label}`; b.onclick = () => { on[m.key] = !on[m.key]; b.classList.toggle("off", !on[m.key]); b.setAttribute("aria-pressed", String(on[m.key])); draw(); }; tools.appendChild(b); });
    const sys = RES.systems, rowH = 50, top = 40, left = 300, right = 1060;
    const rows = []; let y = top, grp = null;
    sys.forEach((s) => { if (s.group !== grp) { grp = s.group; rows.push({ head: grp, y }); y += 38; } rows.push({ s, y }); y += rowH; });
    cv.height = y + 50;
    const X = (v) => left + (right - left) * v / 100; let hits = [];
    function mark(m, px, py) {
      ctx.fillStyle = m.col; ctx.strokeStyle = m.col; ctx.lineWidth = 3; const r = 9; ctx.beginPath();
      if (m.shape === "circle") { ctx.arc(px, py, r, 0, 7); ctx.fill(); }
      else if (m.shape === "ring") { ctx.arc(px, py, r + 2, 0, 7); ctx.stroke(); }
      else if (m.shape === "square") { ctx.fillRect(px - r + 1, py - r + 1, 2 * r - 2, 2 * r - 2); }
      else if (m.shape === "diamond") { ctx.moveTo(px, py - r - 2); ctx.lineTo(px + r + 2, py); ctx.lineTo(px, py + r + 2); ctx.lineTo(px - r - 2, py); ctx.closePath(); ctx.fill(); }
      else if (m.shape === "tri") { ctx.moveTo(px, py - r - 1); ctx.lineTo(px + r + 1, py + r); ctx.lineTo(px - r - 1, py + r); ctx.closePath(); ctx.fill(); }
      else { ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r); ctx.moveTo(px + r, py - r); ctx.lineTo(px - r, py + r); ctx.stroke(); }
    }
    function draw() {
      hits = []; ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.font = "500 15px JetBrains Mono"; ctx.fillStyle = C.muted; ctx.textAlign = "center";
      for (let v = 0; v <= 100; v += 20) { ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(v), top - 10); ctx.lineTo(X(v), cv.height - 40); ctx.stroke(); ctx.fillText(v + "%", X(v), cv.height - 16); }
      for (const r of rows) {
        if (r.head) { ctx.textAlign = "left"; ctx.fillStyle = C.orange; ctx.font = "700 14px JetBrains Mono"; ctx.fillText(r.head.toUpperCase(), 20, r.y + 16); continue; }
        const s = r.s, yy = r.y + rowH / 2;
        ctx.textAlign = "left"; ctx.fillStyle = C.ink; ctx.font = "600 17px Inter"; ctx.fillText(s.name, 20, yy + 6);
        ctx.strokeStyle = "#e3dccb"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
        for (const m of M) {
          if (!on[m.key]) continue; const val = s[m.key];
          if (!val) continue; const px = X(val[0]);
          if (val[1]) { ctx.strokeStyle = m.col; ctx.globalAlpha = .45; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(Math.max(0, val[0] - val[1])), yy); ctx.lineTo(X(Math.min(100, val[0] + val[1])), yy); ctx.stroke(); ctx.globalAlpha = 1; }
          mark(m, px, yy); hits.push({ x: px, y: yy, t: `<b>${s.name}</b><br>${m.label}: ${val[0].toFixed(1)}%${val[1] ? " ± " + val[1].toFixed(1) : ""}` });
        }
      }
    }
    cv.addEventListener("mousemove", (e) => {
      const r = cv.getBoundingClientRect(), px = (e.clientX - r.left) * cv.width / r.width, py = (e.clientY - r.top) * cv.height / r.height;
      let best = null, bd = 400; for (const h of hits) { const d = (h.x - px) ** 2 + (h.y - py) ** 2; if (d < bd) { bd = d; best = h; } }
      if (best) { tip.innerHTML = best.t; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px"; tip.classList.add("on"); } else tip.classList.remove("on");
    });
    cv.addEventListener("mouseleave", () => tip.classList.remove("on"));
    draw();
  }

  /* ------------------------------------------------------------ step setting table */
  function stepTable() {
    const el = $("stepTbl"); if (!RES || !RES.step || !el) return;
    const f = (v) => v.toFixed(1), b = (v, on) => (on ? `<b>${f(v)}</b>` : f(v));
    let h = "<tr><th>System</th><th>Share, long</th><th>Share, short</th><th>Kept</th><th>Coverage</th></tr>";
    for (const r of RES.step) h += `<tr><td>${r.name}</td><td>${b(r.share_long, r.kept === "long")}</td><td>${b(r.share_short, r.kept === "short")}</td><td>${r.kept}</td><td>${f(r.cov_kept)} %</td></tr>`;
    el.innerHTML = h;
  }

  /* ------------------------------------------------------------ misc */
  function fillMacros() { if (!RES) return; document.querySelectorAll("[data-m]").forEach((el) => { const v = RES.macros[el.dataset.m]; if (v != null) el.textContent = v.replace(/%$/, " %").replace(/(\d)e(-?\d+)/, "$1×10^$2").replace(/\^(-?\d+)/, (s, p) => p.split("").map((ch) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(ch)]).join("")); }); }
  let mathTries = 0;
  function math() {
    if (!window.katex) { if (++mathTries < 50) setTimeout(math, 100); else $("eqSplit").textContent = "V(x) = γ q_ex(x) + Φ(x)"; return; }
    katex.render("V(x)=\\underbrace{\\gamma\\,q_{\\mathrm{ex}}(x)}_{\\text{known, decreasing}}+\\underbrace{\\sup_{t\\ge0}\\ \\mathrm{relu}\\big(h-\\gamma q_{\\mathrm{ex}}\\big)(y(t;x))}_{\\Phi(x):\\ \\text{safety value function, learned}}", $("eqSplit"), { displayMode: true, throwOnError: false });
  }
  $("copyBib").addEventListener("click", () => { navigator.clipboard.writeText($("bib").textContent).catch(() => {}); $("copyBib").textContent = "copied"; setTimeout(() => ($("copyBib").textContent = "copy"), 1500); });

  async function main() {
    media(); math(); odds();
    try { CONST = await getJSON("constants.json"); } catch (e) { CONST = null; }
    try { RES = await getJSON("results.json"); } catch (e) { RES = null; }
    fillMacros(); hero(); launchPad(); splitPanel(); levelPanel(); results(); stepTable();
  }
  main();
})();
