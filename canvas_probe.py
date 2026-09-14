#!/usr/bin/env python3
"""Execute generated canvas code in V8 and verify the result numerically.

Motivation: static analysis scored a Kepler animation at 83% when it actually
rendered a 1-pixel circle with the star at its centre. Greps cannot see that.
This module runs the model's own JavaScript against a recording canvas shim,
pumps requestAnimationFrame on a controlled clock, and then measures what was
actually drawn:

  - orbit scale      (catches the 1-pixel orbit)
  - eccentricity     (catches a circle drawn instead of an ellipse)
  - focus placement  (catches the star drawn at the centre)
  - Kepler's 2nd law (catches uniform angular motion faked as an orbit)

Nothing here depends on the model's variable names or comments.
"""
import json
import math
import re

from py_mini_racer import MiniRacer

# --------------------------------------------------------------------------
# Browser shim. Every 2D context call is recorded with the frame index.
# --------------------------------------------------------------------------
SHIM = r"""
var __log = [];
var __frame = 0;
var __t = 0;            // virtual clock, ms
var __rafQueue = [];
var __intervals = [];
// Virtual wall clock. Pages timing their animation with Date.now() were
// graded as barely moving: 400 simulated frames take milliseconds of REAL
// time. Date must follow the same virtual clock as performance.now().
var __T0 = 1789300000000;
var __RealDate = Date;
Date = class extends __RealDate {
  constructor(...a) { if (a.length === 0) { super(__T0 + __t); } else { super(...a); } }
  static now() { return __T0 + __t; }
};

function __mkCtx() {
  var c = {
    canvas: null,
    fillStyle: '#000', strokeStyle: '#000', lineWidth: 1,
    shadowBlur: 0, shadowColor: '#000', globalAlpha: 1, font: '10px sans-serif',
    lineCap: 'butt', lineJoin: 'miter', textAlign: 'start', textBaseline: 'alphabetic',
    _path: [],
  };
  // Current transform [a b c d e f] (device = a*x + c*y + e, b*x + d*y + f).
  // Pages that draw in physical units via translate/scale are correct; without
  // tracking the matrix their orbit looked a few pixels wide.
  c._m = [1, 0, 0, 1, 0, 0];
  c._stack = [];
  function mul(m, n) {
    return [m[0]*n[0] + m[2]*n[1], m[1]*n[0] + m[3]*n[1],
            m[0]*n[2] + m[2]*n[3], m[1]*n[2] + m[3]*n[3],
            m[0]*n[4] + m[2]*n[5] + m[4], m[1]*n[4] + m[3]*n[5] + m[5]];
  }
  function dev(x, y) { var m = c._m; return [m[0]*x + m[2]*y + m[4], m[1]*x + m[3]*y + m[5]]; }
  function scl() { var m = c._m; return Math.sqrt(Math.abs(m[0]*m[3] - m[1]*m[2])); }
  function rec(op, args) { __log.push({f: __frame, op: op, a: args}); }
  c.beginPath   = function () { c._path = []; rec('beginPath', []); };
  c.closePath   = function () { rec('closePath', []); };
  c.moveTo      = function (x, y) { var p = dev(x, y); c._path.push(p); rec('moveTo', p); };
  c.lineTo      = function (x, y) { var p = dev(x, y); c._path.push(p); rec('lineTo', p); };
  c.arc         = function (x, y, r, s, e, cc) { var p = dev(x, y); rec('arc', [p[0], p[1], r * scl(), s, e]); };
  c.ellipse     = function (x, y, rx, ry, rot, s, e, cc) { var p = dev(x, y); rec('ellipse', [p[0], p[1], rx * scl(), ry * scl(), rot, s, e]); };
  c.rect        = function (x, y, w, h) { var p = dev(x, y); rec('rect', [p[0], p[1], w * scl(), h * scl()]); };
  c.fillRect    = function (x, y, w, h) { var p = dev(x, y); rec('fillRect', [p[0], p[1], w * scl(), h * scl()]); };
  c.clearRect   = function (x, y, w, h) { var p = dev(x, y); rec('clearRect', [p[0], p[1], w * scl(), h * scl()]); };
  c.strokeRect  = function (x, y, w, h) { var p = dev(x, y); rec('strokeRect', [p[0], p[1], w * scl(), h * scl()]); };
  c.fill        = function () { rec('fill', [c.fillStyle]); };
  c.stroke      = function () { rec('stroke', [c.strokeStyle, c.lineWidth]); };
  c.fillText    = function (t, x, y) { var p = dev(x, y); rec('fillText', [String(t), p[0], p[1]]); };
  c.strokeText  = function (t, x, y) { var p = dev(x, y); rec('strokeText', [String(t), p[0], p[1]]); };
  c.save = function () { c._stack.push(c._m.slice()); };
  c.restore = function () { if (c._stack.length) c._m = c._stack.pop(); };
  c.translate = function (x, y) { c._m = mul(c._m, [1, 0, 0, 1, x, y]); };
  c.rotate = function (a) { var co = Math.cos(a), si = Math.sin(a); c._m = mul(c._m, [co, si, -si, co, 0, 0]); };
  c.scale = function (x, y) { c._m = mul(c._m, [x, 0, 0, y, 0, 0]); };
  c.transform = function (a, b, cc, d, e, f) { c._m = mul(c._m, [a, b, cc, d, e, f]); };
  c.setTransform = function (a, b, cc, d, e, f) {
    if (typeof a === 'object' && a) { c._m = [a.a, a.b, a.c, a.d, a.e, a.f]; }
    else if (arguments.length >= 6) { c._m = [a, b, cc, d, e, f]; }
    else { c._m = [1, 0, 0, 1, 0, 0]; }
  };
  c.resetTransform = function () { c._m = [1, 0, 0, 1, 0, 0]; };
  c.createRadialGradient = function () {
    return {addColorStop: function () {}};
  };
  c.createLinearGradient = c.createRadialGradient;
  c.createPattern = function () { return null; };
  c.measureText = function (t) { return {width: String(t).length * 6}; };
  c.drawImage = function () {}; c.putImageData = function () {};
  c.getImageData = function (x, y, w, h) {
    return {data: new Array(Math.max(1, w * h * 4)).fill(0), width: w, height: h};
  };
  c.setLineDash = function () {}; c.quadraticCurveTo = function () {};
  c.bezierCurveTo = function () {}; c.arcTo = function () {}; c.clip = function () {};
  return c;
}

function __mkEl(id, tag) {
  var el = {
    id: id, tagName: (tag || 'div').toUpperCase(),
    width: 800, height: 600,
    style: {}, dataset: {},
    textContent: '', innerHTML: '', innerText: '', value: '',
    className: '', classList: {add: function () {}, remove: function () {},
                               toggle: function () {}, contains: function () { return false; }},
    children: [], childNodes: [],
  };
  el._ctx = null;
  el.getContext = function () {
    if (!el._ctx) { el._ctx = __mkCtx(); el._ctx.canvas = el; }
    return el._ctx;
  };
  el.addEventListener = function () {}; el.removeEventListener = function () {};
  el.appendChild = function (c) { el.children.push(c); return c; };
  el.setAttribute = function (k, v) { el[k] = v; };
  el.getAttribute = function (k) { return el[k]; };
  el.getBoundingClientRect = function () {
    return {left: 0, top: 0, width: el.width, height: el.height, right: el.width, bottom: el.height};
  };
  el.focus = function () {}; el.querySelector = function () { return __mkEl('q'); };
  el.querySelectorAll = function () { return []; };
  return el;
}

var __els = {};
var document = {
  getElementById: function (id) {
    if (!__els[id]) __els[id] = __mkEl(id, /canvas/i.test(id) ? 'canvas' : 'div');
    return __els[id];
  },
  querySelector: function (s) {
    var id = String(s).replace(/^[#.]/, '');
    return document.getElementById(id);
  },
  querySelectorAll: function () { return []; },
  createElement: function (tag) { return __mkEl('created_' + tag, tag); },
  addEventListener: function (ev, cb) { if (ev === 'DOMContentLoaded') __rafQueue.push(cb); },
  body: __mkEl('body', 'body'),
  documentElement: __mkEl('html', 'html'),
  head: __mkEl('head', 'head'),
  title: '',
};
document.body.appendChild = function (c) { return c; };

var window = {
  innerWidth: 1280, innerHeight: 800,
  devicePixelRatio: 1,
  addEventListener: function (ev, cb) { if (ev === 'load') __rafQueue.push(cb); },
  removeEventListener: function () {},
  requestAnimationFrame: function (cb) { __rafQueue.push(cb); return __rafQueue.length; },
  cancelAnimationFrame: function () {},
  setTimeout: function (cb) { __rafQueue.push(cb); return 0; },
  // setInterval must REPEAT. It used to be queued once like setTimeout, so any
  // page animating with setInterval (valid, and what Granite wrote) ran one
  // frame and was graded as drawing nothing.
  setInterval: function (cb) { __intervals.push(cb); return __intervals.length; },
  clearInterval: function (id) { if (id > 0) __intervals[id - 1] = null; },
  clearTimeout: function () {},
  performance: {now: function () { return __t; }},
  document: document,
  getComputedStyle: function () { return {}; },
  alert: function () {}, console: {log: function () {}},
};
var requestAnimationFrame = window.requestAnimationFrame;
var cancelAnimationFrame = window.cancelAnimationFrame;
var setTimeout = window.setTimeout;
var setInterval = window.setInterval;
var clearInterval = window.clearInterval;
var clearTimeout = window.clearTimeout;
var performance = window.performance;
var navigator = {userAgent: 'probe'};
var console = {log: function () {}, warn: function () {}, error: function () {},
               info: function () {}, debug: function () {}};
var alert = function () {};

// Pump: run queued callbacks for one frame at a fixed timestep.
function __pump(dtMs) {
  __t += dtMs;
  __frame += 1;
  var q = __rafQueue;
  __rafQueue = [];
  for (var i = 0; i < q.length; i++) {
    try { q[i](__t); } catch (e) { __log.push({f: __frame, op: '__error', a: [String(e)]}); }
  }
  for (var j = 0; j < __intervals.length; j++) {
    if (!__intervals[j]) continue;
    try { __intervals[j](); } catch (e) { __log.push({f: __frame, op: '__error', a: [String(e)]}); }
  }
  return __rafQueue.length;
}
function __dump() { return JSON.stringify(__log); }
function __canvasSize() {
  var out = [];
  for (var k in __els) if (__els[k]._ctx) out.push([k, __els[k].width, __els[k].height]);
  return JSON.stringify(out);
}
"""


def extract_js(html):
    """Concatenate every inline <script> body in document order."""
    return "\n;\n".join(
        m.group(1) for m in re.finditer(r"<script[^>]*>(.*?)</script>", html, re.S | re.I)
    )


def run_canvas(html, frames=240, dt_ms=16.0, timeout_ms=20000):
    """Execute the page's JS and return (draw_log, canvases, error_or_None)."""
    js = extract_js(html)
    if not js.strip():
        return [], [], "no <script> found"
    ctx = MiniRacer()
    try:
        ctx.eval(SHIM, timeout=timeout_ms)
    except Exception as e:
        return [], [], f"shim failed: {e}"
    try:
        ctx.eval(js, timeout=timeout_ms)
    except Exception as e:
        return [], [], f"script threw on load: {str(e)[:200]}"
    try:
        for _ in range(frames):
            ctx.eval(f"__pump({dt_ms})", timeout=timeout_ms)
        log = json.loads(ctx.eval("__dump()", timeout=timeout_ms))
        canvases = json.loads(ctx.eval("__canvasSize()", timeout=timeout_ms))
    except Exception as e:
        return [], [], f"run failed: {str(e)[:200]}"
    return log, canvases, None


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def fit_ellipse(pts):
    """Least-squares conic fit. Returns (cx, cy, a, b, theta) or None.

    Solves the general conic Ax^2+Bxy+Cy^2+Dx+Ey+F=0 with F fixed to -1, which
    is fine for any ellipse not passing through the origin of the shifted frame.
    """
    if len(pts) < 12:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    P = [(x - mx, y - my) for x, y in pts]
    s = max(max(abs(x), abs(y)) for x, y in P) or 1.0
    P = [(x / s, y / s) for x, y in P]

    # Normal equations for [A B C D E] . v = 1  (F = -1)
    rows = [[x * x, x * y, y * y, x, y] for x, y in P]
    n = 5
    ATA = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    ATb = [sum(r[i] for r in rows) for i in range(n)]
    # Gaussian elimination with partial pivoting
    M = [ATA[i][:] + [ATb[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    A, B, C, D, E = (M[i][n] for i in range(n))
    F = -1.0

    disc = B * B - 4 * A * C
    if disc >= 0:                      # not an ellipse (parabola/hyperbola)
        return None
    cx = (2 * C * D - B * E) / disc
    cy = (2 * A * E - B * D) / disc
    num = 2 * (A * E * E + C * D * D + F * B * B - B * D * E - 4 * A * C * F)
    root = math.sqrt(max((A - C) ** 2 + B * B, 0.0))
    da = A + C + root
    db = A + C - root
    # Semi-axes are  -sqrt(num * (A+C +/- root)) / disc.  The leading minus is
    # load-bearing: disc < 0 for an ellipse, so without it both axes come out
    # negative and the fit is rejected. (v1 of this function omitted it and
    # failed on a synthetic a=200,b=160 ellipse -- caught by the self-test.)
    if num * da <= 0 or num * db <= 0:
        return None
    ax1 = -math.sqrt(num * da) / disc
    ax2 = -math.sqrt(num * db) / disc
    if ax1 <= 0 or ax2 <= 0:
        return None
    a, b = max(ax1, ax2), min(ax1, ax2)
    # Major-axis angle from the eigenvector of [[A, B/2], [B/2, C]] belonging to
    # the SMALLER eigenvalue (small eigenvalue <-> long axis). The textbook
    # 0.5*atan2(B, A-C) form is off by 90 degrees for an axis-aligned ellipse
    # with a > b -- it returned pi/2 where the major axis was along x, which
    # put the computed foci perpendicular to the real ones.
    lam = 0.5 * ((A + C) - root)
    vx, vy = (B / 2.0), (lam - A)
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        vx, vy = (lam - C), (B / 2.0)
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        vx, vy = (1.0, 0.0) if A <= C else (0.0, 1.0)
    theta = math.atan2(vy, vx)
    return (cx * s + mx, cy * s + my, a * s, b * s, theta)


def analyse_orbit(log, canvases):
    """Classify drawn marks into star/planet and measure the orbit.

    Heuristic, but name-agnostic: among filled arcs, the mark whose centre
    barely moves across frames is the star; the one that traverses the largest
    path is the planet.
    """
    arcs = {}
    for e in log:
        if e["op"] != "arc":
            continue
        x, y, r = e["a"][0], e["a"][1], e["a"][2]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, r)):
            continue
        key = round(r, 1)
        arcs.setdefault(key, []).append((e["f"], x, y))

    # Frames on which the whole canvas was wiped (clearRect/fillRect covering it).
    cw0 = canvases[0][1] if canvases else 800
    ch0 = canvases[0][2] if canvases else 600
    wipes = [e["f"] for e in log if e["op"] in ("clearRect", "fillRect")
             and len(e["a"]) >= 4 and all(isinstance(v, (int, float)) for v in e["a"][:4])
             and e["a"][2] >= cw0 * 0.9 and e["a"][3] >= ch0 * 0.9]
    last_wipe = max(wipes) if wipes else -1

    tracks = []
    for r, pts in arcs.items():
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        if len(pts) < 8:
            # A mark drawn once (e.g. a star painted at init) is still on screen
            # if nothing wipes the canvas afterwards. Keep it as a stationary
            # candidate; drop it if a full-canvas clear came later.
            if max(p[0] for p in pts) <= last_wipe or max(xs) - min(xs) > 2 or max(ys) - min(ys) > 2:
                continue
        spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        tracks.append({"r": r, "n": len(pts), "spread": spread,
                       "pts": [(p[1], p[2]) for p in pts],
                       "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys)})
    if not tracks:
        return {"error": "no filled arcs drawn"}

    moving = max(tracks, key=lambda t: t["spread"])
    stationary = [t for t in tracks if t["spread"] < max(2.0, moving["spread"] * 0.05)]
    # Among stationary marks pick the LARGEST -- a star is drawn bigger than a
    # focus tick. Picking the smallest selected the marker at the EMPTY focus,
    # and sweeping area about that point is not constant, which showed up as a
    # bogus Kepler-2 CV of 0.94 on a correct implementation.
    star = max(stationary, key=lambda t: t["r"]) if stationary else None

    cw = canvases[0][1] if canvases else 800
    ch = canvases[0][2] if canvases else 600
    out = {"canvas": [cw, ch], "planet_spread_px": moving["spread"],
           "planet_samples": len(moving["pts"]),
           "star_xy": [star["cx"], star["cy"]] if star else None}

    fit = fit_ellipse(moving["pts"])
    if not fit:
        out["fit"] = None
        return out
    cx, cy, a, b, th = fit
    ecc = math.sqrt(max(0.0, 1 - (b * b) / (a * a)))
    out["fit"] = {"cx": cx, "cy": cy, "a": a, "b": b, "theta": th, "e": ecc}

    if star:
        c = a * ecc
        f1 = (cx + c * math.cos(th), cy + c * math.sin(th))
        f2 = (cx - c * math.cos(th), cy - c * math.sin(th))
        d1 = math.hypot(star["cx"] - f1[0], star["cy"] - f1[1])
        d2 = math.hypot(star["cx"] - f2[0], star["cy"] - f2[1])
        out["star_focus_err_px"] = min(d1, d2)
        out["star_centre_err_px"] = math.hypot(star["cx"] - cx, star["cy"] - cy)

        # Kepler's 2nd law: equal areas in equal times. With a fixed timestep
        # the swept area between consecutive samples must be constant.
        areas = []
        pts = moving["pts"]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            ax_, ay_ = x1 - star["cx"], y1 - star["cy"]
            bx_, by_ = x2 - star["cx"], y2 - star["cy"]
            areas.append(abs(ax_ * by_ - ay_ * bx_) / 2.0)
        areas = [v for v in areas if v > 0]
        if len(areas) > 20:
            areas.sort()
            trimmed = areas[len(areas) // 20: -len(areas) // 20 or None]
            mean = sum(trimmed) / len(trimmed)
            var = sum((v - mean) ** 2 for v in trimmed) / len(trimmed)
            out["area_rate_cv"] = (math.sqrt(var) / mean) if mean else None
    return out
