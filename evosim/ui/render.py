"""Scene rendering with moderngl: instanced circles with a heading tick, bond lines, field overlays,
region outlines, selection rings. The only module (besides ui/) that is allowed to be fully 2D-specific.

Initial approach per the spec: each rendered frame copies pos/mass/angle/... GPU -> host (a few MB at
100k cells), builds per-instance colours with numpy, uploads to a VBO. (CUDA-OpenGL interop is the
documented later optimisation.)
"""
from __future__ import annotations

import colorsys
import math

import numpy as np

from ..genome import isa

COLOR_MODES = ("lineage", "genome hash", "organism id", "energy fraction", "dominant trait", "specific trait",
               "state register", "age", "division depth", "origin")

_TRAIT_COLORS = np.array([[0.2, 0.85, 0.25], [0.95, 0.75, 0.15], [0.25, 0.55, 1.0], [0.95, 0.2, 0.2],
                          [0.6, 0.6, 0.65], [0.55, 0.25, 0.7], [0.95, 0.55, 0.85], [0.95, 0.5, 0.15],
                          [0.3, 0.9, 0.9]], np.float32)
_STATE_COLORS = np.array([[0.8, 0.8, 0.8], [0.25, 0.6, 1.0], [1.0, 0.45, 0.2], [0.35, 0.9, 0.4], [0.9, 0.3, 0.7],
                          [0.95, 0.9, 0.25], [0.5, 0.35, 0.9], [0.2, 0.85, 0.8]], np.float32)
_ORIGIN_COLORS = np.array([[0.7, 0.7, 0.7], [0.3, 0.8, 1.0], [1.0, 0.6, 0.2], [0.8, 0.4, 1.0]], np.float32)


def _hash_colors(vals):
    h = (vals.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    hue = (h % np.uint64(360)).astype(np.float32) / 360.0
    sat = 0.55 + 0.4 * ((h >> np.uint64(9)) % np.uint64(100)).astype(np.float32) / 100.0
    val = 0.7 + 0.3 * ((h >> np.uint64(17)) % np.uint64(100)).astype(np.float32) / 100.0
    i = np.floor(hue * 6).astype(int) % 6
    f = hue * 6 - np.floor(hue * 6)
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    r = np.choose(i, [val, q, p, p, t, val])
    g = np.choose(i, [t, val, val, q, p, p])
    b = np.choose(i, [p, p, t, val, val, q])
    return np.stack([r, g, b], 1).astype(np.float32)


def viridis(x):
    x = np.clip(x, 0, 1).astype(np.float32)[:, None]
    c0 = np.array([0.267, 0.005, 0.329], np.float32)
    c1 = np.array([0.128, 0.567, 0.551], np.float32)
    c2 = np.array([0.993, 0.906, 0.144], np.float32)
    lo = c0 + (c1 - c0) * np.clip(x * 2, 0, 1)
    hi = c1 + (c2 - c1) * np.clip(x * 2 - 1, 0, 1)
    return np.where(x < 0.5, lo, hi)


def cell_colors(mode, trait_index, a):
    """a: dict of host arrays for alive cells. Returns (n,3) float32."""
    n = len(a["mass"])
    if mode == "lineage":
        return _hash_colors(a["lineage"])
    if mode == "genome hash":
        return _hash_colors(a["ghash"])
    if mode == "organism id":
        return _hash_colors(a["organism"])
    if mode == "energy fraction":
        return viridis(a["energy"] / np.maximum(a["max_energy"], 1e-6))
    if mode == "dominant trait":
        dom = np.argmax(a["traits"], axis=1)
        col = _TRAIT_COLORS[dom]
        weak = a["traits"].max(axis=1) < 0.05
        col[weak] = 0.35
        return col
    if mode == "specific trait":
        return viridis(a["traits"][:, trait_index])
    if mode == "state register":
        return _STATE_COLORS[a["state"] % len(_STATE_COLORS)]
    if mode == "age":
        return viridis(np.log1p(a["age"]) / 11.0)
    if mode == "division depth":
        return viridis(a["depth"] / max(1.0, float(a["depth"].max())))
    if mode == "origin":
        return _ORIGIN_COLORS[a["origin"] % 4]
    return np.ones((n, 3), np.float32)


CELL_VS = """
#version 330
in vec2 corner;
in vec2 ipos; in float irad; in float iang; in vec4 icol;
uniform vec2 cam; uniform vec2 view; uniform float zoom; uniform vec2 tile; uniform float rscale;
out vec2 uv; out vec4 col; out float ang;
void main(){
  vec2 w = ipos + tile + corner * irad * rscale;
  vec2 p = (w - cam) * zoom / (view * 0.5);
  gl_Position = vec4(p, 0.0, 1.0);
  uv = corner; col = icol; ang = iang;
}
"""
CELL_FS = """
#version 330
in vec2 uv; in vec4 col; in float ang;
uniform float ring;
out vec4 frag;
void main(){
  float d = length(uv);
  if (ring > 0.5) { if (d > 1.0 || d < 0.82) discard; frag = vec4(col.rgb, 1.0); return; }
  if (d > 1.0) discard;
  vec2 dir = vec2(cos(ang), sin(ang));
  float along = dot(uv, dir);
  float across = abs(uv.x * dir.y - uv.y * dir.x);
  vec3 c = col.rgb * (0.78 + 0.22 * (1.0 - d));
  if (along > 0.0 && across < 0.13) c = c * 0.35;
  if (d > 0.9) c = c * 0.6;
  frag = vec4(c, col.a);
}
"""
LINE_VS = """
#version 330
in vec2 p; in vec4 c;
uniform vec2 cam; uniform vec2 view; uniform float zoom; uniform vec2 tile;
out vec4 col;
void main(){ gl_Position = vec4((p + tile - cam) * zoom / (view * 0.5), 0.0, 1.0); col = c; }
"""
LINE_FS = """
#version 330
in vec4 col; out vec4 frag;
void main(){ frag = col; }
"""
TEX_VS = """
#version 330
in vec2 corner;
uniform vec2 cam; uniform vec2 view; uniform float zoom; uniform vec2 tile; uniform vec2 world;
out vec2 uv;
void main(){ vec2 w = corner * world + tile; gl_Position = vec4((w - cam) * zoom / (view * 0.5), 0.0, 1.0); uv = corner; }
"""
TEX_FS = """
#version 330
in vec2 uv; uniform sampler2D tex; uniform float vmin; uniform float vmax; uniform float alpha; uniform vec3 tint;
out vec4 frag;
void main(){
  float v = texture(tex, uv).r;
  float t = clamp((v - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0);
  vec3 c = mix(vec3(0.02), tint, t);
  frag = vec4(c, alpha * (0.15 + 0.85 * t));
}
"""


class SceneRenderer:
    def __init__(self, ctx):
        import moderngl
        self.mgl = moderngl
        self.ctx = ctx
        self.cell_prog = ctx.program(vertex_shader=CELL_VS, fragment_shader=CELL_FS)
        self.line_prog = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.tex_prog = ctx.program(vertex_shader=TEX_VS, fragment_shader=TEX_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4")
        self.quad = ctx.buffer(quad)
        uvq = np.array([0, 0, 1, 0, 0, 1, 1, 1], "f4")
        self.uvquad = ctx.buffer(uvq)
        self.max_inst = 0
        self.inst = None
        self.cell_vao = None
        self.n_inst = 0
        self.line_buf = ctx.buffer(reserve=4 * 6 * 4096)
        self.line_vao = ctx.vertex_array(self.line_prog, [(self.line_buf, "2f 4f", "p", "c")])
        self.n_line_verts = 0
        self.sel_buf = ctx.buffer(reserve=4 * 8 * 256)
        self.sel_vao = None
        self.n_sel = 0
        self.tex = None
        self.tex_shape = None
        self.tex_vao = ctx.vertex_array(self.tex_prog, [(self.uvquad, "2f", "corner")])
        self.overlay_range = (0.0, 1.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.fbo_size = (1, 1)

    # ------------------------------------------------------------------ data
    def set_cells(self, pos, rad, ang, col):
        n = len(pos)
        data = np.empty((n, 8), "f4")
        data[:, 0:2] = pos
        data[:, 2] = rad
        data[:, 3] = ang
        data[:, 4:7] = col
        data[:, 7] = 1.0
        if n > self.max_inst or self.inst is None:
            self.max_inst = max(1024, int(n * 1.5))
            self.inst = self.ctx.buffer(reserve=self.max_inst * 32)
            self.cell_vao = self.ctx.vertex_array(
                self.cell_prog, [(self.quad, "2f", "corner"), (self.inst, "2f 1f 1f 4f/i", "ipos", "irad", "iang", "icol")])
        if n:
            self.inst.write(data.tobytes())
        self.n_inst = n

    def set_lines(self, verts):
        """verts: (m, 6) float32 [x y r g b a], pairs form segments."""
        m = len(verts)
        if m * 24 > self.line_buf.size:
            self.line_buf = self.ctx.buffer(reserve=int(m * 24 * 1.5))
            self.line_vao = self.ctx.vertex_array(self.line_prog, [(self.line_buf, "2f 4f", "p", "c")])
        if m:
            self.line_buf.write(np.ascontiguousarray(verts, "f4").tobytes())
        self.n_line_verts = m

    def set_overlay(self, arr, vmin=None, vmax=None):
        """arr: 2D float array indexed [x, y]."""
        a = np.ascontiguousarray(arr.T.astype("f4"))
        h, w = a.shape
        if self.tex is None or self.tex_shape != (w, h):
            self.tex = self.ctx.texture((w, h), 1, dtype="f4")
            self.tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
            self.tex.repeat_x = self.tex.repeat_y = True
            self.tex_shape = (w, h)
        self.tex.write(a.tobytes())
        self.overlay_range = (float(a.min() if vmin is None else vmin), float(a.max() if vmax is None else vmax))

    def clear_overlay(self):
        self.tex = None

    # ------------------------------------------------------------------ draw
    def draw(self, cam, view, zoom, world, tiled, bg=(0.04, 0.045, 0.06), overlay_alpha=0.6, overlay_tint=(0.2, 0.9, 0.3),
             show_world_border=True, radius_scale=1.0):
        ctx = self.ctx
        ctx.viewport = (0, 0, int(view[0]), int(view[1]))
        ctx.clear(*bg)
        tiles = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)] if tiled else [(0, 0)]
        for p in (self.cell_prog, self.line_prog, self.tex_prog):
            p["cam"].value = tuple(map(float, cam))
            p["view"].value = (float(view[0]), float(view[1]))
            p["zoom"].value = float(zoom)
        self.tex_prog["world"].value = tuple(map(float, world))
        for (tx, ty) in tiles:
            off = (tx * world[0], ty * world[1])
            if self.tex is not None:
                self.tex.use(0)
                self.tex_prog["tex"].value = 0
                self.tex_prog["vmin"].value, self.tex_prog["vmax"].value = self.overlay_range
                self.tex_prog["alpha"].value = overlay_alpha
                self.tex_prog["tint"].value = tuple(map(float, overlay_tint))
                self.tex_prog["tile"].value = off
                self.tex_vao.render(self.mgl.TRIANGLE_STRIP)
            self.cell_prog["tile"].value = off
            self.cell_prog["ring"].value = 0.0
            self.cell_prog["rscale"].value = float(radius_scale)
            if self.n_inst:
                self.cell_vao.render(self.mgl.TRIANGLE_STRIP, instances=self.n_inst)
            self.line_prog["tile"].value = off
            if self.n_line_verts:
                self.line_vao.render(self.mgl.LINES, vertices=self.n_line_verts)
        if self.n_sel and self.sel_vao is not None:
            self.cell_prog["tile"].value = (0.0, 0.0)
            self.cell_prog["ring"].value = 1.0
            self.cell_prog["rscale"].value = 1.35
            self.sel_vao.render(self.mgl.TRIANGLE_STRIP, instances=self.n_sel)
            self.cell_prog["ring"].value = 0.0

    def set_selection(self, pos, rad):
        n = len(pos)
        self.n_sel = min(n, 256)
        if not n:
            return
        data = np.zeros((self.n_sel, 8), "f4")
        data[:, 0:2] = pos[:self.n_sel]
        data[:, 2] = rad[:self.n_sel]
        data[:, 4:8] = (1.0, 1.0, 0.2, 1.0)
        self.sel_buf.write(data.tobytes())
        if self.sel_vao is None:
            self.sel_vao = self.ctx.vertex_array(
                self.cell_prog, [(self.quad, "2f", "corner"), (self.sel_buf, "2f 1f 1f 4f/i", "ipos", "irad", "iang", "icol")])

    def read_pixels(self, w, h):
        data = self.ctx.screen.read(viewport=(0, 0, w, h), components=3)
        self.ctx.error          # reading the default framebuffer can leave a sticky GL error on some drivers; drain it
        return np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1].copy()


def bond_lines(pos_a, pos_b, strain, world, break_strain, color_by_strain=True):
    """Bond segments with minimum-image wrap: draw from a to a + d (tiling shows the wrapped half)."""
    d = pos_b - pos_a
    d -= np.array(world, np.float32) * np.round(d / np.array(world, np.float32))
    n = len(pos_a)
    out = np.zeros((n * 2, 6), np.float32)
    s = np.clip(np.abs(strain) / max(break_strain, 1e-6), 0, 1)
    col = np.stack([0.4 + 0.6 * s, 0.85 - 0.6 * s, 0.9 - 0.8 * s, np.full(n, 0.9)], 1)
    out[0::2, 0:2] = pos_a
    out[1::2, 0:2] = pos_a + d
    out[0::2, 2:6] = col
    out[1::2, 2:6] = col
    return out


def rect_lines(x, y, w, h, color):
    pts = [(x, y), (x + w, y), (x + w, y), (x + w, y + h), (x + w, y + h), (x, y + h), (x, y + h), (x, y)]
    return np.array([[px, py, *color] for px, py in pts], np.float32)


def circle_lines(cx, cy, r, color, seg=48):
    t = np.linspace(0, 2 * math.pi, seg + 1)
    pts = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)
    out = []
    for i in range(seg):
        out.append([*pts[i], *color])
        out.append([*pts[i + 1], *color])
    return np.array(out, np.float32)
