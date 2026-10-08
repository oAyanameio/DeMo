"""Generate the DeMo M0 actor-only architecture figure.

Single source of truth for both artefacts:
  docs/figures/m0_actor_only_architecture.drawio
  docs/figures/m0_actor_only_architecture.png   (preview)

Usage:  python docs/tools/gen_m0_architecture.py
Canvas: 1700x780 layout units (1 unit == 1 px in drawio).
"""
import html
import math
from pathlib import Path
from xml.sax.saxutils import quoteattr

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "figures"
DRAWIO = OUT_DIR / "m0_actor_only_architecture.drawio"
PNG = OUT_DIR / "m0_actor_only_architecture.png"

W, H = 1700, 780
S = 2

FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

INK = "#2B3A45"
MUTE = "#7A8794"
MUTE_2 = "#98A4AE"
ARROW = "#5A6B7A"
BUS = "#8A97A3"

PANEL_L_F, PANEL_L_S = "#F5F7F9", "#DCE3E9"
PANEL_M_F, PANEL_M_S = "#F0F6FC", "#CFE0F1"
PANEL_R_F, PANEL_R_S = "#F6F2FB", "#E0D6EF"

MODE, MODE_TXT = "#E0A63C", "#A9761A"
STATE, STATE_TXT = "#4A86C8", "#2F6FB0"
COUPLE, COUPLE_TXT = "#7A5BB5", "#5B3F9E"

BOX_F, BOX_G, BOX_S = "#FFFFFF", "#E9F0F6", "#A9C3DE"
BOX_TXT, BOX_SUB = "#1F4E79", "#6B7C8C"

CUBE = {
    "pink": ("#F2BAC1", "#D98B94", "#C5838B"),
    "green": ("#BDDCC1", "#83B491", "#7CA487"),
}

nodes = []
edges = []


def n(kind, **kw):
    kw.setdefault("id", f"n{len(nodes)}")
    kw["kind"] = kind
    nodes.append(kw)
    return kw["id"]


def edge(pts, color=ARROW, width=1.6, arrow=True, dash=False):
    edges.append({"id": f"e{len(edges)}", "pts": pts, "color": color,
                  "width": width, "arrow": arrow, "dash": dash})


def text(x, y, w, h, value, size=12, bold=False, color=INK, align="c", valign="m"):
    return n("text", x=x, y=y, w=w, h=h, value=value, size=size, bold=bold,
             color=color, align=align, valign=valign)


def panel(x, y, w, h, fill, stroke):
    return n("rrect", x=x, y=y, w=w, h=h, radius=14, fill=fill, stroke=stroke, sw=1.3)


def box(x, y, w, h, fill, grad, stroke, accent=None, radius=10, sw=1.4):
    n("rrect", x=x, y=y, w=w, h=h, radius=radius, fill=fill, grad=grad,
      stroke=stroke, sw=sw)
    if accent:
        n("rrect", x=x + 4, y=y + 7, w=4, h=h - 14, radius=2,
          fill=accent, stroke=None, sw=0)


def group(x, y, w, h, label, color, label_color):
    return n("group", x=x, y=y, w=w, h=h, radius=12, stroke=color, sw=1.5,
             fill=None, label=label, label_color=label_color)


def cubes(fx, fy, count, size, gap, colors):
    for i in range(count):
        front, back, stroke = CUBE[colors[i % len(colors)]]
        cx = fx + i * (size + gap)
        n("rrect", x=cx + 5, y=fy - 5, w=size, h=size, radius=5,
          fill=back, stroke=stroke, sw=1.0)
        n("rrect", x=cx, y=fy, w=size, h=size, radius=5,
          fill=front, stroke=stroke, sw=1.2)


def dot(x, y, fill=BUS):
    return n("ellipse", x=x - 5, y=y - 5, w=10, h=10, fill=fill, stroke=fill, sw=1)


# ------------------------------------------------------------------ layout
panel(20, 90, 310, 600, PANEL_L_F, PANEL_L_S)
panel(360, 90, 770, 600, PANEL_M_F, PANEL_M_S)
panel(1160, 90, 520, 600, PANEL_R_F, PANEL_R_S)

for hx, ttl, sub, col, acc in (
    (20, "Scene Context Encoding", "Agent Encoding - Scene Context", "#4A5866", "#8FA6B8"),
    (360, "Decoupled Query Decoding", "State Consistency - Mode Localization", STATE_TXT, STATE),
    (1160, "Hybrid Coupling", "Query Coupling - Output Heads", COUPLE_TXT, COUPLE),
):
    text(hx + 20, 100, 300, 24, ttl, size=16, bold=True, color=col, align="l")
    text(hx + 20, 126, 320, 16, sub, size=10, color=MUTE, align="l")
    n("rrect", x=hx + 20, y=147, w=44, h=3, radius=1, fill=acc, stroke=None, sw=0)

# -- left panel
cubes(75, 170, 6, 30, 4, ["pink", "pink", "pink", "green", "green", "green"])
text(25, 212, 300, 20, "Scene Context Features", size=13, bold=True, color=INK)
text(25, 233, 300, 17, "(B, N, C)", size=10.5, color=MUTE)

box(70, 312, 210, 112, BOX_F, BOX_G, BOX_S, accent=STATE)
text(88, 326, 176, 18, "Agent Encoding", size=12.5, bold=True, color=BOX_TXT, align="l")
text(88, 345, 176, 15, "Linear + Uni-Mamba x4", size=10, color=BOX_SUB, align="l")
n("rect", x=88, y=366, w=176, h=1, fill="#C9D6E2", stroke=None, sw=0)
text(88, 374, 176, 18, "Scene Context Encoder", size=12.5, bold=True, color=BOX_TXT, align="l")
text(88, 393, 176, 15, "Pos / Angle + Self-Attn x5", size=10, color=BOX_SUB, align="l")

box(90, 520, 170, 96, "#FFFFFF", "#F3F6F9", "#B9C6D2", accent="#8FA6B8", sw=1.2)
text(90, 534, 170, 18, "Observed History", size=12.5, bold=True, color=INK)
text(90, 556, 170, 15, "(B, N, L, 4)", size=10.5, color=MUTE)
text(90, 576, 170, 14, "pos / vel / mask", size=9.5, color=MUTE_2)
n("rrect", x=110, y=598, w=58, h=6, radius=3, fill="#D3DBE2", stroke=None, sw=0)
n("rrect", x=174, y=598, w=36, h=6, radius=3, fill="#E3E9EE", stroke=None, sw=0)

edge([(175, 520), (175, 424)])
edge([(175, 312), (175, 200)])

# -- middle panel
edge([(275, 185), (370, 185), (370, 215)], color=BUS, width=2.0, arrow=False)
edge([(370, 215), (370, 460)], color=BUS, width=2.4, arrow=False)
dot(370, 215)
dot(370, 460)
edge([(374, 215), (456, 215)])
edge([(374, 460), (456, 460)])

for gy, gcolor, gtxt, gname, qlabel, qcolor, mlabel, blabel in (
    (168, MODE, MODE_TXT, "Mode Localization Module", "Mode Queries (K)", MODE_TXT,
     "Mode Cross-Attn \u00d73", "Mode Self-Attn \u00d73"),
    (413, STATE, STATE_TXT, "State Consistency Module", "State Queries (Ts)", STATE_TXT,
     "State Cross-Attn \u00d72", "State Bi-Mamba \u00d72"),
):
    group(385, gy, 520, 200, gname, gcolor, gtxt)
    cubes(456, gy + 34, 4, 26, 5, ["pink", "pink", "green", "green"])
    text(586, gy + 36, 130, 16, qlabel, size=10.5, color=qcolor, align="l")
    for mx, mtext in ((415, mlabel), (650, blabel)):
        n("rrect", x=mx, y=gy + 105, w=200, h=62, radius=9,
          fill="#FFFFFF", grad="#FBFDFF", stroke=gcolor, sw=1.4)
        text(mx, gy + 121, 200, 18, mtext, size=12, bold=True, color=gtxt)
    edge([(515, gy + 60), (515, gy + 105)])
    edge([(615, gy + 136), (650, gy + 136)])
    edge([(850, gy + 136), (925, gy + 136)])

edge([(930, 304), (930, 549)], color=BUS, width=2.4, arrow=False)
dot(930, 304)
dot(930, 549)
dot(930, 426, fill="#6B7C8C")
edge([(935, 426), (955, 426)])

n("rrect", x=955, y=366, w=120, h=120, radius=10,
  fill="#FFFFFF", stroke="#9AABB4", sw=1.3)
STOPS = [(0.00, (78, 155, 209)), (0.30, (143, 184, 201)),
         (0.55, (243, 217, 142)), (0.78, (239, 158, 99)), (1.00, (214, 91, 78))]


def heat(t):
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return "#%02X%02X%02X" % tuple(
                round(c0[i] + (c1[i] - c0[i]) * f) for i in range(3))
    return "#%02X%02X%02X" % STOPS[-1][1]


for r in range(5):
    for c in range(5):
        t = 0.10 + 0.80 * (c / 4) + (((r * 7 + c * 3) % 5) - 2) * 0.035
        n("rect", x=958 + c * 23, y=369 + r * 23, w=21, h=21, fill=heat(t),
          stroke="#FFFFFF", sw=1.0)

text(955, 492, 120, 19, "Coupled Tensor", size=12.5, bold=True, color=INK)
text(955, 512, 120, 15, "(B, K, Ts, C)", size=10.5, color=MUTE)

# -- right panel
group(1215, 260, 410, 332, "Hybrid Coupling Module", "#C3B0DE", COUPLE_TXT)
for i, label in enumerate(["Hybrid Cross-Attn \u00d73", "Joint Self-Attn \u00d73",
                           "Mode Interaction \u00d73", "Coupled Bi-Mamba \u00d72"]):
    y = 286 + i * 76
    n("rrect", x=1250, y=y, w=340, h=52, radius=9,
      fill="#FFFFFF", grad="#F4EEFB", stroke="#C3B0DE", sw=1.4)
    n("rrect", x=1254, y=y + 7, w=4, h=38, radius=2, fill=COUPLE, stroke=None, sw=0)
    text(1250, y + 17, 340, 18, label, size=12.5, bold=True, color="#4A3577")
    if i < 3:
        edge([(1420, y + 52), (1420, y + 76)])

edge([(1075, 426), (1145, 426), (1145, 312), (1250, 312)])
text(1215, 604, 410, 20,
     "Multi-modal future:  K trajectories (B, K, Tf, 2) + mode scores",
     size=10.5, color="#8C7BB0")


# ------------------------------------------------------------------ drawio
def _attr(v):
    return quoteattr(str(v))


def drawio_style(nd):
    k = nd["kind"]
    if k == "text":
        return ("html=1;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;"
                "fontSize=%s;fontColor=%s;fontStyle=%d;fontFamily=Helvetica;"
                "align=%s;verticalAlign=%s;spacing=0;"
                % (nd["size"], nd["color"], 1 if nd["bold"] else 0,
                   nd["align"], nd["valign"]))
    if k == "group":
        return ("rounded=1;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 5;"
                "fillColor=none;strokeColor=%s;strokeWidth=%s;"
                "verticalAlign=top;align=left;spacingTop=7;spacingLeft=12;"
                "fontSize=12.5;fontStyle=1;fontColor=%s;fontFamily=Helvetica;"
                % (nd["stroke"], nd["sw"], nd["label_color"]))
    if k == "ellipse":
        return ("ellipse;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;strokeWidth=%s;"
                % (nd["fill"], nd["stroke"], nd["sw"]))
    arc = 30 if nd.get("radius", 0) >= 12 else 18
    out = ["rounded=1;whiteSpace=wrap;html=1;", "arcSize=%d;" % arc,
           "strokeColor=%s;strokeWidth=%s;" % (nd["stroke"], nd["sw"])]
    if nd.get("grad"):
        out.append("fillColor=%s;gradientColor=%s;gradientDirection=south;"
                   % (nd["fill"], nd["grad"]))
    else:
        out.append("fillColor=%s;" % nd["fill"])
    return "".join(out)


def build_drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for nd in nodes:
        val = nd.get("value") or nd.get("label") or ""
        cells.append(
            '<mxCell id=%s value=%s style=%s vertex="1" parent="1">'
            '<mxGeometry x=%s y=%s width=%s height=%s as="geometry"/></mxCell>'
            % (_attr(nd["id"]), _attr(html.escape(val)), _attr(drawio_style(nd)),
               _attr(nd["x"]), _attr(nd["y"]), _attr(nd.get("w", 0)),
               _attr(nd.get("h", 0)))
        )
    for ed in edges:
        pts = ed["pts"]
        style = ("edgeStyle=none;rounded=1;html=1;jettySize=auto;"
                 "strokeColor=%s;strokeWidth=%s;endArrow=%s;endFill=1;%s"
                 % (ed["color"], ed["width"], "block" if ed["arrow"] else "none",
                    "dashed=1;" if ed["dash"] else ""))
        mid = ""
        if len(pts) > 2:
            mid = '<Array as="points">' + "".join(
                '<mxPoint x=%s y=%s/>' % (_attr(p[0]), _attr(p[1])) for p in pts[1:-1]
            ) + "</Array>"
        cells.append(
            '<mxCell id=%s style=%s edge="1" parent="1">'
            '<mxGeometry relative="1" as="geometry">'
            '<mxPoint x=%s y=%s as="sourcePoint"/>'
            '<mxPoint x=%s y=%s as="targetPoint"/>%s'
            "</mxGeometry></mxCell>"
            % (_attr(ed["id"]), _attr(style),
               _attr(pts[0][0]), _attr(pts[0][1]),
               _attr(pts[-1][0]), _attr(pts[-1][1]), mid)
        )
    body = "".join(cells)
    return (
        '<mxfile host="app.diagrams.net" agent="DeMo" version="24.7.17" type="device">'
        '<diagram id="m0-actor-only-architecture" name="M0 Actor-only Architecture">'
        '<mxGraphModel dx="%d" dy="%d" grid="0" gridSize="10" guides="1" tooltips="1" '
        'connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
        "<root>%s</root></mxGraphModel></diagram></mxfile>" % (W, H, W, H, body)
    )


# ------------------------------------------------------------------ PIL
_fonts = {}


def font(size, bold):
    key = (round(size * S), bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_B if bold else FONT_R, key[0])
    return _fonts[key]


def rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def sc(v):
    return v * S


def px(box):
    x, y, w, h = box
    return [sc(x), sc(y), sc(x + w) - 1, sc(y + h) - 1]


def dashed_round_rect(d, box, radius, color, width, dash=8, gap=6):
    x0, y0, x1, y1 = px(box)
    r = sc(radius)
    for (sx, sy), (ex, ey) in (
        ((x0 + r, y0), (x1 - r, y0)), ((x1, y0 + r), (x1, y1 - r)),
        ((x1 - r, y1), (x0 + r, y1)), ((x0, y1 - r), (x0, y0 + r)),
    ):
        length = math.hypot(ex - sx, ey - sy)
        if length <= 0:
            continue
        ux, uy = (ex - sx) / length, (ey - sy) / length
        pos, step = 0.0, sc(dash + gap)
        while pos < length:
            end = min(pos + sc(dash), length)
            d.line([sx + ux * pos, sy + uy * pos, sx + ux * end, sy + uy * end],
                   fill=color, width=width)
            pos += step
    dd = 2 * r
    d.arc([x0, y0, x0 + dd, y0 + dd], 180, 270, fill=color, width=width)
    d.arc([x1 - dd, y0, x1, y0 + dd], 270, 360, fill=color, width=width)
    d.arc([x1 - dd, y1 - dd, x1, y1], 0, 90, fill=color, width=width)
    d.arc([x0, y1 - dd, x0 + dd, y1], 90, 180, fill=color, width=width)


def fill_rrect(img, d, box, radius, fill, grad, stroke, sw):
    x0, y0, x1, y1 = px(box)
    r = sc(radius)
    bw, bh = int(x1 - x0 + 1), int(y1 - y0 + 1)
    if grad:
        top, bottom = rgb(fill), rgb(grad)
        strip = Image.new("RGB", (1, bh))
        for i in range(bh):
            t = i / max(1, bh - 1)
            strip.putpixel((0, i), tuple(
                round(top[c] + (bottom[c] - top[c]) * t) for c in range(3)))
        strip = strip.resize((bw, bh), Image.BILINEAR)
        mask = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=r, fill=255)
        img.paste(strip, (int(x0), int(y0)), mask)
    else:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=rgb(fill))
    if stroke:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=rgb(stroke),
                            width=max(1, round(sw * S)))


def draw_text(d, nd):
    f = font(nd["size"], nd["bold"])
    x, y, w, h = nd["x"], nd["y"], nd["w"], nd["h"]
    bb = d.textbbox((0, 0), nd["value"], font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    bx, by = sc(x), sc(y)
    if nd["align"] == "c":
        tx = bx + (sc(w) - tw) / 2 - bb[0]
    elif nd["align"] == "l":
        tx = bx - bb[0]
    else:
        tx = bx + sc(w) - tw - bb[0]
    if nd["valign"] == "m":
        ty = by + (sc(h) - th) / 2 - bb[1]
    elif nd["valign"] == "t":
        ty = by - bb[1]
    else:
        ty = by + sc(h) - th - bb[1]
    d.text((tx, ty), nd["value"], font=f, fill=rgb(nd["color"]))


def draw_edge(d, ed):
    pts = [(sc(x), sc(y)) for x, y in ed["pts"]]
    col = rgb(ed["color"])
    wd = max(1, round(ed["width"] * S))
    head = None
    if ed["arrow"]:
        tip, prev = pts[-1], pts[-2]
        ang = math.atan2(tip[1] - prev[1], tip[0] - prev[0])
        size = sc(8)
        pts = pts[:-1] + [(tip[0] - math.cos(ang) * size * 0.85,
                           tip[1] - math.sin(ang) * size * 0.85)]
        head = [tip,
                (tip[0] + size * math.cos(ang + math.radians(148)),
                 tip[1] + size * math.sin(ang + math.radians(148))),
                (tip[0] + size * math.cos(ang - math.radians(148)),
                 tip[1] + size * math.sin(ang - math.radians(148)))]
    if len(pts) > 1:
        d.line(pts, fill=col, width=wd, joint="curve")
        for p in pts[1:-1]:
            d.ellipse([p[0] - wd / 2, p[1] - wd / 2,
                       p[0] + wd / 2, p[1] + wd / 2], fill=col)
    if head:
        d.polygon(head, fill=col)


def build_png():
    img = Image.new("RGB", (sc(W), sc(H)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for nd in nodes:
        k = nd["kind"]
        if k == "rrect":
            fill_rrect(img, d, (nd["x"], nd["y"], nd["w"], nd["h"]), nd["radius"],
                       nd["fill"], nd.get("grad"), nd["stroke"], nd["sw"])
        elif k == "rect":
            x0, y0, x1, y1 = px((nd["x"], nd["y"], nd["w"], nd["h"]))
            d.rectangle([x0, y0, x1, y1], fill=rgb(nd["fill"]),
                        outline=rgb(nd["stroke"]) if nd["stroke"] else None,
                        width=max(1, round(nd["sw"] * S)) if nd["stroke"] else 0)
        elif k == "ellipse":
            d.ellipse(px((nd["x"], nd["y"], nd["w"], nd["h"])), fill=rgb(nd["fill"]),
                      outline=rgb(nd["stroke"]), width=max(1, round(nd["sw"] * S)))
        elif k == "group":
            dashed_round_rect(d, (nd["x"], nd["y"], nd["w"], nd["h"]), nd["radius"],
                              rgb(nd["stroke"]), max(1, round(nd["sw"] * S)))
            d.text((sc(nd["x"] + 12), sc(nd["y"] + 8)), nd["label"],
                   font=font(12.5, True), fill=rgb(nd["label_color"]))
        elif k == "text":
            draw_text(d, nd)
    for ed in edges:
        draw_edge(d, ed)
    img.resize((W, H), Image.LANCZOS).save(PNG)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAWIO.write_text(build_drawio(), encoding="utf-8")
    build_png()
    print("wrote", DRAWIO)
    print("wrote", PNG)
