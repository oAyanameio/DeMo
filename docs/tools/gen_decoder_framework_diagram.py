"""Generate an editable draw.io decoder diagram and matching PNG preview."""

from math import atan2, cos, sin
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures"
DRAWIO = FIGURES / "demo_decoder_framework.drawio"
PNG = FIGURES / "demo_decoder_framework.png"
WIDTH, HEIGHT, SCALE = 2220, 1000, 2

INK = "#263746"
MUTED = "#657786"
ARROW = "#536878"
STATE_FILL, STATE_STROKE = "#EAF2FA", "#82A9D0"
MODE_FILL, MODE_STROKE = "#FBF3E4", "#D7B66F"
HYBRID_FILL, HYBRID_STROKE = "#F1ECF8", "#A895C6"
INPUT_FILL, INPUT_STROKE = "#EAF2F8", "#9DB8CC"
OUTPUT_FILL, OUTPUT_STROKE = "#EAF4EF", "#8BB39C"
WHITE = "#FFFFFF"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

panels = [
    ("scene_panel", "SCENE CONTEXT", (30, 145, 285, 735), INPUT_FILL, INPUT_STROKE),
    ("query_panel", "DECOUPLED QUERY DECODER", (345, 145, 915, 735), "#F7F9FB", "#D6DEE6"),
    ("hybrid_panel", "HYBRID COUPLING", (1290, 145, 470, 735), HYBRID_FILL, HYBRID_STROKE),
    ("output_panel", "PREDICTIONS", (1790, 145, 400, 735), OUTPUT_FILL, OUTPUT_STROKE),
]

lanes = [
    ("state_lane", "STATE CONSISTENCY", (365, 215, 875, 285), "#F4F8FC", STATE_STROKE),
    ("mode_lane", "MODE LOCALIZATION", (365, 530, 875, 305), "#FFFCF5", MODE_STROKE),
]

nodes = [
    ("scene_memory", "Scene encoder output\nContext tokens · actor-validity mask",
     (60, 445, 225, 105), INPUT_FILL, INPUT_STROKE, 12, True),
    ("state_query", "Future-time query\nembedding (MLP)",
     (390, 300, 170, 78), STATE_FILL, STATE_STROKE, 12, True),
    ("state_cross", "State cross-attention\nScene-context retrieval",
     (610, 300, 185, 78), STATE_FILL, STATE_STROKE, 12, True),
    ("state_temporal", "Temporal Bi-Mamba\nRMSNorm",
     (835, 300, 185, 78), STATE_FILL, STATE_STROKE, 13, True),
    ("state_features", "State features",
     (1060, 300, 170, 78), STATE_FILL, STATE_STROKE, 13, True),
    ("state_aux", "State forecast head\nAuxiliary state trajectory",
     (1035, 410, 220, 68), WHITE, STATE_STROKE, 11, False),
    ("mode_query", "Focal actor token\n+ learned mode queries",
     (390, 645, 170, 78), MODE_FILL, MODE_STROKE, 12, True),
    ("mode_cross", "Mode cross-attention\nScene-context retrieval",
     (610, 645, 185, 78), MODE_FILL, MODE_STROKE, 12, True),
    ("mode_self", "Mode self-attention",
     (835, 645, 185, 78), MODE_FILL, MODE_STROKE, 13, True),
    ("mode_features", "Mode features",
     (1060, 645, 170, 78), MODE_FILL, MODE_STROKE, 13, True),
    ("mode_aux", "Mode trajectory head\nAuxiliary trajectories · scores · scales",
     (1020, 750, 250, 68), WHITE, MODE_STROKE, 11, False),
    ("hybrid_fusion", "Hybrid query coupling\nCombine state + mode features",
     (1325, 240, 400, 78), HYBRID_FILL, HYBRID_STROKE, 13, True),
    ("hybrid_cross", "Hybrid cross-attention\nScene-context refinement",
     (1325, 355, 400, 72), WHITE, HYBRID_STROKE, 12, True),
    ("joint_attention", "Joint self-attention",
     (1325, 455, 400, 68), WHITE, HYBRID_STROKE, 13, True),
    ("mode_interaction", "Mode-interaction self-attention",
     (1325, 550, 400, 68), WHITE, HYBRID_STROKE, 12, True),
    ("coupled_temporal", "Coupled temporal Bi-Mamba\nRMSNorm",
     (1325, 645, 400, 72), WHITE, HYBRID_STROKE, 12, True),
    ("coupled_head", "Coupled trajectory head\nMeans · mode scores · uncertainty scales",
     (1325, 750, 400, 78), HYBRID_FILL, HYBRID_STROKE, 12, True),
    ("final_output", "Final multimodal forecast\nTrajectories · mode probabilities · uncertainty",
     (1840, 430, 300, 118), OUTPUT_FILL, OUTPUT_STROKE, 13, True),
]

# Edge tuple: id, source, target, anchors, bends, color, dashed.
edges = [
    ("e_time_state", "state_query", "state_cross", "1,0.5", "0,0.5", [], STATE_STROKE, False),
    ("e_state_cross_temporal", "state_cross", "state_temporal", "1,0.5", "0,0.5", [], STATE_STROKE, False),
    ("e_state_temporal_features", "state_temporal", "state_features", "1,0.5", "0,0.5", [], STATE_STROKE, False),
    ("e_state_aux", "state_features", "state_aux", "0.5,1", "0.5,0", [], STATE_STROKE, True),
    ("e_mode_query_cross", "mode_query", "mode_cross", "1,0.5", "0,0.5", [], MODE_STROKE, False),
    ("e_mode_cross_self", "mode_cross", "mode_self", "1,0.5", "0,0.5", [], MODE_STROKE, False),
    ("e_mode_self_features", "mode_self", "mode_features", "1,0.5", "0,0.5", [], MODE_STROKE, False),
    ("e_mode_aux", "mode_features", "mode_aux", "0.5,1", "0.5,0", [], MODE_STROKE, True),
    ("e_state_hybrid", "state_features", "hybrid_fusion", "1,0.5", "0,0.25", [(1270, 339), (1270, 260)], STATE_STROKE, False),
    ("e_mode_hybrid", "mode_features", "hybrid_fusion", "1,0.5", "0,0.8", [(1278, 684), (1278, 302)], MODE_STROKE, False),
    ("e_hybrid_cross", "hybrid_fusion", "hybrid_cross", "0.5,1", "0.5,0", [], HYBRID_STROKE, False),
    ("e_hybrid_joint", "hybrid_cross", "joint_attention", "0.5,1", "0.5,0", [], HYBRID_STROKE, False),
    ("e_joint_mode", "joint_attention", "mode_interaction", "0.5,1", "0.5,0", [], HYBRID_STROKE, False),
    ("e_mode_temporal", "mode_interaction", "coupled_temporal", "0.5,1", "0.5,0", [], HYBRID_STROKE, False),
    ("e_temporal_head", "coupled_temporal", "coupled_head", "0.5,1", "0.5,0", [], HYBRID_STROKE, False),
    ("e_head_output", "coupled_head", "final_output", "1,0.5", "0,0.5", [(1775, 789), (1775, 489)], OUTPUT_STROKE, False),
    ("e_scene_state", "scene_memory", "state_cross", "1,0", "0.5,0", [(325, 445), (325, 225), (702, 225)], INPUT_STROKE, False),
    ("e_scene_mode_query", "scene_memory", "mode_query", "1,0.5", "0,0.5", [(325, 497), (325, 684)], INPUT_STROKE, False),
    ("e_scene_mode", "scene_memory", "mode_cross", "1,1", "0.5,1", [(325, 550), (325, 850), (702, 850)], INPUT_STROKE, False),
    ("e_scene_hybrid", "scene_memory", "hybrid_cross", "1,0.25", "0.5,0", [(325, 471), (325, 195), (1525, 195)], INPUT_STROKE, False),
]


def card_style(fill, stroke, size=13, bold=True):
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.3;"
        f"fontColor={INK};fontFamily=Arial;fontSize={size};"
        f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;spacing=7;"
    )


def text_style(color=MUTED, size=11, align="left", bold=False):
    return (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
        f"fontColor={color};fontFamily=Arial;fontSize={size};"
        f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;spacing=0;"
    )


def add_vertex(parent, cell_id, label, box, style):
    x, y, width, height = box
    cell = ET.SubElement(parent, "mxCell", {
        "id": cell_id, "value": label.replace("\n", "<br>"), "style": style,
        "vertex": "1", "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(width), "height": str(height),
        "as": "geometry",
    })


def build_drawio():
    root = ET.Element("mxfile", {
        "host": "app.diagrams.net", "agent": "DeMo architecture figure generator",
        "version": "24.7.17", "type": "device",
    })
    diagram = ET.SubElement(root, "diagram", {
        "id": "demo-decoder-framework", "name": "Decoder Framework",
    })
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(WIDTH), "dy": str(HEIGHT), "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(WIDTH), "pageHeight": str(HEIGHT),
        "math": "0", "shadow": "0",
    })
    graph = ET.SubElement(model, "root")
    ET.SubElement(graph, "mxCell", {"id": "0"})
    ET.SubElement(graph, "mxCell", {"id": "1", "parent": "0"})

    for cell_id, label, box, fill, stroke in panels:
        add_vertex(graph, cell_id, label, box, card_style(fill, stroke, 15))
    for cell_id, label, box, fill, stroke in lanes:
        style = (
            "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1.2;dashed=1;dashPattern=6 4;"
            f"fontColor={stroke};fontFamily=Arial;fontSize=12;fontStyle=1;"
            "align=left;verticalAlign=top;spacingTop=10;spacingLeft=14;"
        )
        add_vertex(graph, cell_id, label, box, style)

    for cell_id, source, target, source_anchor, target_anchor, bends, color, dashed in edges:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={color};strokeWidth=1.6;endArrow=block;endFill=1;"
            f"exitX={source_anchor.split(',')[0]};exitY={source_anchor.split(',')[1]};"
            f"entryX={target_anchor.split(',')[0]};entryY={target_anchor.split(',')[1]};"
        )
        if dashed:
            style += "dashed=1;dashPattern=6 4;"
        cell = ET.SubElement(graph, "mxCell", {
            "id": cell_id, "style": style, "edge": "1", "parent": "1",
            "source": source, "target": target,
        })
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if bends:
            points = ET.SubElement(geometry, "Array", {"as": "points"})
            for x, y in bends:
                ET.SubElement(points, "mxPoint", {"x": str(x), "y": str(y)})

    for cell_id, label, box, fill, stroke, size, bold in nodes:
        add_vertex(graph, cell_id, label, box, card_style(fill, stroke, size, bold))

    add_vertex(graph, "title", "DeMo Decoder Framework", (32, 24, 1450, 42),
               text_style(INK, 24, bold=True))
    add_vertex(graph, "subtitle", "Decoupled state and mode queries are coupled to produce the final multimodal forecast",
               (34, 72, 1800, 28), text_style(MUTED, 13))
    add_vertex(graph, "legend", "Blue: state consistency   ·   Amber: mode localization   ·   Purple: hybrid coupling   ·   Dashed arrows: auxiliary forecasts",
               (34, 915, 1750, 25), text_style(MUTED, 11))

    DRAWIO.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(DRAWIO, encoding="utf-8", xml_declaration=True)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, round(size * SCALE))


def rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def wrap_lines(label, font_obj, width):
    lines = []
    for paragraph in label.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textlength(candidate, font=font_obj) <= width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_label(label, box, size, bold, color=INK, align="center"):
    x, y, width, height = box
    font_obj = font(size, bold)
    lines = wrap_lines(label, font_obj, width * SCALE - 28 * SCALE)
    line_gap = 4 * SCALE
    heights = [draw.textbbox((0, 0), line or "Ag", font=font_obj)[3] -
               draw.textbbox((0, 0), line or "Ag", font=font_obj)[1] for line in lines]
    cursor_y = y * SCALE + (height * SCALE - sum(heights) - line_gap * (len(lines) - 1)) / 2
    for line, line_height in zip(lines, heights):
        line_width = draw.textlength(line, font=font_obj)
        if align == "left":
            cursor_x = x * SCALE + 24 * SCALE
        elif align == "right":
            cursor_x = (x + width) * SCALE - 24 * SCALE - line_width
        else:
            cursor_x = x * SCALE + (width * SCALE - line_width) / 2
        draw.text((round(cursor_x), round(cursor_y)), line, font=font_obj, fill=rgb(color))
        cursor_y += line_height + line_gap


def anchor(node_id, fraction):
    node = next(item for item in nodes if item[0] == node_id)
    x, y, width, height = node[2]
    ax, ay = (float(value) for value in fraction.split(","))
    return x + ax * width, y + ay * height


def route_points(edge):
    _, source, target, source_anchor, target_anchor, bends, _, _ = edge
    start = anchor(source, source_anchor)
    end = anchor(target, target_anchor)
    if bends:
        return [start, *bends, end]
    sx = float(source_anchor.split(",")[0])
    tx = float(target_anchor.split(",")[0])
    if sx in (0, 1) and tx in (0, 1):
        middle = (start[0] + end[0]) / 2
        return [start, (middle, start[1]), (middle, end[1]), end]
    if sx not in (0, 1) and tx not in (0, 1):
        middle = (start[1] + end[1]) / 2
        return [start, (start[0], middle), (end[0], middle), end]
    if sx in (0, 1):
        return [start, (end[0], start[1]), end]
    return [start, (start[0], end[1]), end]


def draw_arrow(points, color, dashed=False):
    scaled = [(x * SCALE, y * SCALE) for x, y in points]
    stroke = rgb(color)
    for start, end in zip(scaled, scaled[1:]):
        x0, y0 = start
        x1, y1 = end
        if not dashed:
            draw.line((x0, y0, x1, y1), fill=stroke, width=3)
            continue
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if not distance:
            continue
        ux, uy = (x1 - x0) / distance, (y1 - y0) / distance
        offset = 0
        while offset < distance:
            finish = min(offset + 18, distance)
            draw.line((x0 + ux * offset, y0 + uy * offset,
                       x0 + ux * finish, y0 + uy * finish), fill=stroke, width=3)
            offset = finish + 12
    tip, previous = scaled[-1], scaled[-2]
    angle = atan2(tip[1] - previous[1], tip[0] - previous[0])
    length, half_width = 18, 8
    base = (tip[0] - cos(angle) * length, tip[1] - sin(angle) * length)
    left = (base[0] + cos(angle + 1.5708) * half_width,
            base[1] + sin(angle + 1.5708) * half_width)
    right = (base[0] + cos(angle - 1.5708) * half_width,
             base[1] + sin(angle - 1.5708) * half_width)
    draw.polygon([tip, left, right], fill=stroke)


def build_png():
    global draw
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), WHITE)
    draw = ImageDraw.Draw(image)

    for _, label, box, fill, stroke in panels:
        x, y, width, height = box
        rect = (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE)
        draw.rounded_rectangle(rect, radius=24, fill=rgb(fill), outline=rgb(stroke), width=3)
        draw_label(label, (x + 14, y + 6, width - 28, 40), 15, True, INK, "left")
    for _, label, box, fill, stroke in lanes:
        x, y, width, height = box
        rect = (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE)
        draw.rounded_rectangle(rect, radius=20, fill=rgb(fill), outline=rgb(stroke), width=2)
        draw_label(label, (x + 12, y + 7, width - 24, 28), 12, True, stroke, "left")

    for edge in edges:
        draw_arrow(route_points(edge), edge[6], edge[7])

    for _, label, box, fill, stroke, size, bold in nodes:
        x, y, width, height = box
        rect = (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE)
        draw.rounded_rectangle(rect, radius=18, fill=rgb(fill), outline=rgb(stroke), width=3)
        draw_label(label, box, size, bold)

    draw_label("DeMo Decoder Framework", (32, 24, 1450, 42), 24, True, INK, "left")
    draw_label("Decoupled state and mode queries are coupled to produce the final multimodal forecast",
               (34, 72, 1800, 28), 13, False, MUTED, "left")
    draw_label("Blue: state consistency   ·   Amber: mode localization   ·   Purple: hybrid coupling   ·   Dashed arrows: auxiliary forecasts",
               (34, 915, 1750, 25), 11, False, MUTED, "left")

    PNG.parent.mkdir(parents=True, exist_ok=True)
    image.save(PNG, format="PNG", dpi=(300, 300), optimize=True)


if __name__ == "__main__":
    build_drawio()
    build_png()
    print(DRAWIO)
    print(PNG)
