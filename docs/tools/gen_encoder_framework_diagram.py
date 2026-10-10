"""Generate an editable encoder-framework diagram and its PNG preview."""

from math import atan2, cos, sin
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures"
DRAWIO = FIGURES / "demo_encoder_framework.drawio"
PNG = FIGURES / "demo_encoder_framework.png"
WIDTH, HEIGHT, SCALE = 1820, 900, 2

INK = "#263746"
MUTED = "#657786"
ARROW = "#536878"
INPUT_FILL, INPUT_STROKE = "#EAF2F8", "#9DB8CC"
ACTOR_FILL, ACTOR_STROKE = "#EEF4FA", "#91ACCA"
SCENE_FILL, SCENE_STROKE = "#F1EDF8", "#AD9BC8"
WHITE = "#FFFFFF"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

panels = [
    ("input_panel", "ACTOR INPUTS", (30, 140, 360, 650), INPUT_FILL, INPUT_STROKE),
    ("actor_panel", "SHARED ACTOR-WISE ENCODER", (420, 140, 850, 650), ACTOR_FILL, ACTOR_STROKE),
    ("scene_panel", "SCENE CONTEXT ENCODER", (1300, 140, 490, 650), SCENE_FILL, SCENE_STROKE),
]

nodes = [
    ("histories", "Per-actor histories + validity masks\nFocal actor + neighboring actors",
     (70, 250, 290, 115), INPUT_FILL, INPUT_STROKE, 13, True),
    ("metadata", "Actor metadata\nCenter position · heading · actor type",
     (70, 545, 290, 100), WHITE, INPUT_STROKE, 12, False),
    ("motion_features", "Motion feature construction\nRelative displacement · velocity change",
     (460, 250, 210, 90), WHITE, ACTOR_STROKE, 12, True),
    ("history_projection", "History feature\nprojection (MLP)",
     (720, 250, 190, 90), WHITE, ACTOR_STROKE, 13, True),
    ("temporal_mamba", "Temporal sequence encoder\nMamba",
     (960, 250, 260, 90), ACTOR_FILL, ACTOR_STROKE, 14, True),
    ("last_state", "Last valid state selection",
     (960, 385, 260, 72), WHITE, ACTOR_STROKE, 13, True),
    ("type_embedding", "Actor-type embedding",
     (460, 520, 210, 68), WHITE, ACTOR_STROKE, 12, True),
    ("position_embedding", "Position & heading\nprojection (MLP)",
     (460, 615, 210, 68), WHITE, ACTOR_STROKE, 12, True),
    ("actor_fusion", "Actor representation fusion\nAddition of temporal and context features",
     (790, 545, 300, 92), ACTOR_FILL, ACTOR_STROKE, 12, True),
    ("actor_mask", "Actor validity mask",
     (1370, 245, 350, 58), WHITE, SCENE_STROKE, 12, False),
    ("scene_transformer", "Scene interaction\nTransformer self-attention",
     (1370, 375, 350, 110), SCENE_FILL, SCENE_STROKE, 14, True),
    ("scene_norm", "Layer normalization",
     (1370, 535, 350, 62), WHITE, SCENE_STROKE, 13, True),
    ("scene_features", "Contextualized actor representations",
     (1370, 650, 350, 82), WHITE, SCENE_STROKE, 13, True),
]

# Each edge stores its source/target anchor fractions and optional orthogonal bends.
edges = [
    ("e_history_motion", "histories", "motion_features", "1,0.5", "0,0.5", [], ARROW, False),
    ("e_motion_projection", "motion_features", "history_projection", "1,0.5", "0,0.5", [], ARROW, False),
    ("e_projection_mamba", "history_projection", "temporal_mamba", "1,0.5", "0,0.5", [], ARROW, False),
    ("e_mamba_last_state", "temporal_mamba", "last_state", "0.5,1", "0.5,0", [], ARROW, False),
    ("e_last_state_fusion", "last_state", "actor_fusion", "0.5,1", "0.7,0", [(1090, 505), (1000, 505)], ARROW, False),
    ("e_metadata_type", "metadata", "type_embedding", "1,0.3", "0,0.5", [(405, 575), (405, 554)], ARROW, False),
    ("e_metadata_position", "metadata", "position_embedding", "1,0.75", "0,0.5", [(405, 620), (405, 649)], ARROW, False),
    ("e_type_fusion", "type_embedding", "actor_fusion", "1,0.5", "0,0.3", [(735, 554)], ARROW, False),
    ("e_position_fusion", "position_embedding", "actor_fusion", "1,0.5", "0,0.8", [(735, 649), (735, 618)], ARROW, False),
    ("e_fusion_scene", "actor_fusion", "scene_transformer", "1,0.5", "0,0.65", [(1190, 591), (1190, 446)], ARROW, False),
    ("e_mask_scene", "actor_mask", "scene_transformer", "0.5,1", "0.5,0", [], SCENE_STROKE, True),
    ("e_transformer_norm", "scene_transformer", "scene_norm", "0.5,1", "0.5,0", [], SCENE_STROKE, False),
    ("e_norm_output", "scene_norm", "scene_features", "0.5,1", "0.5,0", [], SCENE_STROKE, False),
    ("e_validity_mask", "histories", "actor_mask", "1,0.15", "0.5,0", [(380, 267), (380, 195), (1545, 195)], SCENE_STROKE, True),
]


def drawio_style(fill, stroke, font_size=13, bold=True):
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.3;"
        f"fontColor={INK};fontFamily=Arial;fontSize={font_size};"
        f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;spacing=8;"
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
        "id": "demo-encoder-framework", "name": "Encoder Framework",
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
        add_vertex(graph, cell_id, label, box, drawio_style(fill, stroke, 15, True))

    node_map = {node[0]: node for node in nodes}
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
        add_vertex(graph, cell_id, label, box, drawio_style(fill, stroke, size, bold))

    add_vertex(graph, "title", "DeMo Encoder Framework", (34, 25, 1200, 42),
               "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
               f"fontColor={INK};fontFamily=Arial;fontSize=24;fontStyle=1;align=left;verticalAlign=middle;")
    add_vertex(graph, "subtitle", "Actor-wise temporal encoding followed by scene-level interaction",
               (36, 73, 1300, 28),
               "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
               f"fontColor={MUTED};fontFamily=Arial;fontSize=13;align=left;verticalAlign=middle;")
    add_vertex(graph, "shared_note", "Shared temporal encoder applied independently across actors",
               (920, 200, 340, 27),
               "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
               f"fontColor={MUTED};fontFamily=Arial;fontSize=11;fontStyle=2;align=center;verticalAlign=middle;")
    add_vertex(graph, "legend", "Solid arrows: feature flow   ·   Dashed arrows: validity-mask control",
               (36, 830, 1000, 25),
               "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
               f"fontColor={MUTED};fontFamily=Arial;fontSize=11;align=left;verticalAlign=middle;")
    add_vertex(graph, "scope_note", "Map-free actor-only encoder",
               (1320, 830, 460, 25),
               "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
               f"fontColor={MUTED};fontFamily=Arial;fontSize=11;align=right;verticalAlign=middle;")
    ET.indent(root, space="  ")
    DRAWIO.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(DRAWIO, encoding="utf-8", xml_declaration=True)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, round(size * SCALE))


def rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def measure_lines(label, font_obj, width):
    output = []
    for paragraph in label.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            output.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textlength(candidate, font=font_obj) <= width:
                line = candidate
            else:
                output.append(line)
                line = word
        output.append(line)
    return output


def draw_label(label, box, size, bold, color=INK, align="center"):
    x, y, width, height = box
    font_obj = font(size, bold)
    lines = measure_lines(label, font_obj, width * SCALE - 28 * SCALE)
    line_gap = 4 * SCALE
    heights = [draw.textbbox((0, 0), line or "Ag", font=font_obj)[3] -
               draw.textbbox((0, 0), line or "Ag", font=font_obj)[1] for line in lines]
    cursor_y = y * SCALE + (height * SCALE - sum(heights) - line_gap * (len(lines) - 1)) / 2
    for line, line_height in zip(lines, heights):
        line_width = draw.textlength(line, font=font_obj)
        if align == "left":
            cursor_x = x * SCALE + 28 * SCALE
        elif align == "right":
            cursor_x = (x + width) * SCALE - 28 * SCALE - line_width
        else:
            cursor_x = x * SCALE + (width * SCALE - line_width) / 2
        draw.text((round(cursor_x), round(cursor_y)), line, font=font_obj, fill=rgb(color))
        cursor_y += line_height + line_gap


def node_anchor(node_id, anchor):
    node = next(item for item in nodes if item[0] == node_id)
    x, y, width, height = node[2]
    ax, ay = (float(value) for value in anchor.split(","))
    return x + ax * width, y + ay * height


def draw_arrow(points, color, dashed=False):
    scaled = [(x * SCALE, y * SCALE) for x, y in points]
    line_color = rgb(color)
    line_width = 3
    for start, end in zip(scaled, scaled[1:]):
        x0, y0 = start
        x1, y1 = end
        if not dashed:
            draw.line((x0, y0, x1, y1), fill=line_color, width=line_width)
            continue
        segment = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if not segment:
            continue
        ux, uy = (x1 - x0) / segment, (y1 - y0) / segment
        offset = 0
        while offset < segment:
            finish = min(offset + 18, segment)
            draw.line((x0 + ux * offset, y0 + uy * offset,
                       x0 + ux * finish, y0 + uy * finish), fill=line_color, width=line_width)
            offset = finish + 12
    tip = scaled[-1]
    previous = scaled[-2]
    angle = atan2(tip[1] - previous[1], tip[0] - previous[0])
    length, half_width = 18, 8
    base = (tip[0] - cos(angle) * length, tip[1] - sin(angle) * length)
    left = (base[0] + cos(angle + 1.5708) * half_width,
            base[1] + sin(angle + 1.5708) * half_width)
    right = (base[0] + cos(angle - 1.5708) * half_width,
             base[1] + sin(angle - 1.5708) * half_width)
    draw.polygon([tip, left, right], fill=line_color)


def build_png():
    global draw
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), WHITE)
    draw = ImageDraw.Draw(image)

    for _, label, box, fill, stroke in panels:
        x, y, width, height = box
        rect = (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE)
        draw.rounded_rectangle(rect, radius=24, fill=rgb(fill), outline=rgb(stroke), width=3)
        draw_label(label, (x + 14, y + 7, width - 28, 42), 15, True, INK, "left")

    for edge in edges:
        _, source, target, source_anchor, target_anchor, bends, color, dashed = edge
        start = node_anchor(source, source_anchor)
        end = node_anchor(target, target_anchor)
        route = [start, *bends, end]
        if not bends:
            sx = float(source_anchor.split(",")[0])
            tx = float(target_anchor.split(",")[0])
            if sx in (0.0, 1.0) and tx in (0.0, 1.0):
                bend_x = (start[0] + end[0]) / 2
                route = [start, (bend_x, start[1]), (bend_x, end[1]), end]
            elif sx not in (0.0, 1.0) and tx not in (0.0, 1.0):
                bend_y = (start[1] + end[1]) / 2
                route = [start, (start[0], bend_y), (end[0], bend_y), end]
            elif sx in (0.0, 1.0):
                route = [start, (end[0], start[1]), end]
            else:
                route = [start, (start[0], end[1]), end]
        draw_arrow(route, color, dashed)

    for cell_id, label, box, fill, stroke, size, bold in nodes:
        x, y, width, height = box
        rect = (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE)
        draw.rounded_rectangle(rect, radius=18, fill=rgb(fill), outline=rgb(stroke), width=3)
        draw_label(label, box, size, bold)

    draw_label("DeMo Encoder Framework", (34, 25, 1200, 42), 24, True, INK, "left")
    draw_label("Actor-wise temporal encoding followed by scene-level interaction",
               (36, 73, 1300, 28), 13, False, MUTED, "left")
    draw_label("Shared temporal encoder applied independently across actors",
               (920, 200, 340, 27), 11, False, MUTED)
    draw_label("Solid arrows: feature flow   ·   Dashed arrows: validity-mask control",
               (36, 830, 1000, 25), 11, False, MUTED, "left")
    draw_label("Map-free actor-only encoder", (1320, 830, 460, 25), 11, False, MUTED, "right")

    PNG.parent.mkdir(parents=True, exist_ok=True)
    image.save(PNG, format="PNG", dpi=(300, 300), optimize=True)


if __name__ == "__main__":
    build_drawio()
    build_png()
    print(DRAWIO)
    print(PNG)
