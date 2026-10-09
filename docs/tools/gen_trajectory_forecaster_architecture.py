"""Generate an editable draw.io figure for DeMo's actor-only encoder."""

from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DRAWIO = ROOT / "docs" / "figures" / "demo_trajectory_forecaster_architecture.drawio"
WIDTH, HEIGHT = 1540, 760

INK = "#263746"
MUTED = "#657786"
ARROW = "#536878"
PANEL = "#F7F9FB"
PANEL_STROKE = "#D6DEE6"
INPUT = "#EAF2F8"
INPUT_STROKE = "#9DB8CC"
TEMPORAL = "#EEF4FA"
TEMPORAL_STROKE = "#91ACCA"
SCENE = "#F1EDF8"
SCENE_STROKE = "#AD9BC8"

panels = []
nodes = []
edges = []


def add_node(cell_id, label, x, y, w, h, style):
    nodes.append((cell_id, label, x, y, w, h, style))


def add_edge(cell_id, source, target, color=ARROW, dashed=False,
             source_xy="0.5,1", target_xy="0.5,0", waypoints=None):
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;"
        f"strokeColor={color};strokeWidth=1.6;endArrow=block;endFill=1;"
        f"exitX={source_xy.split(',')[0]};exitY={source_xy.split(',')[1]};"
        f"entryX={target_xy.split(',')[0]};entryY={target_xy.split(',')[1]};"
    )
    if dashed:
        style += "dashed=1;dashPattern=6 4;"
    edges.append((cell_id, source, target, style, waypoints or []))


def panel_style(fill, stroke):
    return (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.3;"
        f"fontColor={INK};fontFamily=Arial;fontSize=15;fontStyle=1;"
        "align=left;verticalAlign=top;spacingTop=14;spacingLeft=16;"
    )


def card_style(fill="#FFFFFF", stroke="#B8C5D0", size=14, bold=True):
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.3;"
        f"fontColor={INK};fontFamily=Arial;fontSize={size};"
        f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;"
        "spacing=7;"
    )


def text_style(color=MUTED, size=12, align="left", bold=False):
    return (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
        f"fontColor={color};fontFamily=Arial;fontSize={size};"
        f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;spacing=0;"
    )


# Major encoder stages.
panels.extend([
    ("panel_input", "INPUT & MOTION FEATURES", 30, 125, 330, 535,
     panel_style(INPUT, INPUT_STROKE)),
    ("panel_temporal", "ACTOR-WISE TEMPORAL ENCODING", 390, 125, 610, 535,
     panel_style(TEMPORAL, TEMPORAL_STROKE)),
    ("panel_scene", "SCENE CONTEXT ENCODING", 1030, 125, 480, 535,
     panel_style(SCENE, SCENE_STROKE)),
])

# Input and motion feature construction.
add_node("history", "Observed trajectory\nhistory", 65, 235, 120, 72,
         card_style(INPUT, INPUT_STROKE))
add_node("validity", "History validity\nmask", 205, 235, 120, 72,
         card_style(INPUT, INPUT_STROKE))
add_node("motion", "Motion feature construction\nRelative displacement · velocity change · validity",
         65, 410, 260, 92, card_style("#FFFFFF", INPUT_STROKE, 13))
add_node("actor_metadata", "Actor metadata\nCenter position · heading · actor type",
         65, 565, 260, 68, card_style("#FFFFFF", INPUT_STROKE, 12, False))

# Per-actor temporal encoding and metadata fusion.
add_node("history_projection", "History feature projection\nMLP", 430, 245, 180, 78,
         card_style("#FFFFFF", TEMPORAL_STROKE))
add_node("temporal_mamba", "Temporal sequence encoder\nMamba", 670, 245, 250, 78,
         card_style(TEMPORAL, TEMPORAL_STROKE))
add_node("last_state", "Last valid state selection", 670, 365, 250, 70,
         card_style("#FFFFFF", TEMPORAL_STROKE))
add_node("type_embedding", "Actor-type embedding", 430, 480, 180, 68,
         card_style("#FFFFFF", TEMPORAL_STROKE, 13))
add_node("position_embedding", "Position & heading projection\nMLP", 430, 565, 180, 68,
         card_style("#FFFFFF", TEMPORAL_STROKE, 12))
add_node("actor_fusion_type", "Feature addition", 670, 480, 250, 62,
         card_style(TEMPORAL, TEMPORAL_STROKE, 13))
add_node("actor_fusion_position", "Actor representation fusion", 670, 565, 250, 68,
         card_style(TEMPORAL, TEMPORAL_STROKE, 13))

# Scene-level actor interaction and encoder output.
add_node("scene_transformer", "Scene interaction\nTransformer", 1090, 300, 360, 105,
         card_style(SCENE, SCENE_STROKE, 15))
add_node("scene_mask", "Actor validity mask", 1090, 235, 170, 44,
         card_style("#FFFFFF", SCENE_STROKE, 11, False))
add_node("scene_encoding", "Scene-context actor representations", 1090, 485, 360, 82,
         card_style("#FFFFFF", SCENE_STROKE, 14))

# Data flow through history encoding, actor metadata fusion, and scene interaction.
add_edge("e_history_motion", "history", "motion", source_xy="0.5,1", target_xy="0.3,0")
add_edge("e_validity_motion", "validity", "motion", source_xy="0.5,1", target_xy="0.7,0")
add_edge("e_motion_projection", "motion", "history_projection",
         source_xy="1,0.5", target_xy="0,0.5")
add_edge("e_projection_mamba", "history_projection", "temporal_mamba",
         source_xy="1,0.5", target_xy="0,0.5")
add_edge("e_mamba_last", "temporal_mamba", "last_state")
add_edge("e_last_actor_fusion", "last_state", "actor_fusion_type")
add_edge("e_type_fusion", "type_embedding", "actor_fusion_type",
         source_xy="1,0.5", target_xy="0,0.5")
add_edge("e_fusion_position", "actor_fusion_type", "actor_fusion_position")
add_edge("e_position_fusion", "position_embedding", "actor_fusion_position",
         source_xy="1,0.5", target_xy="0,0.5")
add_edge("e_metadata_type", "actor_metadata", "type_embedding",
         source_xy="1,0.25", target_xy="0,0.5")
add_edge("e_metadata_position", "actor_metadata", "position_embedding",
         source_xy="1,0.75", target_xy="0,0.5")
add_edge("e_actor_scene", "actor_fusion_position", "scene_transformer",
         source_xy="1,0.5", target_xy="0,0.75")
add_edge("e_mask_scene", "validity", "scene_mask", color=SCENE_STROKE,
         dashed=True, source_xy="1,0.5", target_xy="0.5,0",
         waypoints=[(345, 271), (345, 170), (1175, 170)])
add_edge("e_scene_mask_transformer", "scene_mask", "scene_transformer",
         color=SCENE_STROKE, dashed=True, source_xy="0.5,1", target_xy="0.5,0")
add_edge("e_scene_output", "scene_transformer", "scene_encoding")


root = ET.Element("mxfile", {
    "host": "app.diagrams.net",
    "agent": "DeMo architecture figure generator",
    "version": "24.7.17",
    "type": "device",
})
diagram = ET.SubElement(root, "diagram", {
    "id": "demo-actor-only-encoder",
    "name": "DeMo Encoder",
})
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": str(WIDTH), "dy": str(HEIGHT), "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1",
    "pageWidth": str(WIDTH), "pageHeight": str(HEIGHT),
    "math": "0", "shadow": "0",
})
cells = ET.SubElement(model, "root")
ET.SubElement(cells, "mxCell", {"id": "0"})
ET.SubElement(cells, "mxCell", {"id": "1", "parent": "0"})


def add_vertex(cell_id, label, x, y, w, h, style):
    cell = ET.SubElement(cells, "mxCell", {
        "id": cell_id, "value": label, "style": style,
        "vertex": "1", "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "as": "geometry",
    })


for panel in panels:
    add_vertex(*panel)

for cell_id, source, target, style, waypoints in edges:
    cell = ET.SubElement(cells, "mxCell", {
        "id": cell_id, "style": style, "edge": "1", "parent": "1",
        "source": source, "target": target,
    })
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if waypoints:
        points = ET.SubElement(geometry, "Array", {"as": "points"})
        for x, y in waypoints:
            ET.SubElement(points, "mxPoint", {"x": str(x), "y": str(y)})

for node in nodes:
    add_vertex(*node)

add_vertex(
    "title", "DeMo Actor-Only Encoder Architecture", 30, 22, 1300, 42,
    text_style(INK, 23, bold=True),
)
add_vertex(
    "subtitle", "History encoding → actor representation fusion → scene interaction",
    32, 68, 1400, 28, text_style(MUTED, 13),
)
add_vertex(
    "legend", "Solid arrows: feature flow     Dashed arrows: validity-mask control",
    32, 685, 1000, 26, text_style(MUTED, 11),
)
add_vertex(
    "scope_note", "Map-free actor-only encoder",
    1080, 685, 420, 26, text_style(MUTED, 11, align="right"),
)

DRAWIO.parent.mkdir(parents=True, exist_ok=True)
ET.indent(root, space="  ")
ET.ElementTree(root).write(DRAWIO, encoding="utf-8", xml_declaration=True)
print(DRAWIO)
