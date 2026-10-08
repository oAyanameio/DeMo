"""Generate an editable draw.io architecture figure for DeMo's forecaster."""

from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DRAWIO = ROOT / "docs" / "figures" / "demo_trajectory_forecaster_architecture.drawio"
WIDTH, HEIGHT = 2160, 960

INK = "#263746"
MUTED = "#657786"
EDGE = "#536878"
PANEL = "#F7F9FB"
PANEL_STROKE = "#D6DEE6"
INPUT = "#E9F2F8"
INPUT_STROKE = "#9DB8CC"
STATE = "#EAF2FA"
STATE_STROKE = "#82A9D0"
MODE = "#FBF3E4"
MODE_STROKE = "#D7B66F"
HYBRID = "#F1ECF8"
HYBRID_STROKE = "#A895C6"
OUTPUT = "#EAF4EF"
OUTPUT_STROKE = "#8BB39C"


root = ET.Element("mxfile", {
    "host": "app.diagrams.net",
    "agent": "DeMo architecture figure generator",
    "version": "24.7.17",
    "type": "device",
})
diagram = ET.SubElement(root, "diagram", {
    "id": "demo-trajectory-forecaster",
    "name": "DeMo Trajectory Forecaster",
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


def vertex(cell_id, label, x, y, w, h, style, parent="1"):
    cell = ET.SubElement(cells, "mxCell", {
        "id": cell_id, "value": label, "style": style,
        "vertex": "1", "parent": parent,
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "as": "geometry",
    })


def panel_style(fill=PANEL, stroke=PANEL_STROKE):
    return (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.4;"
        f"fontColor={INK};fontFamily=Arial;fontSize=16;fontStyle=1;"
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


def note_style(color=MUTED, size=11, align="left"):
    return (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
        f"fontColor={color};fontFamily=Arial;fontSize={size};"
        f"align={align};verticalAlign=middle;spacing=0;"
    )


def edge(cell_id, source, target, color=EDGE, dashed=False, source_xy="0.5,1", target_xy="0.5,0"):
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;"
        f"strokeColor={color};strokeWidth=1.6;endArrow=block;endFill=1;"
        f"exitX={source_xy.split(',')[0]};exitY={source_xy.split(',')[1]};"
        f"entryX={target_xy.split(',')[0]};entryY={target_xy.split(',')[1]};"
    )
    if dashed:
        style += "dashed=1;dashPattern=6 4;"
    cell = ET.SubElement(cells, "mxCell", {
        "id": cell_id, "style": style, "edge": "1", "parent": "1",
        "source": source, "target": target,
    })
    ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})


# Background panels are placed first so connectors and modules remain editable.
vertex("panel_input", "1  INPUT & HISTORY ENCODING", 25, 112, 385, 800,
       panel_style(INPUT, INPUT_STROKE))
vertex("panel_context", "2  SCENE CONTEXT ENCODER", 435, 112, 315, 770,
       panel_style())
vertex("panel_decoder", "3  DECOUPLED QUERY DECODER", 775, 112, 570, 770,
       panel_style())
vertex("panel_hybrid", "4  HYBRID COUPLING", 1370, 112, 425, 770,
       panel_style(HYBRID, HYBRID_STROKE))
vertex("panel_output", "5  PREDICTIONS", 1820, 112, 315, 770,
       panel_style(OUTPUT, OUTPUT_STROKE))

# Sub-lane backgrounds.
vertex("lane_state", "STATE CONSISTENCY", 800, 190, 520, 270,
       "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#F5F9FD;"
       f"strokeColor={STATE_STROKE};strokeWidth=1.2;dashed=1;dashPattern=6 4;"
       f"fontColor={STATE_STROKE};fontFamily=Arial;fontSize=12;fontStyle=1;"
       "align=left;verticalAlign=top;spacingTop=10;spacingLeft=12;")
vertex("lane_mode", "MODE LOCALIZATION", 800, 490, 520, 300,
       "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#FFFCF5;"
       f"strokeColor={MODE_STROKE};strokeWidth=1.2;dashed=1;dashPattern=6 4;"
       f"fontColor={MODE_STROKE};fontFamily=Arial;fontSize=12;fontStyle=1;"
       "align=left;verticalAlign=top;spacingTop=10;spacingLeft=12;")

# Input and agent-history path.
vertex("history", "Observed actor histories\npositions  p<sub>t</sub>  +  validity mask  m<sub>t</sub>\n[B, N, L, 2] + [B, N, L]",
       55, 205, 325, 88, card_style(INPUT, INPUT_STROKE, 13))
vertex("motion", "Motion feature construction\n[Δp<sub>t</sub>, Δv<sub>t</sub>, m<sub>t</sub>] ∈ ℝ<sup>4</sup>\n[B, N, L, 4]",
       55, 325, 325, 90, card_style())
vertex("history_proj", "History projection\nMLP: 4 → 64 → 128",
       55, 445, 325, 76, card_style())
vertex("history_mamba", "Agent temporal encoder\nUni-Mamba × 4  ·  C = 128",
       55, 550, 325, 82, card_style(STATE, STATE_STROKE))
vertex("history_token", "Last valid hidden state\nactor tokens  H<sub>hist</sub> ∈ ℝ<sup>B×N×128</sup>",
       55, 660, 325, 74, card_style())
vertex("actor_meta", "Actor context metadata\ncenter + heading (cos θ, sin θ) + type",
       55, 775, 325, 70, card_style("#FFFFFF", INPUT_STROKE, 12, False))

# Scene context encoder.
vertex("context_input", "History tokens + actor-type embedding",
       465, 220, 255, 65, card_style())
vertex("context_fusion", "Add projected center / heading\nMLP: 4 → 128",
       465, 330, 255, 80, card_style())
vertex("context_transformer", "Scene interaction\nTransformer blocks × 5\n8 heads · LayerNorm",
       465, 465, 255, 110, card_style(INPUT, INPUT_STROKE, 13))
vertex("scene_memory", "Scene memory  E\n[B, N, 128]  ·  actor-only",
       465, 640, 255, 74, card_style(OUTPUT, OUTPUT_STROKE))
vertex("context_mask_note", "Padding actors masked; focal actor is index 0.",
       465, 740, 255, 32, note_style())

# State-query branch.
vertex("time_query", "Future time embeddings\nMLP: t → 64 → 128\nQ<sub>s</sub> ∈ ℝ<sup>B×T×C</sup>",
       820, 245, 145, 100, card_style(STATE, STATE_STROKE, 12))
vertex("state_cross", "Cross-attention × 2\n(query: future time; key/value: E)",
       1000, 245, 145, 100, card_style(STATE, STATE_STROKE, 12))
vertex("state_mamba", "Bi-Mamba × 2\nRMSNorm",
       1000, 370, 145, 62, card_style(STATE, STATE_STROKE, 12))
vertex("state_latent", "State features\nH<sub>s</sub> ∈ ℝ<sup>B×T×C</sup>",
       1175, 350, 125, 78, card_style(STATE, STATE_STROKE, 12))

# Mode-query branch.
vertex("mode_query", "Focal scene token E<sub>0</sub>\nlearned mode queries\nK = 20",
       820, 590, 145, 95, card_style(MODE, MODE_STROKE, 12))
vertex("mode_cross", "Cross-attention × 3\n(query: modes; key/value: E)",
       1000, 565, 145, 88, card_style(MODE, MODE_STROKE, 12))
vertex("mode_self", "Mode self-attention × 3",
       1000, 675, 145, 60, card_style(MODE, MODE_STROKE, 12))
vertex("mode_latent", "Mode features\nH<sub>m</sub> ∈ ℝ<sup>B×K×C</sup>",
       1175, 660, 125, 78, card_style(MODE, MODE_STROKE, 12))

# Hybrid fusion and coupled representation refinement.
vertex("hybrid_sum", "Hybrid query fusion\nH<sub>m</sub> ⊕ H<sub>s</sub> → [B, K, T, C]",
       1410, 235, 345, 78, card_style(HYBRID, HYBRID_STROKE, 13))
vertex("hybrid_cross", "Hybrid cross-attention × 3  (context: E)",
       1410, 350, 345, 64, card_style())
vertex("hybrid_joint", "Joint self-attention × 3",
       1410, 445, 345, 64, card_style())
vertex("hybrid_mode", "Mode-interaction self-attention × 3",
       1410, 540, 345, 64, card_style())
vertex("hybrid_mamba", "Coupled temporal Bi-Mamba × 2",
       1410, 635, 345, 64, card_style())
vertex("hybrid_head", "Coupled trajectory head\nmean + mode score + Laplace scale",
       1410, 735, 345, 82, card_style(HYBRID, HYBRID_STROKE, 13))

# Prediction heads and outputs.
vertex("final_output", "FINAL MULTIMODAL FORECAST\nŶ ∈ ℝ<sup>B×K×T×2</sup>\nπ ∈ ℝ<sup>B×K</sup>  ·  scales b",
       1850, 230, 255, 120, card_style(OUTPUT, OUTPUT_STROKE, 14))
vertex("mode_aux", "Mode trajectory head (auxiliary)\nŶ<sub>mode</sub>, π<sub>mode</sub>, b<sub>mode</sub>",
       1850, 460, 255, 90, card_style("#FFFFFF", MODE_STROKE, 12, False))
vertex("state_aux", "State forecast head (auxiliary)\nŶ<sub>state</sub> ∈ ℝ<sup>B×T×2</sup>",
       1850, 625, 255, 90, card_style("#FFFFFF", STATE_STROKE, 12, False))
vertex("neighbor_aux", "Neighbor prediction head\nother-agent trajectories (auxiliary)",
       1850, 755, 255, 70, card_style("#FFFFFF", PANEL_STROKE, 11, False))

# Optional missingness conditioning is deliberately not part of the default path.
vertex("optional_gap", "Optional C-MAS-Δ\nmask + log-gap conditioning\n(replaces standard history Mamba only when enabled)",
       55, 845, 325, 60,
       "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
       f"strokeColor={MODE_STROKE};strokeWidth=1.3;dashed=1;dashPattern=6 4;"
       f"fontColor={INK};fontFamily=Arial;fontSize=11;align=center;verticalAlign=middle;spacing=5;")
vertex("shared_context_note", "Shared scene memory E is the key/value context for every cross-attention block.",
       800, 825, 525, 30, note_style(MUTED, 10))

# Connectors are inserted after background shapes and before foreground cards.
for ident, source, target, color, dashed, source_xy, target_xy in [
    ("e_history_motion", "history", "motion", EDGE, False, "0.5,1", "0.5,0"),
    ("e_motion_proj", "motion", "history_proj", EDGE, False, "0.5,1", "0.5,0"),
    ("e_proj_mamba", "history_proj", "history_mamba", EDGE, False, "0.5,1", "0.5,0"),
    ("e_mamba_token", "history_mamba", "history_token", EDGE, False, "0.5,1", "0.5,0"),
    ("e_token_context", "history_token", "context_input", EDGE, False, "1,0.5", "0,0.5"),
    ("e_meta_fusion", "actor_meta", "context_fusion", EDGE, False, "1,0.2", "0,0.8"),
    ("e_context_fusion", "context_input", "context_fusion", EDGE, False, "0.5,1", "0.5,0"),
    ("e_fusion_transformer", "context_fusion", "context_transformer", EDGE, False, "0.5,1", "0.5,0"),
    ("e_transformer_memory", "context_transformer", "scene_memory", EDGE, False, "0.5,1", "0.5,0"),
    ("e_time_cross", "time_query", "state_cross", STATE_STROKE, False, "1,0.5", "0,0.5"),
    ("e_state_cross_mamba", "state_cross", "state_mamba", STATE_STROKE, False, "0.5,1", "0.5,0"),
    ("e_state_latent", "state_mamba", "state_latent", STATE_STROKE, False, "1,0.5", "0,0.5"),
    ("e_mode_cross", "mode_query", "mode_cross", MODE_STROKE, False, "1,0.5", "0,0.5"),
    ("e_mode_self", "mode_cross", "mode_self", MODE_STROKE, False, "0.5,1", "0.5,0"),
    ("e_mode_latent", "mode_self", "mode_latent", MODE_STROKE, False, "1,0.5", "0,0.5"),
    ("e_state_fusion", "state_latent", "hybrid_sum", STATE_STROKE, False, "1,0.5", "0,0.3"),
    ("e_mode_fusion", "mode_latent", "hybrid_sum", MODE_STROKE, False, "1,0.5", "0,0.8"),
    ("e_fusion_cross", "hybrid_sum", "hybrid_cross", HYBRID_STROKE, False, "0.5,1", "0.5,0"),
    ("e_hybrid_joint", "hybrid_cross", "hybrid_joint", HYBRID_STROKE, False, "0.5,1", "0.5,0"),
    ("e_hybrid_mode", "hybrid_joint", "hybrid_mode", HYBRID_STROKE, False, "0.5,1", "0.5,0"),
    ("e_hybrid_mamba", "hybrid_mode", "hybrid_mamba", HYBRID_STROKE, False, "0.5,1", "0.5,0"),
    ("e_hybrid_head", "hybrid_mamba", "hybrid_head", HYBRID_STROKE, False, "0.5,1", "0.5,0"),
    ("e_final", "hybrid_head", "final_output", OUTPUT_STROKE, False, "1,0.35", "0,0.5"),
    ("e_state_aux", "state_latent", "state_aux", STATE_STROKE, True, "1,1", "0,0.5"),
    ("e_mode_aux", "mode_latent", "mode_aux", MODE_STROKE, True, "1,1", "0,0.5"),
    ("e_neighbor_aux", "scene_memory", "neighbor_aux", MUTED, True, "1,1", "0,0.5"),
    ("e_gap_optional", "optional_gap", "history_mamba", MODE_STROKE, True, "0.5,0", "0.5,1"),
]:
    edge(ident, source, target, color, dashed, source_xy, target_xy)

# Header, compact legend, and figure note.
vertex("title", "DeMo: Actor-Only Trajectory Forecasting Architecture",
       28, 18, 1600, 42,
       "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
       f"fontColor={INK};fontFamily=Arial;fontSize=24;fontStyle=1;align=left;verticalAlign=middle;")
vertex("subtitle", "TrajectoryForecaster · map-free scene context · default setting: L = 8, T = 12, K = 20, C = 128",
       30, 64, 1700, 28, note_style(MUTED, 13))
vertex("legend", "Blue: state consistency     Amber: mode localization     Purple: hybrid coupling     Dashed outline / arrow: optional or auxiliary path",
       30, 925, 1820, 24, note_style(MUTED, 11))
vertex("figure_note", "Actor-only implementation; no HD-map or lane encoder.",
       1690, 925, 445, 24, note_style(MUTED, 11, "right"))

# Keep panels behind connectors and modules above connectors for clean routing.
ordered_cells = list(cells)
structural = [cell for cell in ordered_cells if cell.get("id") in {"0", "1"}]
panels = [cell for cell in ordered_cells if (cell.get("id") or "").startswith("panel_")]
connectors = [cell for cell in ordered_cells if cell.get("edge") == "1"]
foreground = [cell for cell in ordered_cells
              if cell not in structural and cell not in panels and cell not in connectors]
for cell in ordered_cells:
    cells.remove(cell)
for cell in structural + panels + connectors + foreground:
    cells.append(cell)

ET.indent(root, space="  ")
DRAWIO.parent.mkdir(parents=True, exist_ok=True)
ET.ElementTree(root).write(DRAWIO, encoding="utf-8", xml_declaration=True)
print(DRAWIO)
