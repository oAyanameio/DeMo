from base64 import b64encode
from html import escape
from pathlib import Path
from xml.sax.saxutils import quoteattr

from PIL import Image


SOURCE = Path('/home/lbh/.codex/attachments/40335601-9840-4e47-a788-258a52a62ed2/codex-clipboard-b783d3cd-a837-4a95-a603-8b8437bb56f4.png')
OUTPUT = Path('trajectory_decoding_architecture.drawio')


def attr(value):
    return quoteattr(str(value))


def cell(cell_id, parent='1', value='', style='', vertex=False, edge=False, source=None, target=None, geometry=''):
    attributes = [f'id={attr(cell_id)}', f'parent={attr(parent)}']
    if value != '':
        attributes.append(f'value={attr(value)}')
    if style:
        attributes.append(f'style={attr(style)}')
    if vertex:
        attributes.append('vertex="1"')
    if edge:
        attributes.append('edge="1"')
    if source is not None:
        attributes.append(f'source={attr(source)}')
    if target is not None:
        attributes.append(f'target={attr(target)}')
    return f'<mxCell {" ".join(attributes)}>{geometry}</mxCell>'


def geometry(x=None, y=None, width=None, height=None, relative=False, points=None):
    attributes = ['as="geometry"']
    if relative:
        attributes.append('relative="1"')
    for name, value in [('x', x), ('y', y), ('width', width), ('height', height)]:
        if value is not None:
            attributes.append(f'{name}={attr(value)}')
    body = ''
    if points:
        body = ''.join(
            f'<mxPoint x={attr(point_x)} y={attr(point_y)} as="point"/>'
            for point_x, point_y in points
        )
    return f'<mxGeometry {" ".join(attributes)}>{body}</mxGeometry>'


def vertex(cell_id, value, x, y, width, height, style, parent='1'):
    return cell(cell_id, parent=parent, value=value, style=style, vertex=True, geometry=geometry(x, y, width, height))


def edge(cell_id, source, target, style, parent='1', points=None, value=''):
    return cell(
        cell_id,
        parent=parent,
        value=value,
        style=style,
        edge=True,
        source=source,
        target=target,
        geometry=geometry(relative=True, points=points),
    )


def image_data(source_image, box):
    cropped = source_image.crop(box)
    output = bytearray()
    cropped.save(output := __import__('io').BytesIO(), format='PNG')
    return 'data:image/png;base64,' + b64encode(output.getvalue()).decode('ascii')


def text_style(font_size=16, bold=False, italic=False, color='#111111', align='center', valign='middle', extra=''):
    parts = [
        'html=1',
        'whiteSpace=wrap',
        'overflow=hidden',
        'strokeColor=none',
        'fillColor=none',
        'spacing=0',
        f'fontSize={font_size}',
        f'fontColor={color}',
        f'align={align}',
        f'verticalAlign={valign}',
    ]
    if bold:
        parts.append('fontStyle=1')
    if italic:
        parts.append('fontStyle=2' if not bold else 'fontStyle=3')
    if extra:
        parts.append(extra)
    return ';'.join(parts) + ';'


def box_style(fill, stroke='#000000', rounded=False, dashed=False, width=1, opacity=100, extra=''):
    parts = [
        'html=1',
        f'fillColor={fill}',
        f'strokeColor={stroke}',
        f'strokeWidth={width}',
        f'opacity={opacity}',
        f'rounded={1 if rounded else 0}',
    ]
    if dashed:
        parts.append('dashed=1')
        parts.append('dashPattern=6 4')
    if extra:
        parts.append(extra)
    return ';'.join(parts) + ';'


def image_style(data_uri):
    return f'shape=image;image={data_uri};imageAspect=0;aspect=fixed;html=1;'


def arrow_style(color='#111111', width=1.4, end='block', dashed=False, curved=False, extra=''):
    parts = [
        'edgeStyle=orthogonalEdgeStyle',
        'rounded=0',
        'orthogonalLoop=1',
        'jettySize=auto',
        f'strokeColor={color}',
        f'strokeWidth={width}',
        f'endArrow={end}',
        'endFill=1',
        'html=1',
    ]
    if dashed:
        parts.append('dashed=1')
    if curved:
        parts.append('curved=1')
        parts.append('edgeStyle=elbowEdgeStyle')
    if extra:
        parts.append(extra)
    return ';'.join(parts) + ';'


def add_text(cells, name, text, x, y, width, height, size=16, bold=False, italic=False, color='#111111', align='center', valign='middle', extra=''):
    cells.append(vertex(name, text, x, y, width, height, text_style(size, bold, italic, color, align, valign, extra)))


def add_rect(cells, name, x, y, width, height, fill, stroke='#000000', rounded=False, dashed=False, line_width=1, opacity=100, value='', extra=''):
    cells.append(vertex(name, value, x, y, width, height, box_style(fill, stroke, rounded, dashed, line_width, opacity, extra)))


def add_image(cells, name, data_uri, x, y, width, height):
    cells.append(vertex(name, '', x, y, width, height, image_style(data_uri)))


def add_arrow(cells, name, source, target, color='#111111', width=1.4, end='block', dashed=False, curved=False, points=None, extra=''):
    cells.append(edge(name, source, target, arrow_style(color, width, end, dashed, curved, extra), points=points))


def add_token_row(cells, prefix, start_x, y, colors, size=17, gap=2, skew=False):
    token_ids = []
    for index, color in enumerate(colors):
        token_id = f'{prefix}_{index}'
        token_ids.append(token_id)
        add_rect(cells, token_id, start_x + index * (size + gap), y, size, size, color, '#8a8a8a', False, False, 0.8)
        if skew:
            cells[-1] = cells[-1].replace('rounded=0;', 'rounded=0;skewX=-18;')
    return token_ids


def add_tensor_grid(cells, prefix, x, y, columns, rows, cell_width=20, cell_height=24):
    palette = ['#337ab7', '#337ab7', '#f4d6bc', '#f4b183', '#f4d6bc', '#ffd966', '#fff2cc']
    tensor_ids = []
    for row in range(rows):
        for column in range(columns):
            color = palette[column % len(palette)]
            tensor_id = f'{prefix}_{row}_{column}'
            tensor_ids.append(tensor_id)
            add_rect(cells, tensor_id, x + column * cell_width, y + row * cell_height, cell_width, cell_height, color, '#777777', False, False, 0.6)
    return tensor_ids


def main():
    source_image = Image.open(SOURCE).convert('RGB')
    cells = []
    cells.append('<mxCell id="0"/>')
    cells.append('<mxCell id="1" parent="0"/>')

    outer_main = 'main_panel'
    add_rect(cells, outer_main, 195, 8, 1253, 445, '#f2f7fc', '#f2f7fc', True, False, 0, 100)
    add_rect(cells, 'main_title_band', 195, 8, 1253, 63, '#dcebf7', '#dcebf7', False, False, 0)
    add_text(cells, 'main_title', 'Trajectory decoding with decoupled queries', 480, 22, 680, 32, size=20, bold=True)

    add_rect(cells, 'scene_panel', 6, 8, 182, 445, '#f7f7f7', '#f7f7f7', True, False, 0)
    add_rect(cells, 'scene_header', 6, 8, 182, 63, '#e7e7e7', '#e7e7e7', False, False, 0)
    add_text(cells, 'scene_header_text', 'Scene context<br/>encoding', 16, 18, 162, 46, size=18, bold=True)

    add_token_row(cells, 'scene_context_tokens', 29, 102, ['#f4b7b7', '#f4b7b7', '#f4b7b7', '#b9d7a8', '#b9d7a8', '#b9d7a8'], size=17, gap=2, skew=True)
    add_text(cells, 'scene_context_label', 'Scene Context', 24, 130, 145, 27, size=16, bold=True)
    add_rect(cells, 'encoder', 34, 201, 130, 50, '#d9d9d9', '#d9d9d9', False, False, 0, 100, '', 'shape=mxgraph.basic.trapezoid;direction=south;')
    add_text(cells, 'encoder_text', 'Encoder', 45, 212, 108, 26, size=17, bold=True)
    add_image(cells, 'left_map', image_data(source_image, (48, 303, 149, 406)), 48, 303, 101, 103)
    add_text(cells, 'left_map_label', 'HD Map &amp; Agents', 17, 408, 160, 28, size=16, bold=True)
    add_text(cells, 'left_arrow_1', '↑', 88, 151, 23, 50, size=27, bold=True)
    add_text(cells, 'left_arrow_2', '↑', 88, 257, 23, 42, size=27, bold=True)

    add_token_row(cells, 'mode_input', 221, 204, ['#f5b800'] * 6, size=18, gap=0)
    add_text(cells, 'mode_input_shape_label', '<i>(B, K, C)</i>', 207, 226, 140, 20, size=15, italic=True)
    add_text(cells, 'mode_input_label', 'Mode Queries', 205, 245, 150, 22, size=16, bold=True)
    add_image(cells, 'mode_car', image_data(source_image, (263, 158, 291, 204)), 263, 158, 28, 46)
    add_text(cells, 'mode_sun_left', '↗', 243, 98, 28, 42, size=30, color='#f0a800')
    add_text(cells, 'mode_sun_mid', '↑', 266, 88, 28, 50, size=30, color='#f0a800')
    add_text(cells, 'mode_sun_right', '↖', 295, 98, 28, 42, size=30, color='#f0a800')

    add_token_row(cells, 'mode_memory', 490, 90, ['#f4b7b7', '#f4b7b7', '#f4b7b7', '#b9d7a8', '#b9d7a8', '#b9d7a8'], size=17, gap=2, skew=True)
    add_token_row(cells, 'state_memory', 490, 274, ['#f4b7b7', '#f4b7b7', '#f4b7b7', '#b9d7a8', '#b9d7a8', '#b9d7a8'], size=17, gap=2, skew=True)

    add_rect(cells, 'mode_module', 385, 126, 317, 115, 'none', '#79a9e8', False, True, 1.4)
    add_rect(cells, 'state_module', 385, 311, 317, 108, 'none', '#79a9e8', False, True, 1.4)
    add_rect(cells, 'mode_cross', 396, 145, 121, 50, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_rect(cells, 'mode_self', 566, 145, 122, 50, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_text(cells, 'mode_cross_text', 'Mode<br/><b>CrossAttn</b>', 400, 150, 113, 40, size=16, bold=False)
    add_text(cells, 'mode_self_text', 'Mode<br/><b>SelfAttn</b>', 570, 150, 114, 40, size=16, bold=False)
    add_text(cells, 'mode_module_label', '(a)  Mode Localization Module', 403, 208, 281, 25, size=16, italic=False)

    add_rect(cells, 'state_query_bar', 303, 291, 18, 55, '#5598cf', '#407eae', False, False, 0.7)
    for index in range(4):
        add_rect(cells, f'state_query_line_{index}', 303, 291 + index * 13.75, 18, 0.8, '#cfe6f7', '#cfe6f7', False, False, 0)
    add_image(cells, 'state_car', image_data(source_image, (242, 322, 272, 367)), 242, 322, 30, 45)
    add_text(cells, 'state_query_shape_label', '<i>(B, T<sub>s</sub>, C)</i>', 239, 370, 145, 21, size=15, italic=True)
    add_text(cells, 'state_query_label', 'State Queries', 228, 392, 150, 23, size=16, bold=True)
    add_rect(cells, 'state_cross', 396, 327, 121, 50, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_rect(cells, 'state_bimamba', 566, 327, 122, 50, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_text(cells, 'state_cross_text', 'State<br/><b>CrossAttn</b>', 400, 332, 113, 40, size=16)
    add_text(cells, 'state_bimamba_text', 'State<br/><b>BiMamba</b>', 570, 332, 114, 40, size=16)
    add_text(cells, 'state_module_label', '(b)  State Consistency Module', 402, 389, 284, 25, size=16)

    add_tensor_grid(cells, 'central_tensor', 729, 192, 6, 5, 20, 24)
    add_text(cells, 'central_tensor_label', '<i>(B, K, T<sub>s</sub>, C)</i>', 716, 315, 165, 27, size=15, italic=True)
    add_text(cells, 'loss_ts', '<i>ℒ<sub>ts</sub></i>', 735, 107, 48, 23, size=17, italic=True)
    add_text(cells, 'loss_m', '<i>ℒ<sub>m</sub></i>', 738, 393, 45, 22, size=17, italic=True)

    add_rect(cells, 'hybrid_panel', 888, 101, 345, 322, '#e5edf8', '#e5edf8', True, False, 0)
    add_rect(cells, 'hybrid_module', 905, 149, 293, 221, 'none', '#79a9e8', False, True, 1.4)
    add_token_row(cells, 'hybrid_memory', 1077, 115, ['#f4b7b7', '#f4b7b7', '#f4b7b7', '#b9d7a8', '#b9d7a8', '#b9d7a8'], size=17, gap=2, skew=True)
    add_rect(cells, 'hybrid_cross', 924, 161, 160, 32, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_rect(cells, 'hybrid_self', 924, 214, 160, 32, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_rect(cells, 'hybrid_mode', 924, 267, 160, 32, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_rect(cells, 'hybrid_state', 924, 320, 160, 32, '#bfbfbf', '#a8a8a8', True, False, 1)
    add_text(cells, 'hybrid_cross_text', 'Hybrid CrossAttn', 931, 165, 146, 24, size=15, bold=True, align='left')
    add_text(cells, 'hybrid_self_text', 'Hybrid SelfAttn', 931, 218, 146, 24, size=15, bold=True, align='left')
    add_text(cells, 'hybrid_mode_text', 'Mode SelfAttn', 931, 271, 146, 24, size=15, bold=True, align='left')
    add_text(cells, 'hybrid_state_text', 'State BiMamba', 931, 324, 146, 24, size=15, bold=True, align='left')
    add_text(cells, 'hybrid_cross_shape', '<i>(B, KT<sub>s</sub>, C)</i>', 1090, 162, 100, 29, size=14, italic=True, align='left')
    add_text(cells, 'hybrid_self_shape', '<i>(B, KT<sub>s</sub>, C)</i>', 1090, 215, 100, 29, size=14, italic=True, align='left')
    add_text(cells, 'hybrid_mode_shape', '<i>(BT<sub>s</sub>, K, C)</i>', 1090, 268, 104, 29, size=14, italic=True, align='left')
    add_text(cells, 'hybrid_state_shape', '<i>(BK, T<sub>s</sub>, C)</i>', 1090, 321, 104, 29, size=14, italic=True, align='left')
    add_text(cells, 'hybrid_module_label', '(c)  Hybrid Coupling Module', 924, 386, 270, 25, size=16)

    add_image(cells, 'right_map', image_data(source_image, (1287, 193, 1390, 294)), 1288, 193, 101, 101)
    add_rect(cells, 'trajectory_legend', 1277, 316, 62, 21, '#ffe79c', '#7d7d7d', False, False, 0.8)
    add_text(cells, 'trajectory_text', '<i>Trajectories</i>', 1254, 343, 110, 25, size=16, italic=True, bold=True)
    add_rect(cells, 'probability_legend', 1277, 374, 62, 21, '#dae8fc', '#7d7d7d', False, False, 0.8)
    add_text(cells, 'probability_text', '<i>Probabilities</i>', 1252, 400, 115, 25, size=16, italic=True, bold=True)
    add_text(cells, 'loss_reg', '<i>ℒ<sub>reg</sub></i>', 1375, 315, 55, 22, size=17, italic=True, align='left')
    add_text(cells, 'loss_cls', '<i>ℒ<sub>cls</sub></i>', 1375, 374, 55, 22, size=17, italic=True, align='left')

    add_arrow(cells, 'edge_mode_query', 'mode_car', 'mode_cross', '#111111', 1.3, 'block', extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
    add_arrow(cells, 'edge_mode_cross_self', 'mode_cross', 'mode_self', '#111111', 1.3, 'block')
    add_arrow(cells, 'edge_state_query', 'state_car', 'state_cross', '#111111', 1.3, 'block', extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
    add_arrow(cells, 'edge_state_cross_bimamba', 'state_cross', 'state_bimamba', '#111111', 1.3, 'block')
    add_arrow(cells, 'edge_mode_memory', 'mode_memory_0', 'mode_cross', '#111111', 1.2, 'block', extra='exitX=0;exitY=0.5;entryX=0.5;entryY=0;')
    add_arrow(cells, 'edge_state_memory', 'state_memory_0', 'state_cross', '#111111', 1.2, 'block', extra='exitX=0;exitY=0.5;entryX=0.5;entryY=0;')
    add_arrow(cells, 'edge_mode_tensor', 'mode_self', 'central_tensor_0_0', '#111111', 1.3, 'block', points=[(748, 170), (748, 188)])
    add_arrow(cells, 'edge_state_tensor', 'state_bimamba', 'central_tensor_4_2', '#111111', 1.3, 'block', points=[(788, 352), (788, 330)])
    add_arrow(cells, 'edge_tensor_hybrid', 'central_tensor_2_5', 'hybrid_cross', '#111111', 1.4, 'block', extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
    add_arrow(cells, 'edge_hybrid_1', 'hybrid_cross', 'hybrid_self', '#111111', 1.2, 'block')
    add_arrow(cells, 'edge_hybrid_2', 'hybrid_self', 'hybrid_mode', '#111111', 1.2, 'block')
    add_arrow(cells, 'edge_hybrid_3', 'hybrid_mode', 'hybrid_state', '#111111', 1.2, 'block')
    add_arrow(cells, 'edge_hybrid_memory', 'hybrid_memory_0', 'hybrid_cross', '#111111', 1.2, 'block', extra='exitX=0;exitY=0.5;entryX=0.5;entryY=0;')
    add_arrow(cells, 'edge_hybrid_output', 'hybrid_state', 'right_map', '#111111', 1.3, 'block', extra='exitX=1;exitY=0.5;entryX=0;entryY=0.5;')

    add_arrow(cells, 'edge_scene_context', 'scene_context_label', 'encoder', '#111111', 1.1, 'block', extra='exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
    add_arrow(cells, 'edge_encoder_map', 'left_map', 'encoder', '#111111', 1.1, 'block', extra='exitX=0.5;exitY=0;entryX=0.5;entryY=1;')
    add_arrow(cells, 'edge_loss_ts', 'mode_self', 'loss_ts', '#111111', 1.1, 'block', extra='exitX=0.8;exitY=0;entryX=0.5;entryY=1;')
    add_arrow(cells, 'edge_loss_m', 'state_bimamba', 'loss_m', '#111111', 1.1, 'block', extra='exitX=0.8;exitY=1;entryX=0.5;entryY=0;')
    add_arrow(cells, 'edge_reg', 'right_map', 'loss_reg', '#111111', 1.1, 'open', extra='exitX=0.5;exitY=1;entryX=0;entryY=0.5;')
    add_arrow(cells, 'edge_cls', 'right_map', 'loss_cls', '#111111', 1.1, 'open', extra='exitX=0.5;exitY=1;entryX=0;entryY=0.5;')

    cells.extend([
        edge('orange_arrow_left', 'mode_car', 'mode_sun_left', arrow_style('#f0a800', 1.2, 'classic', curved=True), points=[(256, 158), (246, 137), (246, 111)]),
        edge('orange_arrow_mid', 'mode_car', 'mode_sun_mid', arrow_style('#f0a800', 1.2, 'classic', curved=True), points=[(277, 158), (277, 132), (277, 105)]),
        edge('orange_arrow_right', 'mode_car', 'mode_sun_right', arrow_style('#f0a800', 1.2, 'classic', curved=True), points=[(285, 158), (300, 137), (312, 111)]),
    ])

    model = (
        '<mxfile host="app.diagrams.net" modified="2026-09-21T00:00:00.000Z" agent="Codex" version="24.7.17">'
        '<diagram id="trajectory-decoding" name="Trajectory Decoding">'
        '<mxGraphModel dx="1460" dy="465" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1460" pageHeight="465" math="1" shadow="0">'
        '<root>' + ''.join(cells) + '</root>'
        '</mxGraphModel></diagram></mxfile>'
    )
    OUTPUT.write_text(model, encoding='utf-8')
    print(f'Wrote {OUTPUT.resolve()}')


if __name__ == '__main__':
    main()
