from pathlib import Path
from functools import lru_cache
import html
import xml.etree.ElementTree as ET

import ujson as json

from core.data_schema import Parsed, DigitalInk
from core.utils import clear_folder
from core.constants import RAW_IAM_DIR, PARSED_IAM_DIR


@lru_cache(maxsize=None)
def get_id_to_path_map() -> dict[str, Path]:
    line_dir = RAW_IAM_DIR / 'lineStrokes'
    id_to_path = {}
    for path in line_dir.rglob('*.xml'):
        id = path.stem
        id_to_path[id] = path
    return id_to_path


def id_to_element(id: str) -> ET.Element | None:
    id_to_path = get_id_to_path_map()
    if id not in id_to_path:
        print(f"Warning: ID '{id}' not found in line directory")
        return None
    path = id_to_path[id]
    return ET.parse(path).getroot()


def element_to_strokes(element: ET.Element) -> list[list[tuple[int, int]]]:
    strokes = []
    for stroke in element.findall('.//Stroke'):
        stroke_points = []
        for point in stroke.findall('.//Point'):
            stroke_points.append((int(point.get('x')), int(point.get('y'))))  # type: ignore
        strokes.append(stroke_points)
    return strokes


def parse_element(element: ET.Element) -> list[Parsed]:
    form_element = element.find('.//Form')
    writer = form_element.get('writerID')  # type: ignore
    contents = []
    for content in element.findall('.//TextLine'):
        text = content.get('text')  # type: ignore
        assert text is not None
        text = html.unescape(text)
        id = content.get('id')  # type: ignore
        line_element = id_to_element(id)  # type: ignore
        if line_element is None:
            continue
        strokes = element_to_strokes(line_element)
        contents.append(Parsed(id=id,  # type: ignore
                               text=text,  # type: ignore
                               writer=writer,  # type: ignore
                               ink=DigitalInk.from_coords(strokes)))
    return contents


def save_parsed(parsed: Parsed) -> None:
    parsed_path = PARSED_IAM_DIR / f'{parsed.id}.json'
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parsed_path, 'w') as f:
        json.dump(parsed.model_dump(), f, indent=4)


def main() -> None:
    original_dir = RAW_IAM_DIR / 'original'
    for path in sorted(original_dir.rglob('*.xml')):
        element = ET.parse(path).getroot()
        parsed_elements = parse_element(element)
        for parsed in parsed_elements:
            save_parsed(parsed)


if __name__ == "__main__":
    clear_folder(PARSED_IAM_DIR)
    main()