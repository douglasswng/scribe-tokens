import html
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import ujson as json

from constants import (
    DATASET,
    PARSED_DIR,
    RAW_DIR,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
)
from schemas.ink import DigitalInk
from schemas.parsed import Parsed
from utils.clear_folder import clear_folder

assert DATASET == "iam", "Dataset must be iam"

LINE_STROKES_DIR = RAW_DIR / "lineStrokes"
ORIGINAL_DIR = RAW_DIR / "original"
TRAIN1_PATH = RAW_DIR / "trainset.txt"
VAL1_PATH = RAW_DIR / "testset_v.txt"
VAL2_PATH = RAW_DIR / "testset_t.txt"
TEST_PATH = RAW_DIR / "testset_f.txt"


@lru_cache(maxsize=None)
def get_id_to_path_map() -> dict[str, Path]:
    id_to_path = {}
    for path in LINE_STROKES_DIR.rglob("*.xml"):
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
    for stroke in element.findall(".//Stroke"):
        stroke_points = []
        for point in stroke.findall(".//Point"):
            stroke_points.append((int(point.get("x")), int(point.get("y"))))  # type: ignore
        strokes.append(stroke_points)
    return strokes


def parse_element(element: ET.Element) -> list[Parsed]:
    form_element = element.find(".//Form")
    writer = form_element.get("writerID")  # type: ignore
    contents = []
    for content in element.findall(".//TextLine"):
        text = content.get("text")  # type: ignore
        assert text is not None
        text = html.unescape(text)
        id = content.get("id")  # type: ignore
        line_element = id_to_element(id)  # type: ignore
        if line_element is None:
            continue
        strokes = element_to_strokes(line_element)
        contents.append(
            Parsed(
                id=id,  # type: ignore
                text=text,  # type: ignore
                writer=writer,  # type: ignore
                ink=DigitalInk.from_coords(strokes),
            )
        )
    return contents


def save_parsed(parsed: Parsed) -> None:
    parsed_path = PARSED_DIR / f"{parsed.id}.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parsed_path, "w") as f:
        json.dump(parsed.model_dump(), f, indent=4)


def parse_split_files() -> None:
    """Parse raw split files and convert them to standardized format with exact filenames."""
    TRAIN_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Get all parsed file IDs
    all_parsed_ids = {path.stem for path in PARSED_DIR.glob("*.json")}

    # Map raw split files to output paths
    split_mapping = {
        TRAIN1_PATH: TRAIN_SPLIT_PATH,
        VAL1_PATH: VAL_SPLIT_PATH,
        VAL2_PATH: VAL_SPLIT_PATH,  # VAL1 and VAL2 both go to val
        TEST_PATH: TEST_SPLIT_PATH,
    }

    # Track IDs for each output split to handle VAL1 + VAL2
    split_ids = {TRAIN_SPLIT_PATH: [], VAL_SPLIT_PATH: [], TEST_SPLIT_PATH: []}

    # Process each split file
    for split_file, output_path in split_mapping.items():
        # Read prefixes from raw split file
        with open(split_file, "r") as f:
            prefixes = {line.strip() for line in f if line.strip()}

        # Find all matching parsed file IDs
        for parsed_id in all_parsed_ids:
            # Check if any prefix matches this parsed_id
            for prefix in prefixes:
                # Remove trailing 'x' from prefix if present
                clean_prefix = prefix.rstrip("x")
                if parsed_id.startswith(clean_prefix):
                    split_ids[output_path].append(parsed_id)
                    break

    # Write standardized split files
    for output_path, file_ids in split_ids.items():
        with open(output_path, "w") as f:
            for file_id in sorted(set(file_ids)):  # Use set to remove duplicates from VAL1+VAL2
                f.write(f"{file_id}\n")

        print(f"Generated {output_path.name} with {len(set(file_ids))} files")


def main() -> None:
    for path in sorted(ORIGINAL_DIR.rglob("*.xml")):
        element = ET.parse(path).getroot()
        parsed_elements = parse_element(element)
        for parsed in parsed_elements:
            save_parsed(parsed)

    # After parsing data, generate standardized split files
    print("\nGenerating standardized split files...")
    parse_split_files()


if __name__ == "__main__":
    clear_folder(PARSED_DIR)
    main()
