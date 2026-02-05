from pathlib import Path

import ujson as json

from constants import DATASET, PARSED_DIR, RAW_DIR
from schemas.ink import DigitalInk
from schemas.parsed import Parsed

assert DATASET == "deepwriting", "Dataset must be deepwriting"


def get_stroke_data_for_word(
    word_stroke: list[dict], word_ranges: list[list[int]]
) -> list[list[tuple[float, float]]]:
    """
    Convert raw stroke data to list of strokes with coordinates for a specific word.
    Each stroke is a list of (x, y) coordinate tuples.
    Only includes stroke points that are in the word's ranges.
    Properly handles pen events to separate strokes.
    """
    # Flatten all ranges for this word
    word_point_indices = set()
    for range_list in word_ranges:
        word_point_indices.update(range_list)

    # Filter points that belong to this word
    word_points = []
    for point_idx, point_data in enumerate(word_stroke):
        if point_idx in word_point_indices:
            word_points.append(point_data)

    # Parse strokes based on pen events
    return parse_strokes_from_events(word_points)


def parse_strokes_from_events(points: list[dict]) -> list[list[tuple[float, float]]]:
    """
    Parse stroke data based on pen events.
    ev=0: pen down (start new stroke)
    ev=1: pen move (continue current stroke)
    ev=2: pen up (end current stroke)
    """
    if not points:
        return []

    strokes = []
    current_stroke = []

    for point_data in points:
        x = float(point_data["x"])
        y = float(point_data["y"])
        ev = int(point_data.get("ev", 1))  # Default to pen move if not specified

        if ev == 0:  # Pen down - start new stroke
            # If we have a current stroke, save it
            if current_stroke:
                strokes.append(current_stroke)
            # Start new stroke
            current_stroke = [(x, y)]
        elif ev == 1:  # Pen move - continue current stroke
            current_stroke.append((x, y))
        elif ev == 2:  # Pen up - end current stroke
            current_stroke.append((x, y))
            # Save the completed stroke
            if current_stroke:
                strokes.append(current_stroke)
            current_stroke = []

    # If there's a remaining stroke (in case the last event wasn't pen up)
    if current_stroke:
        strokes.append(current_stroke)

    return strokes


def get_stroke_data(word_stroke: list[dict]) -> list[list[tuple[float, float]]]:
    """
    Convert raw stroke data to list of strokes with coordinates.
    Each stroke is a list of (x, y) coordinate tuples.
    Properly handles pen events to separate strokes.
    """
    return parse_strokes_from_events(word_stroke)


def get_word_samples(json_path: Path) -> list[dict]:
    """
    Extract all word samples from a JSON file.
    Each word becomes a separate sample.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_samples = []
    for sample_key, sample_data in data.items():
        if not isinstance(sample_data, dict):
            continue

        writer = str(sample_data.get("user_id", "unknown"))
        word_stroke = sample_data.get("word_stroke", [])
        wholeword_segments = sample_data.get("wholeword_segments", [])

        if not word_stroke or not wholeword_segments:
            continue

        # Check if word segmentation is valid
        if not sample_data.get(
            "is_word_segmentation_valid", True
        ):  # Default to True if not specified
            continue

        # Process each word segment
        for word_idx, word_segment in enumerate(wholeword_segments):
            word_label = word_segment.get("ocr_label", f"word_{word_idx}")
            word_ranges = word_segment.get("ranges", [])

            if not word_label.strip() or not word_ranges:
                continue

            word_samples.append(
                {
                    "text": word_label,
                    "writer": writer,
                    "word_stroke": word_stroke,
                    "word_ranges": word_ranges,
                    "sample_key": sample_key,
                    "word_idx": word_idx,
                }
            )

    return word_samples


def get_word_sample_id(json_path: Path, sample_key: str, word_idx: int, word_text: str) -> str:
    """
    Generate unique ID for a word sample.
    """
    base_name = json_path.stem
    # Clean word text for filename (handle special characters and spaces)
    clean_word = word_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    return f"{base_name}-{sample_key}-w{word_idx}-{clean_word}"


def to_parsed_word(word_sample: dict, sample_id: str) -> Parsed:
    """
    Convert word sample data to Parsed object.
    """
    strokes = get_stroke_data_for_word(word_sample["word_stroke"], word_sample["word_ranges"])
    ink = DigitalInk.from_coords(strokes)
    # ink = ink.scale(WORD_HEIGHT / ink.height)
    parsed = Parsed(
        id=sample_id,
        text=word_sample["text"],
        writer=word_sample["writer"],
        ink=ink,
    )
    return parsed


def save_parsed(parsed: Parsed) -> None:
    """
    Save parsed object to JSON file.
    """
    parsed_path = PARSED_DIR / f"{parsed.id}.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parsed_path, "w") as f:
        json.dump(parsed.model_dump(), f, indent=4)


def process_dataset(dataset_dir: Path) -> None:
    """
    Process a dataset directory, applying scaling if specified.
    """
    print(f"Processing {dataset_dir}...")
    json_files = list(dataset_dir.rglob("*.json"))

    for json_file in json_files[:]:  # Process first file for testing
        print(f"Processing {json_file}")
        word_samples = get_word_samples(json_file)

        for word_sample in word_samples:
            sample_id = get_word_sample_id(
                json_file, word_sample["sample_key"], word_sample["word_idx"], word_sample["text"]
            )
            parsed = to_parsed_word(word_sample, sample_id)
            save_parsed(parsed)


def main() -> None:
    # Process Deepwriting Dataset with scaling
    # deepwriting_dir = RAW_DIR / "Deepwriting Dataset"
    # process_dataset(deepwriting_dir)

    # Process IAM-OnDB Dataset without scaling
    iamondb_dir = RAW_DIR / "Iamondb Dataset"
    process_dataset(iamondb_dir)


if __name__ == "__main__":
    from utils.clear_folder import clear_folder

    clear_folder(PARSED_DIR, confirm=False)
    main()
