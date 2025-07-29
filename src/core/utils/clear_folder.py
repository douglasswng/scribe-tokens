from pathlib import Path
import shutil


def clear_folder(folder: Path, confirm: bool=True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    assert folder.is_dir()

    has_files = any(folder.iterdir())
    if confirm and has_files:
        response = input(f"Are you sure you want to clear {folder}? (y/n): ")
        if response != "y":
            return
    
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)