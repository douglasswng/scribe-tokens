import random
from pathlib import Path
from typing import Self

import ujson as json
from pydantic import BaseModel

from constants import PARSED_DIR
from schemas.ink import DigitalInk


class Parsed(BaseModel):
    id: str
    text: str
    writer: str
    ink: DigitalInk

    def __str__(self) -> str:
        return f"Id: {self.id}\nText: {self.text}\nWriter: {self.writer}\n{self.ink}"

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def load_random(cls) -> Self:
        paths = list(PARSED_DIR.rglob("*.json"))
        random_path = random.choice(paths)
        print(f"Random path: {random_path}")
        return cls.from_path(random_path)

    def visualise(self) -> None:
        self.ink.visualise(name=self.text)
