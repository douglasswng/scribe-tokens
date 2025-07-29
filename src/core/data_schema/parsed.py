from typing import Self
from pathlib import Path
import random

import ujson as json
from pydantic import BaseModel

from core.data_schema.ink import DigitalInk
from core.constants import PARSED_DIR


class Parsed(BaseModel):
    id: str
    text: str
    writer: str
    ink: DigitalInk

    def __str__(self) -> str:
        return (f"Id: {self.id}\n"
                f"Text: {self.text}\n"
                f"Writer: {self.writer}\n"
                f"{self.ink}")

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
    
    @classmethod
    def load_test(cls) -> Self:
        return cls(id="test",
                   text="test",
                   writer="test",
                   ink=DigitalInk.load_test())
    
    def visualise(self) -> None:
        self.ink.visualise(name=self.text)
    

if __name__ == "__main__":
    parsed = Parsed.load_random()
    print(parsed)
    parsed.visualise()