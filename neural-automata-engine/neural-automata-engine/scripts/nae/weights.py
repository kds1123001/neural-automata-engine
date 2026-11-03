from __future__ import annotations

import json
import math
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Union

FloatBuffer = Union[array, Iterable[float]]


@dataclass
class NeuralWeights:
    PERCEPTION: ClassVar[int] = 4

    channels: int
    hidden: int
    w1: FloatBuffer
    b1: FloatBuffer
    w2: FloatBuffer
    b2: FloatBuffer
    step_size: float = 1.0
    state_clip: float = 0.0

    def __post_init__(self) -> None:
        for name, size in self.shapes().items():
            value = getattr(self, name)
            if not (isinstance(value, array) and value.typecode == "f"):
                value = array("f", value)
                setattr(self, name, value)
            if len(value) != size:
                raise ValueError(f"{name}: expected {size} values, got {len(value)}")

    def shapes(self) -> Dict[str, int]:
        pdim = self.channels * self.PERCEPTION
        return {
            "w1": self.hidden * pdim,
            "b1": self.hidden,
            "w2": self.channels * self.hidden,
            "b2": self.channels,
        }

    @classmethod
    def zeros(cls, channels: int, hidden: int, **extra: float) -> "NeuralWeights":
        pdim = channels * cls.PERCEPTION
        return cls(
            channels,
            hidden,
            w1=[0.0] * (hidden * pdim),
            b1=[0.0] * hidden,
            w2=[0.0] * (channels * hidden),
            b2=[0.0] * channels,
            **extra,
        )

    @classmethod
    def random(
        cls,
        channels: int,
        hidden: int,
        *,
        seed: int = 0,
        step_size: float = 0.1,
        state_clip: float = 1.0,
    ) -> "NeuralWeights":
        rng = random.Random(seed)
        pdim = channels * cls.PERCEPTION
        s1 = 1.0 / math.sqrt(pdim)
        s2 = 1.0 / math.sqrt(hidden)
        return cls(
            channels,
            hidden,
            w1=[rng.gauss(0.0, s1) for _ in range(hidden * pdim)],
            b1=[0.0] * hidden,
            w2=[rng.gauss(0.0, s2) for _ in range(channels * hidden)],
            b2=[0.0] * channels,
            step_size=step_size,
            state_clip=state_clip,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "channels": self.channels,
            "hidden": self.hidden,
            "step_size": self.step_size,
            "state_clip": self.state_clip,
            "w1": list(self.w1),
            "b1": list(self.b1),
            "w2": list(self.w2),
            "b2": list(self.b2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "NeuralWeights":
        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NeuralWeights":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
