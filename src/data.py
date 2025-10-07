from dataclasses import dataclass

@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float

    @property
    def coords(self):
        return [self.x, self.y, self.z]