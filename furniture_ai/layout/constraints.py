from __future__ import annotations

from collections.abc import Iterable

from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry


def rect_polygon(cx: float, cy: float, width: float, depth: float, angle: float = 0) -> Polygon:
    if width <= 0 or depth <= 0:
        raise ValueError("Furniture dimensions must be positive")
    polygon = box(cx - width / 2, cy - depth / 2, cx + width / 2, cy + depth / 2)
    if angle:
        polygon = rotate(polygon, angle, origin=(cx, cy), use_radians=False)
    return polygon


def is_valid_placement(
    room_poly: BaseGeometry,
    furniture_poly: BaseGeometry,
    existing: Iterable[BaseGeometry],
    gates: Iterable[BaseGeometry],
    *,
    clearance: float = 20.0,
    wall_margin: float = 5.0,
) -> bool:
    if clearance < 0 or wall_margin < 0:
        raise ValueError("Clearance and wall margin must be non-negative")
    if room_poly.is_empty or furniture_poly.is_empty:
        return False

    inner_room = room_poly.buffer(-wall_margin) if wall_margin else room_poly
    if inner_room.is_empty:
        inner_room = room_poly
    if not inner_room.covers(furniture_poly):
        return False

    if any(furniture_poly.buffer(clearance).intersects(gate) for gate in gates):
        return False
    return not any(furniture_poly.intersects(item) for item in existing)
