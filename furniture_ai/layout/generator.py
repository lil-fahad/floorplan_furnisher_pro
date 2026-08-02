from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from furniture_ai.layout.constraints import is_valid_placement, rect_polygon

_DEFAULT_CATALOG: dict[str, list[dict[str, Any]]] = {
    "Bedroom": [
        {"name": "Bed", "w": 180, "h": 200},
        {"name": "Wardrobe", "w": 60, "h": 180},
    ],
    "Living": [
        {"name": "Sofa", "w": 200, "h": 90},
        {"name": "TV", "w": 120, "h": 40},
    ],
    "Kitchen": [
        {"name": "Counter", "w": 240, "h": 60},
        {"name": "Table", "w": 140, "h": 80},
    ],
}
_WALL_ITEMS = {"Bed", "Wardrobe", "Sofa", "TV", "Counter"}
_LONG_SIDE_FRACTION = {
    "Bed": 0.52,
    "Wardrobe": 0.38,
    "Sofa": 0.50,
    "TV": 0.34,
    "Counter": 0.55,
    "Table": 0.34,
}


def load_catalog(path: str = "configs/furniture_catalog.json") -> dict[str, Any]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        return _DEFAULT_CATALOG
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Furniture catalog must be a JSON object")
    return payload


def infer_room_types(rooms: Sequence[BaseGeometry]) -> list[str]:
    """Infer coarse room types deterministically when no classifier output exists."""
    if not rooms:
        return []

    largest_index = max(range(len(rooms)), key=lambda index: rooms[index].area)
    largest_area = max(rooms[largest_index].area, 1.0)
    inferred: list[str] = []
    for index, room in enumerate(rooms):
        if index == largest_index:
            inferred.append("Living")
            continue
        min_x, min_y, max_x, max_y = room.bounds
        width = max_x - min_x
        height = max_y - min_y
        aspect = max(width, height) / max(min(width, height), 1.0)
        area_ratio = room.area / largest_area
        inferred.append("Kitchen" if aspect >= 2.1 or area_ratio <= 0.30 else "Bedroom")
    return inferred


def _scaled_dimensions(
    room: BaseGeometry,
    spec: dict[str, Any],
    pixels_per_cm: float | None,
) -> tuple[float, float]:
    width_cm = float(spec["w"])
    depth_cm = float(spec["h"])
    if width_cm <= 0 or depth_cm <= 0:
        raise ValueError(f"Invalid dimensions for {spec.get('name', 'furniture')}")
    if pixels_per_cm is not None:
        if pixels_per_cm <= 0:
            raise ValueError("pixels_per_cm must be positive")
        return width_cm * pixels_per_cm, depth_cm * pixels_per_cm

    min_x, min_y, max_x, max_y = room.bounds
    room_short_side = max(min(max_x - min_x, max_y - min_y), 1.0)
    fraction = _LONG_SIDE_FRACTION.get(str(spec.get("name")), 0.35)
    target_long_side = room_short_side * fraction
    scale = target_long_side / max(width_cm, depth_cm)
    return width_cm * scale, depth_cm * scale


def _candidate_centers(room: BaseGeometry, prefer_wall: bool) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = room.bounds
    fractions = (0.15, 0.30, 0.50, 0.70, 0.85)
    points = [
        Point(min_x + (max_x - min_x) * x_fraction, min_y + (max_y - min_y) * y_fraction)
        for x_fraction in fractions
        for y_fraction in fractions
    ]
    points = [point for point in points if room.covers(point)]
    if prefer_wall:
        points.sort(key=lambda point: (point.distance(room.boundary), point.y, point.x))
    else:
        centroid = room.centroid
        points.sort(key=lambda point: (point.distance(centroid), point.y, point.x))
    return [(point.x, point.y) for point in points]


def furnish_room(
    room_poly: BaseGeometry,
    room_type: str,
    gates: Iterable[BaseGeometry],
    existing: list[dict[str, Any]] | None = None,
    *,
    pixels_per_cm: float | None = None,
    catalog: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    active_catalog = catalog or load_catalog()
    items = [dict(item) for item in (existing or [])]
    placed = [
        rect_polygon(
            float(item["cx"]),
            float(item["cy"]),
            float(item["w"]),
            float(item["h"]),
            float(item.get("angle", 0)),
        )
        for item in items
    ]

    min_x, min_y, max_x, max_y = room_poly.bounds
    short_side = max(min(max_x - min_x, max_y - min_y), 1.0)
    clearance = short_side * 0.04
    wall_margin = short_side * 0.01

    for spec in active_catalog.get(room_type, []):
        name = str(spec["name"])
        if any(item.get("name") == name for item in items):
            continue
        width, depth = _scaled_dimensions(room_poly, spec, pixels_per_cm)
        centers = _candidate_centers(room_poly, prefer_wall=name in _WALL_ITEMS)
        placed_item: dict[str, Any] | None = None
        for center_x, center_y in centers:
            for angle in (0.0, 90.0):
                candidate = rect_polygon(center_x, center_y, width, depth, angle)
                if is_valid_placement(
                    room_poly,
                    candidate,
                    placed,
                    gates,
                    clearance=clearance,
                    wall_margin=wall_margin,
                ):
                    placed.append(candidate)
                    placed_item = {
                        "name": name,
                        "cx": center_x,
                        "cy": center_y,
                        "w": width,
                        "h": depth,
                        "angle": angle,
                        "dimension_source": "physical" if pixels_per_cm else "room-relative",
                    }
                    break
            if placed_item is not None:
                items.append(placed_item)
                break
    return items


def furnish_floorplan(
    vec: dict[str, Any],
    detections: list[dict[str, Any]] | None = None,
    *,
    pixels_per_cm: float | None = None,
    room_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    rooms = list(vec.get("rooms", []))
    inferred_types = list(room_types) if room_types is not None else infer_room_types(rooms)
    if len(inferred_types) != len(rooms):
        raise ValueError("room_types length must match the number of room polygons")

    output: dict[str, Any] = {
        "schema_version": "2.0",
        "rooms": [],
        "scale": {"pixels_per_cm": pixels_per_cm, "source": "provided" if pixels_per_cm else "relative"},
    }
    gates = list(vec.get("doors", [])) + list(vec.get("windows", []))
    for index, room in enumerate(rooms):
        existing = [
            detection
            for detection in detections or []
            if room.covers(Point(float(detection["cx"]), float(detection["cy"])))
        ]
        items = furnish_room(
            room,
            inferred_types[index],
            gates,
            existing,
            pixels_per_cm=pixels_per_cm,
        )
        output["rooms"].append(
            {
                "id": f"room-{index + 1}",
                "type": inferred_types[index],
                "type_source": "provided" if room_types is not None else "heuristic",
                "polygon": list(room.exterior.coords),
                "items": items,
            }
        )
    return output
