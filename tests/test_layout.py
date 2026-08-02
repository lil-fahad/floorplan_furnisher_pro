from shapely.geometry import box

from furniture_ai.layout.constraints import rect_polygon
from furniture_ai.layout.generator import furnish_floorplan, infer_room_types


def test_room_type_inference_marks_largest_as_living() -> None:
    rooms = [box(0, 0, 200, 150), box(250, 0, 750, 400), box(0, 200, 150, 280)]
    room_types = infer_room_types(rooms)
    assert room_types[1] == "Living"
    assert len(room_types) == len(rooms)


def test_layout_is_deterministic_and_collision_free() -> None:
    room = box(0, 0, 600, 450)
    vectorized = {"rooms": [room], "doors": [], "windows": []}

    first = furnish_floorplan(vectorized, room_types=["Living"])
    second = furnish_floorplan(vectorized, room_types=["Living"])
    assert first == second

    items = first["rooms"][0]["items"]
    polygons = [
        rect_polygon(item["cx"], item["cy"], item["w"], item["h"], item["angle"])
        for item in items
    ]
    assert items
    assert all(room.covers(polygon) for polygon in polygons)
    for index, polygon in enumerate(polygons):
        assert all(not polygon.intersects(other) for other in polygons[index + 1 :])


def test_physical_scale_is_applied_when_provided() -> None:
    room = box(0, 0, 1000, 1000)
    output = furnish_floorplan(
        {"rooms": [room], "doors": [], "windows": []},
        pixels_per_cm=2,
        room_types=["Bedroom"],
    )
    bed = next(item for item in output["rooms"][0]["items"] if item["name"] == "Bed")
    assert {bed["w"], bed["h"]} == {360.0, 400.0}
    assert bed["dimension_source"] == "physical"
