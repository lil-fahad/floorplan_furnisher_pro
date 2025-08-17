
import random, json
from typing import Dict, Any, List
from pathlib import Path
from furniture_ai.layout.constraints import rect_polygon, is_valid_placement

def load_catalog(path: str = "configs/furniture_catalog.json") -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {
        "Bedroom":[{"name":"Bed","w":180,"h":200},{"name":"Wardrobe","w":60,"h":180}],
        "Living":[{"name":"Sofa","w":200,"h":90},{"name":"TV","w":120,"h":40}],
        "Kitchen":[{"name":"Counter","w":240,"h":60},{"name":"Table","w":140,"h":80}],
    }

def assign_room_type(idx: int) -> str:
    return ["Living","Bedroom","Kitchen","Bedroom"][idx % 4]

def furnish_room(room_poly, room_type: str, gates, rng: random.Random):
    catalog = load_catalog()
    items, placed = [], []
    for spec in catalog.get(room_type, []):
        for _ in range(80):
            minx,miny,maxx,maxy = room_poly.bounds
            cx = rng.uniform(minx+20, maxx-20); cy = rng.uniform(miny+20, maxy-20)
            angle = rng.choice([0,90])
            furn = rect_polygon(cx, cy, spec["w"], spec["h"], angle)
            if is_valid_placement(room_poly, furn, placed, gates):
                placed.append(furn)
                items.append({"name": spec["name"], "cx": cx, "cy": cy, "w": spec["w"], "h": spec["h"], "angle": angle})
                break
    return items

def furnish_floorplan(vec: Dict) -> Dict:
    out, rng = {"rooms": []}, random.Random(42)
    gates = (vec.get("doors",[]) + vec.get("windows",[]))
    for i, room in enumerate(vec.get("rooms", [])):
        rtype = assign_room_type(i)
        items = furnish_room(room, rtype, gates, rng)
        out["rooms"].append({"type": rtype, "polygon": list(room.exterior.coords), "items": items})
    return out
