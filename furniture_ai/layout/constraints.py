
from shapely.geometry import box
from shapely.affinity import rotate

def rect_polygon(cx, cy, w, h, angle=0):
    poly = box(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
    if angle:
        poly = rotate(poly, angle, origin=(cx, cy), use_radians=False)
    return poly

def is_valid_placement(room_poly, furn_poly, existing, gates, clearance=20.0):
    if not room_poly.buffer(-5).contains(furn_poly):
        return False
    for g in gates:
        if furn_poly.buffer(clearance).intersects(g):
            return False
    for e in existing:
        if furn_poly.intersects(e):
            return False
    return True
