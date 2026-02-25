def get_direction(cx, w):
    r = cx / w
    if r < 0.20: return "far left"
    if r < 0.40: return "to your left"
    if r < 0.60: return "straight ahead"
    if r < 0.80: return "to your right"
    return "far right"
