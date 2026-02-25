def get_distance(bh, label):
    ref = {"person":170,"car":150,"truck":250,"bus":280,
           "motorcycle":110,"bicycle":100,"dog":60,"cat":30}.get(label,100)
    if bh == 0: return "far away"
    d = (ref * 700) / bh
    if d < 80:  return "very close"
    if d < 200: return "nearby"
    if d < 450: return "ahead"
    return "far away"

def get_zone(dist):
    return {"very close":"danger","nearby":"warn","ahead":"safe","far away":"safe"}.get(dist,"safe")
