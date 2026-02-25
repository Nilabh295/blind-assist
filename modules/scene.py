from collections import Counter

def generate_scene_description(results, conf_threshold, model_names):
    counts = Counter()
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) >= conf_threshold:
                counts[model_names[int(box.cls[0])]] += 1
    
    if not counts:
        return "Scene is clear, no objects detected."
    
    parts = [f"{n} {l}s" if n > 1 else f"a {l}" for l, n in counts.most_common(5)]
    if len(parts) == 1:   s = parts[0]
    elif len(parts) == 2: s = f"{parts[0]} and {parts[1]}"
    else:                 s = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"I can see {s}."
