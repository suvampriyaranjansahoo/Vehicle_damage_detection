import json
from pathlib import Path


def load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("class_names.json must contain an object mapping indices to labels")
    try:
        names = [data[str(i)] for i in range(len(data))]
    except KeyError as exc:
        raise ValueError("class_names.json indices must be contiguous from 0") from exc
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("class_names.json contains an invalid label")
    if len(names) != len(set(names)):
        raise ValueError("class_names.json contains duplicate labels")
    return names


def save_class_mapping(class_indices: dict[str, int], path: Path) -> list[str]:
    ordered = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]
    payload = {str(i): name for i, name in enumerate(ordered)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ordered
