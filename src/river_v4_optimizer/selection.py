from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping


def _component_sizes(nodes: set[str], pairs: Iterable[Mapping[str, Any]]) -> list[int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        left, right = str(pair["image_i"]), str(pair["image_j"])
        if left in nodes and right in nodes and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(nodes)
    sizes = []
    while remaining:
        root = remaining.pop()
        queue = deque([root])
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current] & remaining:
                remaining.remove(neighbor)
                queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def rewire_selection(
    selection: Mapping[str, Any],
    keyframes: Mapping[str, Mapping[str, Any]],
    geometry: Iterable[Mapping[str, Any]],
    *,
    remove: set[str],
    add: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_before = {str(value) for value in selection.get("selected_keyframes") or ()}
    unknown = (remove | add) - set(keyframes)
    if unknown:
        raise ValueError(f"unknown keyframes: {sorted(unknown)}")
    selected = (selected_before - remove) | add
    admitted = [
        dict(row)
        for row in geometry
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in selected
        and str(row.get("image_j")) in selected
    ]
    components = _component_sizes(selected, admitted)
    if components != [len(selected)]:
        raise ValueError(f"rewired selection is disconnected: {components}")
    segments = {str(keyframes[key]["segment_id"]) for key in selected}
    result = dict(selection)
    result["selected_keyframes"] = sorted(selected)
    result["active_segments"] = sorted(segments)
    result["admitted_pairs"] = admitted
    result["mapping_modes"] = {segment: "TRIANGULATE" for segment in sorted(segments)}
    result["selection_profile"] = "RIVER_V4_DIRECT_MAIN_CONNECTOR_REWIRE"
    receipt = {
        "artifact_type": "RIVER_V4_CONNECTOR_REWIRE",
        "selected_before": len(selected_before),
        "selected_after": len(selected),
        "removed_keyframes": sorted(remove),
        "added_keyframes": sorted(add),
        "admitted_pairs": len(admitted),
        "component_sizes_after": components,
    }
    return result, receipt
