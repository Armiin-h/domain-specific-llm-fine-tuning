"""Stratified cap so rare clause types survive a laptop-sized train set."""

from __future__ import annotations

import random
from collections import defaultdict

from clausedraft.data.corpus import Provision


def stratified_sample(provisions: list[Provision], cap: int, seed: int) -> list[Provision]:
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("cap must be a positive integer")
    if len(provisions) <= cap:
        return list(provisions)

    by_type: dict[str, list[Provision]] = defaultdict(list)
    for item in provisions:
        by_type[item.clause_type].append(item)
    allocation = _allocate({name: len(rows) for name, rows in by_type.items()}, cap)

    chosen: list[Provision] = []
    for clause_type, rows in by_type.items():
        take = allocation[clause_type]
        if take >= len(rows):
            chosen.extend(rows)
            continue
        rng = random.Random(f"{seed}:{clause_type}")
        chosen.extend(rng.sample(rows, take))

    order = {id(item): index for index, item in enumerate(provisions)}
    chosen.sort(key=lambda item: order[id(item)])
    return chosen


def _allocate(counts: dict[str, int], cap: int) -> dict[str, int]:
    total = sum(counts.values())
    if cap >= total:
        return dict(counts)

    remaining = dict(counts)
    assigned = {name: 0 for name in counts}
    left = cap
    while left > 0 and any(count > 0 for count in remaining.values()):
        active = [name for name, count in remaining.items() if count > 0]
        share = left // len(active)
        if share == 0:
            order = sorted(active, key=lambda name: (-remaining[name], name))
            for name in order:
                if left == 0:
                    break
                assigned[name] += 1
                remaining[name] -= 1
                left -= 1
            break
        for name in active:
            take = min(share, remaining[name])
            assigned[name] += take
            remaining[name] -= take
            left -= take
    return assigned
