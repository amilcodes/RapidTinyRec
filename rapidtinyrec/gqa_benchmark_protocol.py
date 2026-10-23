"""Deterministic order balancing for a three-variant, frozen-cohort benchmark."""
import collections
import itertools
import random

VARIANTS = ('eager', 'compiled', 'compiled_gqa')
PATTERN = ((0,1,2), (1,2,0), (2,0,1), (2,1,0), (0,2,1), (1,0,2))


def order_counts(orders):
    positions = {v: [sum(o[j] == v for o in orders) for j in range(3)] for v in VARIANTS}
    precedences = {a+' before '+b: sum(o.index(a) < o.index(b) for o in orders)
                   for a,b in itertools.combinations(VARIANTS,2)}
    return {'positions': positions, 'precedences': precedences}


def build_orders(users):
    """Every full six-request stratum block uses all permutations.

    Choose label permutations without inspecting outputs/timings, making global
    pair precedence exactly balanced and position counts differ by at most two.
    Within every history length, position counts differ by at most one.
    """
    groups = collections.defaultdict(list)
    for user in users:
        groups[user['prompt_tokens']].append(user['cohort_index'])
    assert len(users) == 500
    rng = random.Random(20261023)
    labels = list(itertools.permutations(VARIANTS))
    for attempt in range(20000):
        orders = {}
        for length, indices in sorted(groups.items()):
            chosen = rng.choice(labels)
            for j, index in enumerate(indices):
                orders[index] = tuple(chosen[k] for k in PATTERN[j % 6])
        counts = order_counts(list(orders.values()))
        if set(counts['precedences'].values()) != {250}:
            continue
        if max(x for row in counts['positions'].values() for x in row)-min(x for row in counts['positions'].values() for x in row) > 2:
            continue
        by_length = {str(length): order_counts([orders[i] for i in indices]) for length,indices in groups.items()}
        assert all(max(row)-min(row) <= 1 for c in by_length.values() for row in c['positions'].values())
        return orders, {'seed':20261023,'selected_attempt':attempt,'global':counts,'by_length':by_length}
    raise RuntimeError('Could not balance frozen cohort')
