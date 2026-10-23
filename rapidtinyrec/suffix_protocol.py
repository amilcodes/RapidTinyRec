"""Two-variant order balance, fixed before measuring the frozen cohort."""
import collections

VARIANTS = ('gqa', 'combined_suffix')


def build_orders(users):
    groups = collections.defaultdict(list)
    for user in users:
        groups[user['prompt_tokens']].append(user['cohort_index'])
    orders = {}
    counter = 0
    for length, indices in sorted(groups.items()):
        for index in indices:
            orders[index] = VARIANTS if counter % 2 == 0 else VARIANTS[::-1]
            counter += 1
    assert counter == 500
    counts = {v:sum(order[0] == v for order in orders.values()) for v in VARIANTS}
    assert set(counts.values()) == {250}
    by_length = {str(n):{v:sum(orders[i][0]==v for i in indices) for v in VARIANTS}
                 for n,indices in groups.items()}
    assert all(abs(c['gqa']-c['combined_suffix']) <= 1 for c in by_length.values())
    return orders, {'global':counts,'by_length':by_length,
                    'rule':'Alternate first variant within sorted length strata, carrying parity between strata.'}
