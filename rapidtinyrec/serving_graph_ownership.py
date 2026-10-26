"""Untimed retained graph-pool identities for the pinned Torch CUDA build.

Pool tokens do not enumerate tensor allocations or prove launch stream/overlap.
Keep graph references alive until both lanes have been compared.
"""
import re


def _token(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2 or
            any(type(x) is not int or x < 0 for x in value) or
            (value[0] == 0) == (value[1] == 0)):
        raise ValueError('Expected nonzero two-integer pinned CUDA pool token')
    return tuple(value)


def snapshot_graphs(replay, expected_graphs):
    if type(expected_graphs) is not int or expected_graphs <= 0:
        raise ValueError('Explicit positive graph count required')
    if not replay.frozen or len(replay.entries) != expected_graphs:
        raise ValueError('Frozen graph count differs from the protocol')
    graphs = tuple(entry[1] for entry in replay.entries.values())
    if len({id(graph) for graph in graphs}) != expected_graphs:
        raise ValueError('Distinct shape entries must own distinct graphs')
    records = []
    for index, graph in enumerate(graphs):
        primary = _token(graph.pool())
        retained = [_token(token) for token in graph.pools()]
        if not retained or len(set(retained)) != len(retained) or primary not in retained:
            raise ValueError('Graph must retain its primary pool exactly once')
        records.append(dict(index=index, primary=list(primary),
                            retained=[list(token) for token in sorted(retained)]))
    return records, graphs


def audit_graph_owners(records, expected_graphs):
    """Validate one/two live lane records; no inference from primary pool alone."""
    if type(expected_graphs) is not int or expected_graphs <= 0:
        raise ValueError('Explicit positive graph count required')
    if any(type(k) is not int for k in records) or set(records) not in ({0}, {0, 1}):
        raise ValueError('Explicit one/two lane records required')
    threads = set(); streams = set(); devices = set(); pools = set()
    counts = {}
    for lane, record in records.items():
        identity = record['identity']
        device = identity['device']
        if not isinstance(device, str) or not re.fullmatch(r'cuda:(?:0|[1-9][0-9]*)', device):
            raise ValueError('Explicit indexed CUDA device required')
        if identity['failed'] is not False:
            raise ValueError('Failed lane cannot pass ownership checks')
        thread, stream = identity['owner_thread_id'], identity['stream_id']
        if any(type(x) is not int or x <= 0 for x in (thread, stream)):
            raise ValueError('Live owner-thread and private stream IDs required')
        if thread in threads or (device, stream) in streams:
            raise ValueError('Cross-lane thread or replay stream sharing')
        threads.add(thread); streams.add((device, stream)); devices.add(device)
        rows = record['graphs']
        if (len(rows) != expected_graphs or any(type(r['index']) is not int for r in rows)
                or [r['index'] for r in rows] != list(range(expected_graphs))):
            raise ValueError('Missing, duplicate or reordered graph identity records')
        lane_pools = set()
        for row in rows:
            primary = _token(row['primary'])
            retained = [_token(token) for token in row['retained']]
            if not retained or len(set(retained)) != len(retained) or primary not in retained:
                raise ValueError('Primary/all-retained graph identity mismatch')
            lane_pools.update((device, token) for token in retained)
        if pools & lane_pools:
            raise ValueError('Cross-lane retained CUDA graph pool sharing')
        pools.update(lane_pools); counts[str(lane)] = len(lane_pools)
    if len(devices) != 1:
        raise ValueError('Serving comparison requires one shared CUDA device')
    return dict(graph_pool_identities_disjoint=True, retained_pool_counts=counts,
                scope='Reported retained pool and owner identities only; live tensor storage, actual launch streams, GPU overlap and exact results require separate audits.')
