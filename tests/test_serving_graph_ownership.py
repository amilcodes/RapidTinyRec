import gc
import types
import unittest
import weakref

from rapidtinyrec.serving_graph_ownership import snapshot_graphs, audit_graph_owners


class Graph:
    def __init__(self, primary, retained=None):
        self.primary = primary; self.retained = retained or [primary]
    def pool(self): return self.primary
    def pools(self): return self.retained


def owner(thread, stream, primary, extra):
    graph = Graph(primary, [primary, extra])
    replay = types.SimpleNamespace(frozen=True, entries={'shape': (None, graph, None)})
    rows, refs = snapshot_graphs(replay, 1)
    return dict(identity=dict(owner_thread_id=thread, stream_id=stream, device='cuda:0',
                              failed=False), graphs=rows), refs


class GraphOwnershipTests(unittest.TestCase):
    def records(self):
        a, _ = owner(10, 11, (1, 0), (0, 10))
        b, _ = owner(20, 21, (2, 0), (0, 20))
        return {0: a, 1: b}

    def test_all_retained_pools_are_compared_not_only_primary(self):
        rows = self.records()
        result = audit_graph_owners(rows, 1)
        self.assertEqual(result['retained_pool_counts'], {'0': 2, '1': 2})
        rows[1]['graphs'][0]['retained'][0] = [0, 10]
        with self.assertRaisesRegex(ValueError, 'Cross-lane retained'):
            audit_graph_owners(rows, 1)

    def test_full_token_matters_and_json_roundtrip_is_supported(self):
        import json
        a, _ = owner(10, 11, (1, 0), (0, 1))
        self.assertEqual(audit_graph_owners({0: json.loads(json.dumps(a))}, 1)
                         ['retained_pool_counts'], {'0': 2})

    def test_invalid_or_missing_pool_evidence_fails(self):
        for token in ((0, 0), (1, 2), (-1, 0), (True, 0), (1,), 'pool'):
            rows = self.records(); rows[0]['graphs'][0]['primary'] = token
            with self.assertRaises(ValueError): audit_graph_owners(rows, 1)
        for retained in ([], [[1, 0], [1, 0]], [[99, 0]]):
            rows = self.records(); rows[0]['graphs'][0]['retained'] = retained
            with self.assertRaises(ValueError): audit_graph_owners(rows, 1)

    def test_owner_device_stream_and_graph_cardinality_are_enforced(self):
        for field, value in [('owner_thread_id', 10), ('stream_id', 11),
                             ('device', 'cuda:1'), ('device', 'cuda'),
                             ('failed', True), ('stream_id', 0)]:
            rows = self.records(); rows[1]['identity'][field] = value
            with self.assertRaises(ValueError): audit_graph_owners(rows, 1)
        for graphs in ([], [dict(index=False, primary=[1, 0], retained=[[1, 0]])]):
            rows = self.records(); rows[0]['graphs'] = graphs
            with self.assertRaises(ValueError): audit_graph_owners(rows, 1)

    def test_snapshot_keeps_graphs_alive_and_refuses_uncaptured_or_duplicate_graphs(self):
        graph = Graph((1, 0)); ref = weakref.ref(graph)
        replay = types.SimpleNamespace(frozen=True, entries={'a': (None, graph, None)})
        rows, refs = snapshot_graphs(replay, 1)
        del graph, replay; gc.collect(); self.assertIsNotNone(ref())
        del refs; gc.collect(); self.assertIsNone(ref())
        graph = Graph((2, 0))
        replay = types.SimpleNamespace(frozen=False, entries={'a': (None, graph, None)})
        with self.assertRaises(ValueError): snapshot_graphs(replay, 1)
        replay.frozen = True; replay.entries['b'] = (None, graph, None)
        with self.assertRaises(ValueError): snapshot_graphs(replay, 2)


if __name__ == '__main__': unittest.main()
