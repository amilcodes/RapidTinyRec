"""Serialized setup, ownership gates and global patch lifetime for serving.

The initializer must allocate/warm each complete private lane on its owner
thread. A separate mandatory setup auditor must validate numerical and retained
output evidence. This module neither loads a checkpoint nor supplies that audit.
"""
from contextlib import contextmanager

from .serving_lanes import LaneServer
from .serving_graph_ownership import audit_graph_owners
from .serving_storage import audit_lane_storage


@contextmanager
def prepared_service(initialize, audit_setup, *, lane_count, expected_graphs):
    """Yield (server, preparation report) only after all preparation gates pass.

    All initialize callbacks finish before the first compiler freeze. Attention
    folding remains installed until every owner thread closes, even on failure.
    The auditor receives JSON metadata while snapshots keep owners strongly live.
    No timing/throughput or actual GPU-overlap conclusion is produced here.
    """
    from .folded_attention import fold_decode_attention
    with fold_decode_attention(True):
        with LaneServer(initialize, lambda lane, user: lane.run(user),
                        lambda lane: lane.freeze(), lambda lane: lane.close(),
                        lane_count=lane_count) as server:
            server.freeze()
            snapshots = server.visit(lambda lane: lane.ownership_snapshot(expected_graphs))
            try:
                records = {index: snapshot['record'] for index, snapshot in enumerate(snapshots)}
                graph_report = audit_graph_owners(records, expected_graphs)
                storage_report = audit_lane_storage({index: row['storage'] for index, row in records.items()})
                setup_report = audit_setup(records)
                if not isinstance(setup_report, dict) or setup_report.get('passed') is not True:
                    raise ValueError('Independent setup audit did not pass')
            finally:
                # Drop inspection-only references before owner-thread release or
                # shutdown, also on audit failure. Lanes still own their graphs.
                snapshots = None
            server.visit(lambda lane: lane.release_retained())
            yield server, dict(graph_ownership=graph_report, storage_ownership=storage_report,
                               setup_audit=setup_report, owners=records)
