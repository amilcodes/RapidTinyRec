"""Serialized setup, ownership gates and global patch lifetime for serving.

The initializer must allocate/warm each complete private lane on its owner
thread. A separate mandatory setup auditor must validate numerical and retained
output evidence. This module neither loads a checkpoint nor supplies that audit.
"""
from contextlib import contextmanager
import copy
import threading
import time

from .serving_lanes import LaneServer
from .serving_graph_ownership import audit_graph_owners
from .serving_storage import audit_lane_storage


@contextmanager
def prepared_service(initialize, audit_setup, *, lane_count, expected_graphs, lifecycle=None):
    """Yield (server, preparation report) only after all preparation gates pass.

    All initialize callbacks finish before the first compiler freeze. Attention
    folding remains installed until every owner thread closes, even on failure.
    The auditor receives JSON metadata while snapshots keep owners strongly live.
    No timing/throughput or actual GPU-overlap conclusion is produced here.
    """
    # The caller may retain this journal and save it on failure or after close.
    # No file writes or observer callbacks are introduced in owner execution.
    if lifecycle is None:
        lifecycle = []
    if not isinstance(lifecycle, list) or lifecycle:
        raise ValueError('A fresh empty lifecycle journal is required')
    lane_indices = {}
    def timed(operation, lane, action):
        row = dict(operation=operation, lane=lane, thread_id=threading.get_ident(),
                   started_ns=time.perf_counter_ns(), finished_ns=None, success=False, error=None)
        lifecycle.append(row)
        try:
            result = action()
            row['success'] = True
            return result
        except BaseException as error:
            row['error'] = repr(error)
            raise
        finally:
            row['finished_ns'] = time.perf_counter_ns()
    def initialize_owned(index):
        state = timed('initialize', index, lambda: initialize(index))
        lane_indices[id(state)] = index
        return state
    def owned(operation, state, action):
        return timed(operation, lane_indices[id(state)], action)
    from .folded_attention import fold_decode_attention
    with fold_decode_attention(True):
        timed('patch_installed', None, lambda: None)
        with LaneServer(initialize_owned, lambda lane, user: lane.run(user),
                        lambda lane: owned('freeze', lane, lane.freeze),
                        lambda lane: owned('close', lane, lane.close),
                        lane_count=lane_count) as server:
            server.freeze()
            snapshots = server.visit(lambda lane: owned('snapshot', lane, lambda: lane.ownership_snapshot(expected_graphs)))
            try:
                records = {index: snapshot['record'] for index, snapshot in enumerate(snapshots)}
                def ownership():
                    return (audit_graph_owners(records, expected_graphs),
                            audit_lane_storage({index: row['storage'] for index, row in records.items()}))
                graph_report, storage_report = timed('ownership_audit', None, ownership)
                def checked_setup():
                    report = audit_setup(records)
                    if not isinstance(report, dict) or report.get('passed') is not True:
                        raise ValueError('Independent setup audit did not pass')
                    return report
                setup_report = timed('setup_audit', None, checked_setup)
            finally:
                # Drop inspection-only references before owner-thread release or
                # shutdown, also on audit failure. Lanes still own their graphs.
                snapshots = None
            server.visit(lambda lane: owned('release_retained', lane, lane.release_retained))
            timed('admitted', None, lambda: None)
            yield server, dict(graph_ownership=graph_report, storage_ownership=storage_report,
                               setup_audit=setup_report, owners=records, admission_events=copy.deepcopy(lifecycle))
