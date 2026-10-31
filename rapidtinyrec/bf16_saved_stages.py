"""Validate saved BF16 generation/recovery deadlines, without a live-job claim."""
import math
from .bf16_protocol import PROTOCOL_SHA256
from .bf16_runner import allowance


def number(value):
    assert type(value) in (int,float) and math.isfinite(value)
    return value


def validate_saved_stages(start, generation, recovery, record):
    """Require both prerequisite stages to have finished inside the original budget.

    The original audit may have failed. Its outcome is intentionally neither
    changed nor certified here. A recovery runner needs a separate live identity,
    deadline, immutable raw inventory, source verification and wrapper audit.
    """
    assert start['schema']=='bf16-pilot-work-v1'
    assert start['job_id']==record['job_id']
    assert start['protocol_sha256']==PROTOCOL_SHA256
    origin=number(start['started_monotonic'])
    deadline=number(start['deadline_monotonic'])
    assert deadline==origin+1620
    previous=origin
    for expected,entry in zip(('generation','recovery'),(generation,recovery)):
        assert entry['name']==expected
        began=number(entry['started_monotonic']);ended=number(entry['ended_monotonic'])
        budget=number(entry['timeout_s'])
        assert previous<=began<ended<=deadline
        assert budget>0 and budget==allowance(expected,origin,began)
        assert ended-began<=budget and entry['within_deadline'] is True
        assert entry['child']['completed_normally'] is True
        previous=ended
    return dict(schema='bf16-saved-prerequisite-stages-v1',original_job_id=record['job_id'],
        original_started_monotonic=origin,original_deadline_monotonic=deadline,
        generation_ended_monotonic=generation['ended_monotonic'],
        recovery_ended_monotonic=recovery['ended_monotonic'],
        original_audit_certified=False)
