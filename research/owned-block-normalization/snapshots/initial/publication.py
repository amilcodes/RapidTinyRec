"""Exclusive incremental publication with before/after component guards.

CLI remains disabled. Allocation identity/deadline/log closure is a separate
mandatory outer gate; this module never marks execution certified.
"""
import json
from pathlib import Path
import traceback
from native_binding import verify_native_entry,collect_runtime,validate_runtime
from loaded_binding import verify_loaded
from produce_cases import produce_cases
from audit_evidence import audit_verified_case,audit_verified_inventory
from audit_inputs import verify_generated_inputs


def save(path,record):
    with Path(path).open('x') as f:
        json.dump(record,f,indent=2,allow_nan=False);f.write('\n')


def guard_snapshot(root,expected):
    loaded=verify_loaded(root,expected)
    native=verify_native_entry()
    runtime=collect_runtime();validate_runtime(runtime)
    return dict(loaded=loaded,native=native,runtime=runtime)


def run_guarded(root,expected,output):
    output=Path(output);output.mkdir(exist_ok=False)
    report=dict(schema='request-block-guarded-publication-v1',compute_complete=False,
                records_passed=False,execution_certified=False,submission_ready=False,published_cases=0)
    before=None
    try:
        before=guard_snapshot(root,expected);save(output/'before.json',before)
        def publish(record):
            index=report['published_cases']
            # Preserve a malformed/negative case before checking it. Never
            # replace it with an inferred success or silently omit a stratum.
            save(output/f'case-{index:02d}.json',record);report['published_cases']+=1
            audit_verified_case(record)
        records=produce_cases(case_sink=publish)
        if len(records)!=report['published_cases']:raise ValueError('Incremental case inventory differs')
        inventory=audit_verified_inventory(records);save(output/'inventory.json',inventory)
        seeds=verify_generated_inputs(records);save(output/'inputs.json',seeds)
        report['records_passed']=inventory['records_passed'] and seeds['seed_generation_independently_verified']
        report['compute_complete']=True
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc(),compute_complete=False,records_passed=False)
    finally:
        if before is not None:
            try:
                after=guard_snapshot(root,expected);save(output/'after.json',after)
                if after!=before:raise ValueError('Source/native/runtime changed during producer')
                report['before_after_guards_match']=True
            except BaseException as exc:
                report.update(after_guard_error=repr(exc),after_guard_traceback=traceback.format_exc(),
                              before_after_guards_match=False,compute_complete=False,records_passed=False)
        save(output/'result.json',report)
    return report


if __name__=='__main__':
    raise SystemExit('Disabled: allocation identity/deadline/log closure/config/readiness pending')
