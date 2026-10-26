"""CPU audit of setup/freeze/admission clocks emitted on the executing threads."""


def audit_admission(events, owners, stages, resources_verified_ns, phase_started_ns):
    lanes = len(owners)
    if set(owners) != set(range(lanes)) or lanes not in (1, 2) or len(stages) != lanes:
        raise ValueError('Exact admission lane inventory required')
    expected = [('patch_installed', None)]
    for operation in ('initialize', 'freeze', 'snapshot'):
        expected.extend((operation, i) for i in range(lanes))
    expected += [('ownership_audit', None), ('setup_audit', None)]
    expected.extend(('release_retained', i) for i in range(lanes))
    expected += [('admitted', None)]
    if len(events) != len(expected):
        raise ValueError('Complete ordered admission event set required')
    previous = resources_verified_ns
    if type(previous) is not int or previous < 0 or type(phase_started_ns) is not int:
        raise ValueError('Integer resource/phase boundary clocks required')
    controller = None; initialized = []; frozen = []
    for row, (operation, lane) in zip(events, expected):
        if row['operation'] != operation or row['lane'] != lane or (lane is not None and type(row['lane']) is not int):
            raise ValueError('Admission order differs; all initialization must precede freeze')
        start, finish, thread = (row[k] for k in ('started_ns', 'finished_ns', 'thread_id'))
        if any(type(v) is not int or v <= 0 for v in (start, finish, thread)) or not previous <= start <= finish < phase_started_ns:
            raise ValueError('Admission clocks overlap or escape resource/phase boundaries')
        if row['success'] is not True or row['error'] is not None:
            raise ValueError('Failed lifecycle stage cannot admit requests')
        if lane is None:
            if controller is None: controller = thread
            if thread != controller or thread in {r['identity']['owner_thread_id'] for r in owners.values()}:
                raise ValueError('Admission controller identity changed or aliases an owner')
        elif thread != owners[lane]['identity']['owner_thread_id']:
            raise ValueError('Lifecycle work escaped its persistent owner thread')
        if operation == 'initialize':
            stage = stages[lane]
            if not start <= stage['allocation_started_ns'] <= stage['warmup_finished_ns'] <= finish:
                raise ValueError('Allocation/warmup escaped initialization event')
            initialized.append(finish)
        if operation == 'freeze': frozen.append(start)
        previous = finish
    if max(initialized) > min(frozen):
        raise ValueError('Compiler freeze preceded completion of all lane setup')
    return dict(schema='serving-admission-audit-v1', passed=True, lanes=lanes,
                all_setup_finished_ns=max(initialized), first_freeze_started_ns=min(frozen),
                admitted_ns=events[-1]['finished_ns'], controller_thread_id=controller,
                note='Recorded setup/freeze/admission ordering only; source/allocation/deadline and shutdown binding remain separate.')
