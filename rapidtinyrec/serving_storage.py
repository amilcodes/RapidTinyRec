"""Untimed explicit tensor inventory for future owned serving runtimes.

Callers must retain all enumerated tensors while comparing lanes. These records
prove disjointness only for the supplied live storage, not inventory completeness,
CUDA graph-pool separation, stream ordering, or numerical equivalence.
"""

import re


def describe_tensors(named_tensors):
    """Describe whole untyped storages, including the backing capacity of views."""
    rows=[]
    for name, role, tensor in named_tensors:
        storage=tensor.untyped_storage()
        rows.append(dict(name=name,role=role,device=str(tensor.device),
                         address=storage.data_ptr(),storage_bytes=storage.nbytes(),
                         logical_bytes=tensor.numel()*tensor.element_size()))
    validate_inventory(rows)
    return rows


def validate_inventory(rows):
    if not rows or len({r['name'] for r in rows})!=len(rows):
        raise ValueError('Nonempty inventory with unique tensor names required')
    capacities={}
    for r in rows:
        if not all(isinstance(r[k],str) and r[k] for k in ('name','role','device')):
            raise ValueError('Named role and device required')
        if not re.fullmatch(r'cpu|cuda:(?:0|[1-9][0-9]*)',r['device']):
            raise ValueError('Canonical CPU or indexed CUDA device required')
        for key in ('address','storage_bytes','logical_bytes'):
            if type(r[key]) is not int or r[key]<0:
                raise ValueError('Nonnegative integer address and byte counts required')
        if r['storage_bytes'] and not r['address']:
            raise ValueError('Live storage cannot have a null address')
        if not r['storage_bytes'] and r['logical_bytes']:
            raise ValueError('Nonempty logical tensor needs backing storage')
        # Expanded/overlapping views may have more logical bytes than capacity.
        key=(r['device'],r['address'])
        if r['storage_bytes']:
            if key in capacities and capacities[key]!=r['storage_bytes']:
                raise ValueError('Inconsistent capacity for the same storage address')
            capacities[key]=r['storage_bytes']
    return capacities


def audit_lane_storage(inventories):
    """Reject any cross-lane overlap in each device address space."""
    if any(type(k) is not int for k in inventories) or set(inventories) not in ({0},{0,1}):
        raise ValueError('Require explicit one- or two-lane inventory')
    ranges={};summary={}
    for lane,rows in inventories.items():
        capacities=validate_inventory(rows)
        ranges[lane]=[(device,address,address+size) for (device,address),size in capacities.items()]
        # Union intervals for exact physical accounting even when external views
        # describe overlapping allocations with different base addresses.
        physical=0;device=None;end=0
        for dev,start,stop in sorted(ranges[lane]):
            if dev!=device:device=dev;end=0
            physical+=max(0,stop-max(start,end));end=max(end,stop)
        summary[str(lane)]=dict(tensors=len(rows),unique_storage_ranges=len(capacities),
                               logical_bytes=sum(r['logical_bytes'] for r in rows),
                               unique_storage_bytes=physical)
    if len(ranges)==2:
        for da,sa,ea in ranges[0]:
            for db,sb,eb in ranges[1]:
                if da==db and max(sa,sb)<min(ea,eb):
                    raise ValueError(f'Cross-lane storage overlap on {da}: {sa}:{ea} and {sb}:{eb}')
    return dict(schema='serving-storage-inventory-v1',lanes=summary,
                enumerated_storage_disjoint=True,
                scope='Supplied simultaneously live tensors only; runtime inventory completeness and CUDA graph/stream ownership require separate evidence.')
