"""Untimed phase lifecycle for the future QKV producer; no generation entrypoint.

Outer producer must install original BF16 policy and folded attention. Enter
this scope BEFORE constructing compiler/graphs; close it after profiling.
"""
import contextlib
import time
import torch
from .packed_qkv_context import packed_qkv_bf16
from .qkv_inventory import snapshot, packing_evidence
from .qkv_inventory_schema import expected_schema, validate_snapshot, validate_unchanged, validate_packing

ARMS = ('original_bf16','packed_qkv_bf16','packed_qkv_bf16','original_bf16')
STAGES = ('after_setup','after_measurement','after_profiles')


def synchronize(model):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory(model):
    device = next(model.parameters()).device
    if device.type != 'cuda':
        return {'device':str(device), 'cuda_memory_measured':False}
    return dict(device=str(device), cuda_memory_measured=True,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device))


class PhaseState:
    def __init__(self, phase, model, inputs, masks, tables, config, users, constraints, root_prefix):
        if type(phase) is not int or phase not in range(4):
            raise ValueError('Unknown phase')
        self.phase=phase;self.arm=ARMS[phase]
        self.model=model;self.inputs=inputs;self.masks=masks;self.tables=tables
        self.original_schema=expected_schema(config,users,constraints,root_prefix)
        self.schema=expected_schema(config,users,constraints,root_prefix,packed=self.arm=='packed_qkv_bf16')
        self.config=config;self.records={};self.active=False;self.entered=False;self.complete=False

    def observe(self):
        return snapshot(self.model,self.inputs,self.masks,self.tables)

    def checkpoint(self, stage):
        if not self.active or len(self.records)>=len(STAGES) or stage!=STAGES[len(self.records)]:
            raise ValueError('Phase checkpoints missing, duplicated or out of order')
        synchronize(self.model)
        current=self.observe()
        validate_unchanged(self.installed,current,self.schema)
        self.records[stage]=current
        return current

    @contextlib.contextmanager
    def install(self):
        if self.entered:
            raise ValueError('Phase state cannot be reused')
        self.entered=True
        original=self.observe();validate_snapshot(original,self.original_schema)
        self.before_packing=original
        context=packed_qkv_bf16(self.model) if self.arm=='packed_qkv_bf16' else contextlib.nullcontext()
        synchronize(self.model);before_memory=memory(self.model);started=time.perf_counter()
        try:
            with context:
                synchronize(self.model)
                self.setup=dict(packing_install_s=time.perf_counter()-started,
                                before_memory=before_memory,after_memory=memory(self.model))
                self.installed=self.observe();validate_snapshot(self.installed,self.schema)
                if self.arm=='packed_qkv_bf16':
                    self.packing=packing_evidence(self.model)
                    validate_packing(self.packing,original,self.installed,self.original_schema,self.schema,self.config)
                else:
                    self.packing=None;validate_unchanged(original,self.installed,self.original_schema)
                self.active=True
                try:
                    yield self
                    if tuple(self.records)!=STAGES:
                        raise ValueError('Missing phase closure checkpoints')
                    # Check again on exit so edits after the last checkpoint cannot pass.
                    validate_unchanged(self.installed,self.observe(),self.schema)
                finally:
                    self.active=False
            validate_unchanged(original,self.observe(),self.original_schema)
            self.complete=True
        finally:
            self.active=False
