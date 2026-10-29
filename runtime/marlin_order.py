# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in local port of the reviewed token-order workaround in vLLM PR 52532.

Source: https://github.com/vllm-project/vllm/pull/52532
The reviewed diff is frozen in run/method-preparation/source-review/.
This changes token ordering within each expert, never expert assignments,
weight precision, routing weights, or the padded token count. Its overhead
must be included in every matched runtime's performance measurements.
"""
import functools

import torch


def canonicalize(sorted_ids, expert_ids, padded_count, block_size, valid_count):
    block_experts=expert_ids.to(torch.int64)
    positions=torch.arange(expert_ids.numel(),device=expert_ids.device,dtype=torch.int64)
    active=positions*block_size<padded_count
    starts=torch.cat((torch.ones_like(block_experts[:1],dtype=torch.bool),
        (block_experts[1:]!=block_experts[:-1]) | (active[1:]!=active[:-1])))
    regions=torch.cumsum(starts.to(torch.int64),dim=0)-1
    slot_regions=torch.repeat_interleave(regions,block_size)[:sorted_ids.numel()]
    slots=torch.arange(sorted_ids.numel(),device=sorted_ids.device,dtype=torch.int64)
    token_keys=torch.where(slots<padded_count,sorted_ids.to(torch.int64),0)
    order=torch.argsort(slot_regions*(valid_count+1)+token_keys)
    return sorted_ids.index_select(0,order)


def install():
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe
    original=marlin_moe.moe_align_block_size
    if getattr(original,'moecbench_canonical_order',False):return

    @functools.wraps(original)
    def aligned(topk_ids,block_size,num_experts,*args,**kwargs):
        outputs=original(topk_ids,block_size,num_experts,*args,**kwargs)
        if topk_ids.shape[0]<=1:return outputs
        sorted_ids,expert_ids,padded_count,*extra=outputs
        return (canonicalize(sorted_ids,expert_ids,padded_count,block_size,topk_ids.numel()),
                expert_ids,padded_count,*extra)

    aligned.moecbench_canonical_order=True
    marlin_moe.moe_align_block_size=aligned
