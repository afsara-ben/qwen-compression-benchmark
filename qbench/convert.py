"""Resumable CPU NVFP4 conversion for original and REAP Qwen checkpoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

from .common import digest, read, sha, write


def quantize(weight, maximum):
    import torch
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils as ref
    if weight.ndim != 2 or weight.shape[1] % 16:
        raise ValueError('NVFP4 expects a matrix with K divisible by 16')
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    inverse_global = torch.tensor(2688., dtype=torch.float32) / maximum.clamp_min(1e-12)
    q, scales = ref.ref_nvfp4_quant(weight, inverse_global, 16)
    codes = torch.searchsorted(levels, q.abs().contiguous()).to(torch.uint8)
    codes |= (q < 0).to(torch.uint8) << 3
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    magnitudes = torch.stack((packed & 7, (packed >> 4) & 7), -1).reshape_as(q)
    signs = torch.stack((packed & 8, (packed >> 4) & 8), -1).reshape_as(q)
    if not torch.equal(q, levels[magnitudes.long()] * torch.where(signs != 0, -1., 1.)):
        raise ValueError('FP4 packing differs from the pinned reference')
    scales = scales.to(torch.float8_e4m3fn).contiguous()
    if not torch.isfinite(scales.float()).all():
        raise ValueError('Nonfinite block scales')
    return packed, scales, inverse_global.reciprocal().reshape(1)


def self_test():
    import torch
    torch.manual_seed(17)
    for value in (torch.randn(17, 128).bfloat16(), torch.zeros(4, 32).bfloat16(),
                  torch.tensor([0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 5., 6., -6.]).repeat(2, 1).bfloat16()):
        q, s, g = quantize(value, value.abs().max().float())
        assert q.shape == (value.shape[0], value.shape[1] // 2)
        assert s.shape == (value.shape[0], value.shape[1] // 16)
        assert torch.isfinite(g).all()


def convert(source, output, threads=4):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils as ref
    torch.set_num_threads(threads)
    self_test()
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = read(source / 'config.json')
    hybrid = 'text_config' in cfg
    text_cfg = cfg.get('text_config', cfg)
    experts, layers = text_cfg['num_experts'], text_cfg['num_hidden_layers']
    hidden, intermediate = text_cfg['hidden_size'], text_cfg['moe_intermediate_size']
    if text_cfg['num_experts_per_tok'] != 8:
        raise ValueError('This experiment requires top-8 routing')
    contract = dict(source_config=sha(source / 'config.json'), source_index=sha(source / 'model.safetensors.index.json'),
                    converter=sha(__file__), reference=sha(ref.__file__), recipe='W4A16_NVFP4-max-group16-v1')
    identity = digest(contract)
    completed = output / 'artifact.json'
    if completed.exists():
        artifact = read(completed)
        if artifact['conversion_fingerprint'] != identity:
            raise ValueError('Existing NVFP4 artifact has a different source or recipe')
        for f in artifact['files']:
            if sha(output / f['path']) != f['sha256']:
                raise ValueError('NVFP4 artifact checksum mismatch: ' + f['path'])
        return artifact
    cfg['quantization_config'] = dict(quant_method='modelopt', quant_algo='W4A16_NVFP4', group_size=16,
        ignore=['lm_head', 'visual', 'mlp.gate', 'mlp.shared_expert_gate'] if hybrid else ['lm_head'])
    assert ModelOptNvFp4Config.from_config(cfg['quantization_config']).quant_method == 'W4A16_NVFP4'
    for p in source.iterdir():
        if p.is_file() and not p.name.startswith('model') and p.name != 'config.json' and p.suffix in ('.json', '.jinja', '.txt', '.model'):
            shutil.copy2(p, output / p.name)
    write(output / 'config.json', cfg)
    index = read(source / 'model.safetensors.index.json')['weight_map']
    handles = {name: safe_open(str(source / name), framework='pt', device='cpu') for name in set(index.values())}
    def tensor(name):
        return handles[index[name]].get_tensor(name)
    layer_root = 'model.language_model.layers.' if hybrid else 'model.layers.'
    began = time.monotonic()
    files, weight_map, total, quantized = [], {}, 0, 0
    for number in range(1, layers + 2):
        filename = f'model-{number:05d}-of-{layers+1:05d}.safetensors'
        shard, receipt = output / filename, output / (filename + '.json')
        cached = read(receipt) if receipt.exists() else {}
        if cached.get('conversion_fingerprint') == identity and shard.exists() and sha(shard) == cached.get('sha256'):
            metadata = cached
        else:
            state, count = {}, 0
            def put(stem, weight, maximum, expert=False):
                nonlocal count
                q, scale, global_scale = quantize(weight, maximum)
                state[stem + '.weight'] = q
                state[stem + '.weight_scale'] = scale
                state[stem + '.weight_scale_2'] = global_scale
                if expert:
                    state[stem + '.input_scale'] = torch.ones(1)
                count += 1
            if number == layers + 1:
                state = {k: tensor(k).clone() for k in index if not k.startswith(layer_root)}
            else:
                layer = number - 1
                prefix, maxima = layer_root + str(layer) + '.', {}
                if hybrid:
                    gate_up, down = tensor(prefix+'mlp.experts.gate_up_proj'), tensor(prefix+'mlp.experts.down_proj')
                    assert gate_up.shape == (experts, 2*intermediate, hidden)
                    assert down.shape == (experts, hidden, intermediate)
                    for expert in range(experts):
                        stem = prefix+f'mlp.experts.{expert}.'
                        maximum = gate_up[expert].abs().max().float()
                        put(stem+'gate_proj', gate_up[expert, :intermediate], maximum, True)
                        put(stem+'up_proj', gate_up[expert, intermediate:], maximum, True)
                        put(stem+'down_proj', down[expert], down[expert].abs().max().float(), True)
                    del gate_up, down
                    groups = [['mlp.shared_expert.gate_proj.weight', 'mlp.shared_expert.up_proj.weight']]
                    if text_cfg['layer_types'][layer] == 'full_attention':
                        groups += [[f'self_attn.{x}_proj.weight' for x in ('q', 'k', 'v')]]
                    else:
                        groups += [['linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_z.weight'],
                                   ['linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight']]
                else:
                    groups = [[f'mlp.experts.{e}.{x}_proj.weight' for x in ('gate', 'up')] for e in range(experts)]
                    groups += [[f'self_attn.{x}_proj.weight' for x in ('q', 'k', 'v')]]
                for group in groups:
                    maximum = torch.stack([tensor(prefix+k).abs().max().float() for k in group]).max()
                    for k in group:
                        maxima[prefix+k] = maximum
                for name in sorted(k for k in index if k.startswith(prefix) and (not hybrid or '.mlp.experts.' not in k)):
                    value = tensor(name)
                    excluded = hybrid and name in (prefix+'mlp.gate.weight', prefix+'mlp.shared_expert_gate.weight')
                    if name.endswith('.weight') and value.ndim == 2 and not excluded:
                        maximum = maxima.get(name)
                        put(name[:-7], value, value.abs().max().float() if maximum is None else maximum,
                            '.mlp.experts.' in name)
                    else:
                        state[name] = value.clone()
            temporary = shard.with_suffix('.partial')
            save_file(state, str(temporary), metadata={'format': 'pt'})
            temporary.replace(shard)
            metadata = dict(path=filename, sha256=sha(shard), bytes=shard.stat().st_size,
                            tensor_bytes=sum(x.numel()*x.element_size() for x in state.values()),
                            keys=list(state), quantized_tensors=count, conversion_fingerprint=identity)
            write(receipt, metadata)
            del state
        files.append({k:metadata[k] for k in ('path', 'sha256', 'bytes')})
        weight_map.update({k:filename for k in metadata['keys']})
        total += metadata['tensor_bytes']; quantized += metadata['quantized_tensors']
        write(output / 'status.json', dict(status='preparing', shards_completed=number, shards_total=layers+1, elapsed_s=time.monotonic()-began))
        print(f'NVFP4 {number}/{layers+1}: {filename}', flush=True)
    expected = layers*(3*experts+5) if not hybrid else layers*3*experts + sum(7 if t=='full_attention' else 8 for t in text_cfg['layer_types'])
    assert quantized == expected, (quantized, expected)
    write(output / 'model.safetensors.index.json', dict(metadata={'total_size':total}, weight_map=weight_map))
    artifact = dict(status='verified_on_cpu', quantization='W4A16_NVFP4', activation='bfloat16',
                    conversion_fingerprint=identity, contract=contract, files=files,
                    expert_count=experts, layer_count=layers, experts_per_token=8,
                    quantized_tensor_count=quantized, weight_tensor_bytes=total,
                    calibration='weight maxima only; no activation or evaluation data',
                    elapsed_s=time.monotonic()-began)
    write(completed, artifact)
    write(output / 'status.json', dict(status='prepared', shards_completed=layers+1))
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.self_test:
        self_test()
        print('NVFP4 reference/packing/scales self-test passed')
    else:
        convert(args.source, args.output, args.threads)
