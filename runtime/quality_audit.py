"""Read-only vLLM worker inspection for the quality experiment."""


class QualityAudit:
    def quality_audit(self, reset_peak=False):
        import os
        import torch
        torch.cuda.synchronize()
        model = self.model_runner.get_model()
        weights, buffers, caches = {}, {}, {}
        dtypes, experts, methods, scales = {}, [], {}, {}
        nvfp4_modules = []
        def collect(tensor, out):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                s = tensor.untyped_storage()
                out[s.data_ptr()] = s.nbytes()
            elif isinstance(tensor, dict):
                for t in tensor.values(): collect(t, out)
            elif isinstance(tensor, (tuple, list)):
                for t in tensor: collect(t, out)
        bad = []
        for name, param in model.named_parameters():
            if param.device.type != 'cuda': bad.append(name)
            collect(param, weights)
            dtypes[str(param.dtype)] = dtypes.get(str(param.dtype), 0) + param.numel()
        for name, value in model.named_buffers(): collect(value, buffers)
        for name, module in model.named_modules():
            method = getattr(module, 'quant_method', None)
            if method is not None:
                key = type(method).__name__
                methods[key] = methods.get(key, 0) + 1
                if key == 'ModelOptNvFp4FusedMoE':
                    nvfp4_modules.append(dict(name=name, method=key, use_a16=method.use_a16,
                                              backend=str(method.nvfp4_backend)))
                elif 'NvFp4' in type(getattr(method, 'kernel', None)).__name__:
                    nvfp4_modules.append(dict(name=name, method=key,
                                              backend=type(method.kernel).__name__))
            if hasattr(module, 'local_num_experts'):
                experts.append(dict(name=name, count=int(module.local_num_experts)))
            for attr in ['_k_scale', '_v_scale']:
                value = getattr(module, attr, None)
                if isinstance(value, torch.Tensor) and value.numel() <= 16:
                    scales[name + '.' + attr] = value.detach().cpu().tolist()
        collect(getattr(self.model_runner, 'kv_caches', []), caches)
        cache = self.model_runner.kv_cache_config
        groups = [dict(spec=type(g.kv_cache_spec).__name__,
                       block_size=g.kv_cache_spec.block_size,
                       layer_count=len(g.layer_names),
                       dtype=str(getattr(g.kv_cache_spec, 'dtype', None)),
                       state_dtypes=[str(x) for x in getattr(g.kv_cache_spec, 'dtypes', ())],
                       page_bytes=getattr(g.kv_cache_spec, 'page_size_bytes', None))
                  for g in cache.kv_cache_groups]
        from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
        concurrency = get_max_concurrency_for_kv_cache_config(self.vllm_config, cache)
        required_blocks = round(cache.num_blocks / concurrency)
        bytes_per_block = sum(caches.values()) // cache.num_blocks
        result = dict(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                      non_cuda_parameters=bad, parameter_dtypes=dtypes,
                      weight_storage_bytes=sum(weights.values()),
                      model_buffer_bytes=sum(v for k,v in buffers.items() if k not in weights),
                      cache_storage_bytes=sum(caches.values()), cache_groups=groups,
                      cache_blocks=cache.num_blocks, kv_dtype=self.cache_config.cache_dtype,
                      max_context_concurrency=concurrency, minimum_context_blocks=required_blocks,
                      pool_bytes_per_block=bytes_per_block,
                      minimum_context_cache_bytes=required_blocks*bytes_per_block,
                      experts=experts, quantization_methods=methods, kv_scales=scales,
                      nvfp4_modules=nvfp4_modules,
                      marlin_canonical_order=os.environ.get('MOECBENCH_MARLIN_CANONICAL_ORDER') == '1',
                      allocated_bytes=torch.cuda.memory_allocated(),
                      reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if reset_peak: torch.cuda.reset_peak_memory_stats()
        return result
