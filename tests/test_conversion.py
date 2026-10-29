"""Small real tensor conversions: no production model or GPU required."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from qbench.common import read, write

try:
    import torch
    from safetensors.torch import save_file
    import vllm
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Run with the installed benchmark environment for tensor conversion tests')
class Conversion(unittest.TestCase):
    def checkpoint(self,root,hybrid,experts):
        torch.manual_seed(17)
        layers=2 if hybrid else 1
        text=dict(num_experts=experts,num_experts_per_tok=8,num_hidden_layers=layers,hidden_size=32,moe_intermediate_size=16)
        if hybrid:text['layer_types']=['full_attention','linear_attention']
        cfg={'text_config':text} if hybrid else text
        root.mkdir();write(root/'config.json',cfg)
        state={}
        def add(k,shape):state[k]=torch.randn(*shape).bfloat16()
        for i in range(layers):
            p=('model.language_model.layers.' if hybrid else 'model.layers.')+str(i)+'.'
            if hybrid:
                add(p+'mlp.experts.gate_up_proj',(experts,32,32));add(p+'mlp.experts.down_proj',(experts,32,16))
                for k,shape in [('gate',(experts,32)),('shared_expert_gate',(1,32)),('shared_expert.gate_proj',(16,32)),('shared_expert.up_proj',(16,32)),('shared_expert.down_proj',(32,16))]:add(p+'mlp.'+k+'.weight',shape)
            else:
                for e in range(experts):
                    for k,shape in [('gate_proj',(16,32)),('up_proj',(16,32)),('down_proj',(32,16))]:add(p+f'mlp.experts.{e}.'+k+'.weight',shape)
                add(p+'mlp.gate.weight',(experts,32))
            if i==0:
                for k in ['q','k','v','o']:add(p+'self_attn.'+k+'_proj.weight',(32,32))
            else:
                for k in ['in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj']:add(p+'linear_attn.'+k+'.weight',(32,32))
                add(p+'linear_attn.conv1d.weight',(32,1,4))
            add(p+'input_layernorm.weight',(32,))
        add('lm_head.weight',(16,32));add('model.embed_tokens.weight',(16,32))
        save_file(state,str(root/'model.safetensors'))
        write(root/'model.safetensors.index.json',{'weight_map':{k:'model.safetensors' for k in state}})

    def test_both_architectures_and_expert_counts(self):
        from qbench.convert import convert
        for hybrid in (False,True):
            for experts in (8,16):
                with tempfile.TemporaryDirectory() as tmp:
                    src,out=Path(tmp)/'source',Path(tmp)/'output'
                    self.checkpoint(src,hybrid,experts)
                    artifact=convert(src,out,threads=2)
                    self.assertEqual(artifact['expert_count'],experts)
                    first=out/artifact['files'][0]['path'];stamp=first.stat().st_mtime_ns
                    convert(src,out,threads=2)
                    self.assertEqual(first.stat().st_mtime_ns,stamp)

    def test_interrupted_conversion_resumes_completed_shards(self):
        import qbench.convert as converter
        with tempfile.TemporaryDirectory() as tmp:
            src,out=Path(tmp)/'source',Path(tmp)/'output'
            self.checkpoint(src,True,8)
            original=converter.write
            def interrupted(path,data):
                original(path,data)
                if Path(path).name=='status.json' and data.get('shards_completed')==1:
                    raise InterruptedError('simulated interruption')
            with patch.object(converter,'write',interrupted),self.assertRaises(InterruptedError):
                converter.convert(src,out,threads=2)
            first=out/'model-00001-of-00003.safetensors';stamp=first.stat().st_mtime_ns
            converter.convert(src,out,threads=2)
            self.assertEqual(first.stat().st_mtime_ns,stamp)


if __name__=='__main__':unittest.main()
