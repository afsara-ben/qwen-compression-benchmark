"""Compatibility for RangerX REAP's misplaced vision-tower tensor prefix.

The supplied checkpoint stores vision weights under model.language_model.visual
instead of model.visual. Normalize their names before vLLM's standard mapper;
the language-model-only server then handles the vision tower as usual. Text
weight names, tensor contents, precision, and expert selection are unchanged.
This extension is selected only for the supplied Qwen3.6 REAP checkpoint.
"""
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper

from quality_audit import QualityAudit

REPAIR = {
    'checkpoint': 'RangerX/Qwen3.6-35B-REAP-Pruned-ratio-0.2',
    'revision': '3df0df7fcc715b6b1e59d4a61bebf101dad2d676',
    'source_prefix': 'model.language_model.visual.',
    'mapped_prefix': 'visual.',
    'text_weights_unchanged': True,
}

# Order matters: handle the specific vision prefix before the existing general
# model.language_model prefix. The installed mapper preserves all other rules.
Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper = (
    WeightsMapper(orig_to_new_prefix={REPAIR['source_prefix']: REPAIR['mapped_prefix']})
    | Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper
)


class QualityReapAudit(QualityAudit):
    def quality_audit(self, reset_peak=False):
        result = super().quality_audit(reset_peak=reset_peak)
        result['checkpoint_loader_compatibility'] = REPAIR
        return result
