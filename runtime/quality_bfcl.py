#!/usr/bin/env python3
"""Pinned official BFCL v3 data and scoring, using native Qwen chat templates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='2',
                  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from overrides import override
from transformers import AutoTokenizer
from bfcl_eval.constants.category_mapping import VERSION_PREFIX,TEST_COLLECTION_MAPPING
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval._llm_response_generation import get_involved_test_entries,process_multi_turn_test_case
from bfcl_eval.eval_checker import eval_runner
from quality_server import ROOT,paths,server_context_capacity
from qbench.common import recover_jsonl,write as atomic_json


def inference_failed(record):
    return 'traceback' in record or str(record.get('result','')).startswith('Error during inference:')


def maximum_input_tokens(value):
    if isinstance(value,list):return max((maximum_input_tokens(x) for x in value),default=0)
    return int(value or 0)


def needs_capacity_retry(record,current_capacity):
    previous_capacity=record.get('study_server_context_capacity_tokens',32768)
    return current_capacity>previous_capacity and maximum_input_tokens(record.get('input_token_count'))+4096>previous_capacity


class StudyHandler(QwenFCHandler):
    gpu = 0
    tokenizer_path = ''
    journal = None
    tokenizer_shared = None
    server_capacity = None
    def __init__(self,model_name,temperature):
        super().__init__(model_name,temperature)
        self.tokenizer = self.tokenizer_shared
        self.model_path_or_id = 'quality-model'
        self.max_context_length = self.server_capacity

    @override
    def _format_prompt(self,messages,function):
        tools=[{'type':'function','function':f} for f in function]
        return self.tokenizer.apply_chat_template(messages,tools=tools or None,
            tokenize=False,add_generation_prompt=True,enable_thinking=False)

    @override
    def _parse_query_response_prompting(self,api_response) -> dict:
        parsed=super()._parse_query_response_prompting(api_response)
        history=parsed['model_responses_message_for_chat_history']
        calls=history.get('tool_calls',[])
        if calls and any(not isinstance(call,dict) or
                         not isinstance(call.get('name'),str) or
                         not isinstance(call.get('arguments'),dict) for call in calls):
            # The official parser can return raw strings for malformed calls.
            # Keep that model output verbatim in history so the native template
            # does not crash on the next turn. Official decoding/scoring still
            # receives the original malformed response, without repairing it.
            parsed['model_responses_message_for_chat_history']={
                'role':'assistant','content':parsed['model_responses']}
        return parsed

    @override
    def _query_prompting(self,inference_data: dict):
        prompt=self._format_prompt(inference_data['message'],inference_data['function'])
        ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        allowance=4096
        if len(ids)+allowance>self.max_context_length:
            raise ValueError(f'BFCL input plus output allowance exceeds server capacity {self.max_context_length}: {len(ids)}+{allowance} tokens')
        inference_data['inference_input_log']={'formatted_prompt':prompt,'input_tokens':len(ids),'max_tokens':allowance}
        started=time.time()
        response=self.client.completions.create(model='quality-model',prompt=ids,temperature=0,
            max_tokens=allowance,extra_body={'seed':17},timeout=1800)
        raw=response.choices[0].text
        if self.gpu==1 and '<function=' in raw:
            from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
            from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
            tools=[{'type':'function','function':f} for f in inference_data['function']]
            request=ChatCompletionRequest(model='quality-model',messages=[{'role':'user','content':''}],tools=tools)
            parsed=Qwen3EngineToolParser(self.tokenizer,tools=request.tools).extract_tool_calls(raw,request)
            if parsed.tools_called:
                response.choices[0].text=(parsed.content or '')+'\n'+'\n'.join(
                    '<tool_call>\n'+json.dumps({'name':c.function.name,'arguments':json.loads(c.function.arguments)})+'\n</tool_call>'
                    for c in parsed.tool_calls)
        record={'time':time.time(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'input_tokens':len(ids),'max_tokens':allowance,'usage':response.usage.model_dump(),
                'server_context_capacity_tokens':self.max_context_length,
                'raw_output':raw,'normalized_output':response.choices[0].text}
        # One append syscall keeps concurrent request records intact.
        fd=os.open(str(self.journal),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try: os.write(fd,(json.dumps(record)+'\n').encode())
        finally: os.close(fd)
        return response,time.time()-started


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args(); started=time.time()
    assert VERSION_PREFIX=='BFCL_v3'
    from urllib.parse import urlparse
    endpoint=urlparse(os.environ['QBENCH_ENDPOINT'])
    os.environ.update(VLLM_ENDPOINT=endpoint.hostname,VLLM_PORT=str(endpoint.port))
    dest=ROOT/'results'/a.attempt/'BFCLv3'/('pilot' if a.limit else 'full');dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists(): return
    model_name='quality-model'
    StudyHandler.gpu=a.gpu;StudyHandler.tokenizer_path=paths(a.gpu,False)[1]
    StudyHandler.server_capacity=server_context_capacity(a.gpu)
    StudyHandler.tokenizer_shared=AutoTokenizer.from_pretrained(StudyHandler.tokenizer_path)
    StudyHandler.journal=dest/'requests.jsonl'
    MODEL_CONFIG_MAPPING[model_name]=ModelConfig(model_name=model_name,display_name=model_name,
        url='https://huggingface.co/Qwen',org='local study',license='Apache-2.0',model_handler=StudyHandler)
    _,categories,entries=get_involved_test_entries(['all'],False)
    expected=Counter(x['id'].rsplit('_',1)[0] for x in entries)
    if a.limit:
        seen=Counter();subset=[]
        for x in entries:
            category=x['id'].rsplit('_',1)[0]
            if seen[category]<a.limit:subset.append(x);seen[category]+=1
        entries=subset
    resultdir=dest/'responses';existing=set()
    for path in (resultdir/model_name).glob('*_result.json'):
        records=recover_jsonl(path)
        failed=[record for record in records if inference_failed(record) or needs_capacity_retry(record,StudyHandler.server_capacity)]
        if failed:
            # Retain diagnostics while allowing failed API requests to be retried.
            with (dest/'inference-error-history.jsonl').open('a') as history:
                for record in failed:
                    history.write(json.dumps({'retry_at':time.time(),'record':record})+'\n')
            records=[record for record in records if not inference_failed(record) and not needs_capacity_retry(record,StudyHandler.server_capacity)]
            temporary=path.with_suffix('.tmp')
            temporary.write_text(''.join(json.dumps(record)+'\n' for record in records))
            temporary.replace(path)
        existing.update(record['id'] for record in records)
    pending=process_multi_turn_test_case([deepcopy(x) for x in entries if x['id'] not in existing])
    def infer(entry):
        handler=StudyHandler(model_name,0)
        result=handler._multi_threaded_inference(entry,True,False)
        result['study_server_context_capacity_tokens']=StudyHandler.server_capacity
        return result
    writer=StudyHandler(model_name,0)
    with ThreadPoolExecutor(max_workers=int(os.environ.get('QBENCH_CONCURRENCY','4'))) as pool:
        futures=[pool.submit(infer,x) for x in pending]
        for future in as_completed(futures):
            result=future.result();writer.write(result,resultdir,update_mode=False)
            print('BFCL completed',result['id'],flush=True)
    records=[]
    for path in (resultdir/model_name).glob('*_result.json'):
        records.extend(json.loads(line) for line in path.open() if line.strip())
    errors=[x['id'] for x in records if inference_failed(x)]
    counts=Counter(x['id'].rsplit('_',1)[0] for x in records)
    report={'sample_counts_saved':dict(counts),'expected_counts':dict(expected),
        'inference_errors':errors,'elapsed_s':time.time()-started,
        'study_protocol':{'pilot':bool(a.limit),'workload':'BFCLv3',
            'context_policy':'benchmark_native_prompts_and_output_limits',
            'server_context_capacity_tokens':StudyHandler.server_capacity,
            'capacity_provenance':'Successful prior responses reused only when their full 4096-token allowance fit the original serving capacity. Capacity-limited cases and inference errors rerun.',
            'output_allowance_tokens':4096,
            'official_revision':'faa6bdc668a9f6e6ad71512c51c410bb84f2ec09','thinking':False,
            'native_template':StudyHandler.tokenizer_path,'xml_parser':'vllm Qwen3EngineToolParser' if a.gpu else None}}
    if a.limit:
        atomic_json(dest/'pilot-report.json',report);return
    if errors or counts!=expected:
        atomic_json(dest/'incomplete.json',report)
        raise RuntimeError(f'BFCL incomplete: {len(errors)} inference errors; counts {counts}')
    score_dir=dest/'scores'
    eval_runner.main([model_name],['all'],resultdir,score_dir)
    category_scores={}
    for category in categories:
        path=score_dir/model_name/f'BFCL_v3_{category}_score.json'
        with path.open() as f: category_scores[category]=json.loads(next(f))
        assert category_scores[category]['total_count']==expected[category]
    acc=lambda c:category_scores[c]['accuracy']
    # Exact nesting of the pinned official v3 generate_leaderboard_csv, before rounding.
    simple=sum(acc(c) for c in ['simple','java','javascript'])/3
    non_live=(simple+sum(acc(c) for c in ['multiple','parallel','parallel_multiple','irrelevance']))/5
    live=TEST_COLLECTION_MAPPING['live'];multi=TEST_COLLECTION_MAPPING['multi_turn']
    live_score=sum(category_scores[c]['correct_count'] for c in live)/sum(expected[c] for c in live)
    multi_score=sum(acc(c) for c in multi)/4
    report['results']={'bfcl_v3':{'acc,none':(non_live+live_score+multi_score)/3}}
    report['category_scores']=category_scores
    atomic_json(dest/'summary.json',report)
    print(json.dumps(report['results']),flush=True)


if __name__=='__main__':main()
