/** Compare this independent inference backend with every retained v17a CUDA suite. */
import {readFile,writeFile} from 'node:fs/promises';
import {inflateRawSync} from 'node:zlib';
import {Model,sha256,WEIGHT_SHA} from './runtime.mjs';
const read=async p=>new Uint8Array(await readFile(p));
const model=await Model.fromBytes(await read('build/v17a-compact/config.json'),await read('build/v17a-compact/weights.sparse.zip'),inflateRawSync);
model.contextLimit=1024;
const index=JSON.parse(await readFile('publication/compact-parity-cuda.json','utf8'));
const report={kind:'independent-browser-backend-parity',created_utc:new Date().toISOString(),node:process.version,
  weight_sha256:WEIGHT_SHA,runtime_sha256:await sha256(await read('hosting/runtime.mjs')),
  note:'Sparse linear and attention operations accumulate in JavaScript double precision and store float32 activations. Token parity is measured, not bitwise activation equivalence.',
  browser_context_limit:1024,output_reserve:64,selection:'All retained suite cases fitting the browser limit. Longer cases and dependent continuation turns are explicitly excluded.',
  total:0,identical:0,scored_passes:0,excluded:[],comparisons:[],differences:[],complete:false};
for(const entry of index.comparisons) {
  const suiteBytes=await read(entry.suite), baselineBytes=await read(entry.baseline);
  if(await sha256(suiteBytes)!==entry.suite_sha256 || await sha256(baselineBytes)!==entry.baseline_sha256) throw new Error('Baseline changed');
  const suite=JSON.parse(new TextDecoder().decode(suiteBytes)),baseline=JSON.parse(new TextDecoder().decode(baselineBytes)),rows=[];
  let skippedConversation=false;
  for(let i=0;i<suite.cases.length;i++) {
    const test=suite.cases[i],old=baseline.cases[i];
    if(test.id!==old.id || test.prompt!==old.prompt) throw new Error('Case ordering mismatch');
    if(test.reset??true) {model.reset();skippedConversation=false;}
    if(old.prefix_tokens+64>model.contextLimit || skippedConversation) {
      report.excluded.push({suite:entry.suite,id:test.id,prefix_tokens:old.prefix_tokens,reason:skippedConversation?'Continuation after an excluded turn':'Exceeds browser context limit'});
      skippedConversation=true;continue;
    }
    const result=model.generate(test.prompt,test.max_new_tokens??40);
    const identical=result.text===old.text && JSON.stringify(result.token_ids)===JSON.stringify(old.token_ids) && result.stopped_on_eos===old.stopped_on_eos;
    const correct=(test.acceptable??[test.expected]).includes(result.text)&&result.stopped_on_eos;
    rows.push({id:test.id,text:result.text,token_ids:result.token_ids,stopped_on_eos:result.stopped_on_eos,identical,correct,seconds:result.seconds,prefix_tokens:result.prefix_tokens});
    if(!identical) report.differences.push({suite:entry.suite,id:test.id,old_text:old.text,old_token_ids:old.token_ids,...rows.at(-1)});
  }
  const comparison={suite:entry.suite,suite_sha256:entry.suite_sha256,baseline:entry.baseline,baseline_sha256:entry.baseline_sha256,
    total:rows.length,identical:rows.filter(r=>r.identical).length,scored_passes:rows.filter(r=>r.correct).length,records:rows};
  for(const k of ['total','identical','scored_passes']) report[k]+=comparison[k];
  report.comparisons.push(comparison);
  await writeFile('publication/browser-suite-parity.json',JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({...comparison,records:undefined}));
}
report.complete=true; report.all_outputs_identical=report.total===report.identical;
await writeFile('publication/browser-suite-parity.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({complete:true,total:report.total,identical:report.identical,scored_passes:report.scored_passes}));
if(!report.all_outputs_identical) process.exitCode=1;
