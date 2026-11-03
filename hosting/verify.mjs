import {readFile,writeFile} from 'node:fs/promises';
import {inflateRawSync} from 'node:zlib';
import {Model} from './runtime.mjs';
const read=async p=>new Uint8Array(await readFile(p));
const model=await Model.fromBytes(await read('build/v17a-compact/config.json'),await read('build/v17a-compact/weights.sparse.zip'),inflateRawSync);
const transcript=JSON.parse(await readFile('publication/demo-transcript-compact.json','utf8'));
const rows=[];
for(const conversation of transcript.conversations) {
  model.reset();
  for(const baseline of conversation.turns) {
    const result=model.generate(baseline.prompt);
    const equal=result.text===baseline.text && JSON.stringify(result.token_ids)===JSON.stringify(baseline.token_ids) && result.stopped_on_eos===baseline.stopped_on_eos;
    rows.push({conversation:conversation.id,...result,expected:baseline.text,identical:equal});
    console.log(JSON.stringify({conversation:conversation.id,text:result.text,seconds:result.seconds,identical:equal}));
  }
}
await writeFile('publication/browser-runtime-verification.json',JSON.stringify({created_utc:new Date().toISOString(),backend:'JavaScript sparse float32 tensors, double accumulation',cases:rows.length,identical:rows.filter(r=>r.identical).length,rows},null,2)+'\n');
if(rows.some(r=>!r.identical)) process.exitCode=1;
