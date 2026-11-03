import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {inflateRawSync} from 'node:zlib';
import {Model} from './runtime.mjs';
const config=new Uint8Array(await readFile('build/v17a-compact/config.json'));
const weights=new Uint8Array(await readFile('build/v17a-compact/weights.sparse.zip'));
const first=await Model.fromBytes(config,weights,inflateRawSync),second=await Model.fromBytes(config,weights,inflateRawSync);
first.contextLimit=second.contextLimit=1024;
assert.equal(first.generate('A kelv can fly. What can a kelv do?').text,'Kelv can fly.');
assert.equal(second.generate('Can a kelv fly?').text,'I do not know.');
assert.equal(first.generate('Can a kelv fly?').text,'Yes.');
const state=JSON.stringify({history:first.history,pointers:[...first.tokenizer.pointers],used:first.tokenizer.used});
assert.throws(()=>first.generate('A zorb can fly. '+ 'hello '.repeat(1100)),/Insufficient context space/);
assert.equal(JSON.stringify({history:first.history,pointers:[...first.tokenizer.pointers],used:first.tokenizer.used}),state);
assert.equal(first.generate('Can a kelv fly?').text,'Yes.');
first.reset();assert.equal(first.generate('Can a kelv fly?').text,'I do not know.');
const corrupt=weights.slice();corrupt[100]^=1;
await assert.rejects(()=>Model.fromBytes(config,corrupt,inflateRawSync),/Weight checksum mismatch/);
const wrongConfig=config.slice();wrongConfig[100]^=1;
await assert.rejects(()=>Model.fromBytes(wrongConfig,weights,inflateRawSync),/Config checksum mismatch/);
const report={created_utc:new Date().toISOString(),independent_conversation_state:true,reset_clears_memory:true,
  browser_context_limit:1024,oversized_prompt_rejected:true,rejected_prompt_does_not_mutate_context:true,
  corrupt_weights_rejected:true,changed_config_rejected:true};
await writeFile('publication/browser-boundary-verification.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
