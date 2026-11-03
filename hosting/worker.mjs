import {Model} from './runtime.mjs';
let model;
async function load(path) {
  const response=await fetch(new URL(path,import.meta.url));
  if(!response.ok) throw new Error(`Could not download ${path} (${response.status})`);
  return new Uint8Array(await response.arrayBuffer());
}
self.onmessage=async ({data})=>{
  try {
    if(data.action==='load') {
      const [config,weights]=await Promise.all([load('config.json'),load('weights.sparse.zip')]);
      model=await Model.fromBytes(config,weights);
      model.contextLimit=1024;
      self.postMessage({type:'ready',context:model.contextLimit});
    } else if(data.action==='reset') {
      model.reset(); self.postMessage({type:'reset'});
    } else if(data.action==='generate') {
      if(!model) throw new Error('The model is still loading.');
      if(typeof data.prompt!=='string'||!data.prompt.trim()||data.prompt.length>20000) throw new Error('Enter a message of up to 20,000 characters.');
      const result=model.generate(data.prompt,64,progress=>self.postMessage({type:'progress',...progress}));
      self.postMessage({type:'result',result});
    }
  } catch(error) { self.postMessage({type:'error',message:error.message}); }
};
