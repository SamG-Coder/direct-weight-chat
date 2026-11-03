/** Browser inference for the frozen v17a transformer. No response rules or training.
 * All linear maps come from the original, hash-checked sparse weight archive.
 * Sparse dot products omit mathematical zero terms; JS accumulation differs from
 * PyTorch, so this backend has its own measured token-parity report.
 */
export const WEIGHT_SHA = '4739a3511f5c4b9170f5d09524bd8105f5853a7999cbeefdea7a8cbae01d737d';
export const CONFIG_SHA = '56df33d5842a0d3a1351dae78732c01589bde1ce55a33868f9d13dec171bee91';
const decoder = new TextDecoder();

export async function sha256(bytes) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
}

async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function unpackZip(bytes, decompress = inflate) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let end = bytes.length - 22;
  while (end >= Math.max(0, bytes.length - 65557) && view.getUint32(end, true) !== 0x06054b50) end--;
  if (end < 0) throw new Error('Invalid weight archive');
  const count = view.getUint16(end + 10, true);
  let pos = view.getUint32(end + 16, true);
  const members = new Map();
  for (let n = 0; n < count; n++) {
    if (view.getUint32(pos, true) !== 0x02014b50) throw new Error('Invalid ZIP directory');
    const method = view.getUint16(pos + 10, true), size = view.getUint32(pos + 20, true);
    const original = view.getUint32(pos + 24, true), len = view.getUint16(pos + 28, true);
    const extra = view.getUint16(pos + 30, true), comment = view.getUint16(pos + 32, true);
    const local = view.getUint32(pos + 42, true), name = decoder.decode(bytes.subarray(pos + 46, pos + 46 + len));
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    const compressed = bytes.subarray(start, start + size);
    const raw = method === 0 ? compressed.slice() : method === 8 ? await decompress(compressed) : null;
    if (!raw || raw.length !== original || members.has(name)) throw new Error('Invalid ZIP member');
    members.set(name, raw);
    pos += 46 + len + extra + comment;
  }
  return members;
}

export class Tokenizer {
  constructor(config) {
    this.config = config;
    this.labels = config.vocabulary;
    this.ids = new Map(this.labels.map((x, i) => [x.toLowerCase(), i]));
    this.pointers = new Map(); this.reverse = new Map(); this.used = [0, 0];
    this.maxPhrase = Math.max(...this.labels.map(x => x.trim().split(/\s+/).length));
  }
  encode(text) {
    // Python's original regex uses Unicode-aware \w in the punctuation branch.
    const words = text.match(/[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\p{L}\p{N}_\s]/gu) || [];
    const out = [];
    for (let i = 0; i < words.length;) {
      let found = false;
      for (let len = Math.min(this.maxPhrase, words.length - i); len > 0; len--) {
        const key = words.slice(i, i + len).join(' ').toLowerCase();
        if (this.ids.has(key) && !key.startsWith('<')) { out.push(this.ids.get(key)); i += len; found = true; break; }
      }
      if (found) continue;
      const original = words[i++], key = original.toLowerCase();
      if (!this.pointers.has(key)) {
        const section = 'aeiou'.includes(key[0]) ? 0 : 1;
        if (this.used[section] >= 64) throw new Error('The fixed pointer vocabulary is full; start a new conversation.');
        const tid = this.config.dynamic_start + 64 * section + this.used[section]++;
        this.pointers.set(key, tid); this.reverse.set(tid, original);
      }
      out.push(this.pointers.get(key));
    }
    return out;
  }
  tokenText(id) { return this.reverse.get(id) ?? this.labels[id]; }
  decode(ids) {
    return ids.filter(x => x !== this.config.special.eos).map(x => this.tokenText(x)).join(' ')
      .replace(/\s+([.,?!:;])/g, '$1').replace(/(?<![\p{L}\p{N}_])-\s+(?=\d)/gu, '-')
      .replace(/\bi\b/g, 'I').replace(/(^|[.!?]\s+)([a-z])/g, (_, a, b) => a + b.toUpperCase());
  }
}

class Tensor {
  constructor(spec, members) {
    this.shape = spec.shape;
    const ib = members.get(spec.prefix + '.indices.u32'), vb = members.get(spec.prefix + '.values.f32');
    if (!ib || !vb || ib.length !== spec.entries * 4 || vb.length !== ib.length) throw new Error('Truncated tensor');
    this.indices = new Uint32Array(Uint8Array.from(ib).buffer);
    this.values = new Float32Array(Uint8Array.from(vb).buffer);
    const total = this.shape.reduce((a,b) => a*b, 1);
    for (let i=0; i<this.indices.length; i++) if (this.indices[i]>=total || (i && this.indices[i]<=this.indices[i-1])) throw new Error('Invalid sparse indices');
    if (this.shape.length === 1) {
      this.dense = new Float32Array(this.shape[0]);
      for (let i=0;i<this.indices.length;i++) this.dense[this.indices[i]]=this.values[i];
    } else {
      const [rows, cols] = this.shape;
      this.rowPtr = new Uint32Array(rows + 1); this.col = new Uint32Array(this.indices.length);
      for (let i=0;i<this.indices.length;i++) { this.rowPtr[Math.floor(this.indices[i]/cols)+1]++; this.col[i]=this.indices[i]%cols; }
      for (let r=1;r<=rows;r++) this.rowPtr[r]+=this.rowPtr[r-1];
      this.active = [];
      for (let r=0;r<rows;r++) if (this.rowPtr[r]!==this.rowPtr[r+1]) this.active.push(r);
    }
  }
  addRow(row, output, offset) {
    for (let k=this.rowPtr[row];k<this.rowPtr[row+1];k++) output[offset+this.col[k]] += this.values[k];
  }
}

export class Model {
  static async fromBytes(configBytes, weightBytes, decompress) {
    if (weightBytes.length !== 1528592 || await sha256(weightBytes) !== WEIGHT_SHA) throw new Error('Weight checksum mismatch');
    if (await sha256(configBytes) !== CONFIG_SHA) throw new Error('Config checksum mismatch');
    const members = await unpackZip(weightBytes, decompress);
    const metadata = JSON.parse(decoder.decode(members.get('compact.json')));
    if (metadata.config_sha256 !== CONFIG_SHA) throw new Error('Archive config mismatch');
    return new Model(JSON.parse(decoder.decode(configBytes)), metadata, members);
  }
  constructor(config, metadata, members) {
    this.config=config; this.w=new Map(); this.width=config.width;
    if (config.format!=='direct-causal-transformer-f32-v1' || metadata.format!=='direct-causal-transformer-sparse-f32-v1') throw new Error('Unsupported model');
    if (Object.keys(metadata.tensors).length!==Object.keys(config.tensors).length) throw new Error('Tensor inventory mismatch');
    for (const [name, spec] of Object.entries(metadata.tensors)) {
      if (JSON.stringify(spec.shape)!==JSON.stringify(config.tensors[name]?.shape)) throw new Error('Tensor shape mismatch');
      this.w.set(name,new Tensor(spec,members));
    }
    // Discover structural sparsity from coefficients only; never from a prompt.
    this.attentionDims=[];
    for(let layer=0;layer<config.layers;layer++) {
      const prefix=`blocks.${layer}.attn.qkv`, weight=this.w.get(prefix+'.weight'), bias=this.w.get(prefix+'.bias').dense;
      const active=new Set(weight.active);
      bias.forEach((v,i)=>{if(v!==0)active.add(i);});
      const headDim=this.width/config.heads, heads=[];
      for(let head=0;head<config.heads;head++) {
        const dot=[], values=[];
        for(let d=0;d<headDim;d++) {
          const j=head*headDim+d;
          if(active.has(j)&&active.has(this.width+j)) dot.push(j);
          if(active.has(2*this.width+j)) values.push(j);
        }
        if(values.length) heads.push({dot,values});
      }
      this.attentionDims.push(heads);
    }
    this.reset();
  }
  reset() { this.tokenizer=new Tokenizer(this.config); this.history=[this.config.special.bos]; }
  norm(x,prefix) {
    const w=this.width, gain=this.w.get(prefix+'.weight').dense, bias=this.w.get(prefix+'.bias').dense;
    const out=new Float32Array(x.length);
    for(let offset=0;offset<x.length;offset+=w) {
      let mean=0; for(let j=0;j<w;j++) mean+=x[offset+j]; mean/=w;
      let variance=0; for(let j=0;j<w;j++) variance+=(x[offset+j]-mean)**2; variance/=w;
      const scale=1/Math.sqrt(variance+this.config.norm_epsilon);
      for(let j=0;j<w;j++) out[offset+j]=(x[offset+j]-mean)*scale*gain[j]+bias[j];
    }
    return out;
  }
  linear(x,prefix) {
    const a=this.w.get(prefix+'.weight'), bias=this.w.get(prefix+'.bias').dense;
    const [rows,cols]=a.shape, count=x.length/cols, out=new Float32Array(count*rows);
    for(let t=0;t<count;t++) {
      const xo=t*cols, yo=t*rows;
      out.set(bias,yo);
      for(const row of a.active) {
        let sum=0;
        for(let k=a.rowPtr[row];k<a.rowPtr[row+1];k++) sum+=x[xo+a.col[k]]*a.values[k];
        out[yo+row]=sum+bias[row];
      }
    }
    return out;
  }
  forward(ids,cache=null,progress=()=>{}) {
    const c=this.config,w=this.width,n=ids.length,offset=cache?cache[0].length:0,total=offset+n;
    if(total>c.context_length) throw new Error('Context is full; start a new conversation.');
    let h=new Float32Array(n*w);
    for(let t=0;t<n;t++) { this.w.get('token_embedding').addRow(ids[t],h,t*w); this.w.get('position_embedding').addRow(offset+t,h,t*w); }
    const newCache=[];
    for(let layer=0;layer<c.layers;layer++) {
      const p=`blocks.${layer}`,qkv=this.linear(this.norm(h,p+'.ln1'),p+'.attn.qkv');
      const keys=new Float32Array(total*w),values=new Float32Array(total*w);
      if(cache) { keys.set(cache[layer].k); values.set(cache[layer].v); }
      for(let t=0;t<n;t++) { keys.set(qkv.subarray(t*3*w+w,t*3*w+2*w),(offset+t)*w); values.set(qkv.subarray(t*3*w+2*w,t*3*w+3*w),(offset+t)*w); }
      newCache.push({k:keys,v:values,length:total});
      const attention=new Float32Array(n*w),scale=1/Math.sqrt(w/c.heads),scores=new Float32Array(total);
      for(let t=0;t<n;t++) {
        const upto=offset+t+1,qo=t*3*w;
        for(const head of this.attentionDims[layer]) {
          let max=-Infinity;
          for(let j=0;j<upto;j++) {
            let dot=0; for(const d of head.dot) dot+=qkv[qo+d]*keys[j*w+d];
            scores[j]=dot*scale; if(scores[j]>max) max=scores[j];
          }
          let sum=0; for(let j=0;j<upto;j++) { scores[j]=Math.exp(scores[j]-max); sum+=scores[j]; }
          for(let j=0;j<upto;j++) scores[j]/=sum;
          for(const d of head.values) { let value=0; for(let j=0;j<upto;j++) value+=scores[j]*values[j*w+d]; attention[t*w+d]=value; }
        }
      }
      const residual=this.linear(attention,p+'.attn.out');
      for(let i=0;i<h.length;i++) h[i]+=residual[i];
      const up=this.linear(this.norm(h,p+'.ln2'),p+'.mlp.up');
      for(let i=0;i<up.length;i++) up[i]=Math.max(0,up[i]);
      const down=this.linear(up,p+'.mlp.down');
      for(let i=0;i<h.length;i++) h[i]+=down[i];
      progress(layer+1,c.layers);
    }
    const logits=this.linear(this.norm(h.subarray((n-1)*w),'final_norm'),'lm_head');
    for(const value of logits) if(!Number.isFinite(value)) throw new Error('Nonfinite token logits');
    return {logits,cache:newCache};
  }
  generate(prompt,maxNewTokens=64,progress=()=>{}) {
    const started=performance.now(),c=this.config;
    const previous={pointers:new Map(this.tokenizer.pointers),reverse:new Map(this.tokenizer.reverse),used:[...this.tokenizer.used]};
    let prefix;
    try {
      prefix=[...this.history,c.special.user,...this.tokenizer.encode(prompt),c.special.assistant];
      if(prefix.length+maxNewTokens>(this.contextLimit??c.context_length)) throw new Error('Insufficient context space; start a new conversation.');
    } catch(error) { Object.assign(this.tokenizer,previous);throw error; }
    let state=this.forward(prefix,null,(layer,total)=>progress({phase:'prefill',layer,total,prefix_tokens:prefix.length}));
    const output=[]; let stopped=false;
    for(let step=0;step<maxNewTokens;step++) {
      let token=0;
      for(let j=1;j<state.logits.length;j++) if(state.logits[j]>state.logits[token]) token=j;
      output.push(token); progress({phase:'generate',tokens:output.length});
      if(token===c.special.eos) { stopped=true; break; }
      state=this.forward([token],state.cache);
    }
    this.history=prefix.concat(output);
    return {prompt,text:this.tokenizer.decode(output),token_ids:output,stopped_on_eos:stopped,
      generated_tokens:output.length,seconds:(performance.now()-started)/1000,device:'browser-cpu',prefix_tokens:prefix.length};
  }
}
