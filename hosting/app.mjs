const $=id=>document.getElementById(id);
const worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
let ready=false,busy=false;
function state(text) { $('status').textContent=text; }
function controls() { $('send').disabled=!ready||busy; $('prompt').disabled=!ready||busy; $('reset').disabled=!ready||busy; document.querySelectorAll('.example').forEach(b=>b.disabled=!ready||busy); }
function message(role,text,seconds) {
  $('empty')?.remove();
  const box=document.createElement('div'),who=document.createElement('div'),bubble=document.createElement('div');
  box.className='message '+role; who.className='who'; bubble.className='bubble';
  who.textContent=role==='user'?'YOU':'V17A · FROZEN WEIGHTS';
  if(seconds!==undefined) { const timing=document.createElement('span');timing.className='timing';timing.textContent=` / ${Math.round(seconds*1000).toLocaleString()} ms on this device`;who.append(timing); }
  bubble.textContent=text; box.append(who,bubble);$('messages').append(box);$('messages').scrollTop=$('messages').scrollHeight;
}
function error(text) { $('error').textContent=text;$('error').hidden=false; }
worker.onmessage=({data})=>{
  if(data.type==='ready') { ready=true;state('Ready · inference in this browser');controls(); }
  if(data.type==='reset') { $('messages').replaceChildren();$('error').hidden=true;$('context').textContent='Context: 1 / 1,024 tokens';busy=false;state('New conversation · ready');controls();$('prompt').focus(); }
  if(data.type==='progress') state(data.phase==='prefill'?`Reading conversation · layer ${data.layer} / ${data.total}`:`Generating · ${data.tokens} tokens`);
  if(data.type==='result') { const r=data.result;message('assistant',r.text,r.seconds);$('context').textContent=`Context: ${(r.prefix_tokens+r.generated_tokens).toLocaleString()} / 1,024 tokens`;busy=false;state('Ready · inference in this browser');controls();$('prompt').focus();if(!r.stopped_on_eos)error('The model reached the output limit. This response may be incomplete.'); }
  if(data.type==='error') { busy=false;error(data.message);state(ready?'Model error · new conversation may help':'Loading failed · refresh to retry');controls(); }
};
worker.onerror=event=>{ busy=false;ready=false;error(event.message||'The browser worker stopped. Refresh to reload the model.');state('Worker stopped');controls(); };
$('compose').onsubmit=event=>{
  event.preventDefault();const prompt=$('prompt').value.trim();if(!ready||busy||!prompt)return;
  $('error').hidden=true;$('prompt').value='';message('user',prompt);busy=true;controls();state('Reading conversation…');worker.postMessage({action:'generate',prompt});
};
$('prompt').onkeydown=event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('compose').requestSubmit();}};
$('reset').onclick=()=>{if(busy)return;busy=true;controls();worker.postMessage({action:'reset'});};
document.querySelectorAll('.example').forEach(button=>button.onclick=()=>{$('prompt').value=button.dataset.prompt;$('prompt').focus();});
controls();worker.postMessage({action:'load'});
