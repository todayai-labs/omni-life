'use strict';
const cases = JSON.parse(document.getElementById('case-data').textContent);
const models = JSON.parse(document.getElementById('model-data').textContent);
const $ = id => document.getElementById(id);
const video = $('episode-video');
const dialog = $('media-dialog');
let selected = 0, language = 'en', feature = 0, previousWindow = null;
function make(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function seconds(t) { return Number(t.toFixed(2)).toString() + ' s'; }
function drawState() {
  const c = cases[selected], t = Math.min(c.duration,Math.max(video.currentTime || 0,0));
  $('clip-clock').textContent = seconds(t) + ' / ' + seconds(c.duration);
  $('needle').style.left = Math.min(99.8,t * 100 / c.duration) + '%';
  $('scrub').value = t;
  $('scrub').setAttribute('aria-valuetext', seconds(t));
  const window = c.timeline.find(w=>t >= w.start && t < w.end) || c.timeline.at(-1);
  if (window !== previousWindow) {
    $('action-name').textContent = 'Reference · ' + window.action;
    $('reference').textContent = window[language] || (language === 'en' ? 'Remain silent.' : '保持沉默。');
    $('source-time').textContent = 'Original video window · ' + seconds(window.sourceStart) + ' – ' + seconds(window.sourceEnd);
    previousWindow = window;
  }
  const current = c.keyframes.findLastIndex(k => k.time <= t + 0.015);
  [...$('keyframes').children].forEach((btn,i)=>btn.setAttribute('aria-current',String(i===current)));
}
function updateText() {
  $('task').textContent = cases[selected].task[language];
  $('task').lang = language === 'en' ? 'en' : 'zh';
  $('lang').textContent = language === 'en' ? '中文' : 'English';
  previousWindow = null; drawState();
}
function seek(t) {
  if (video.readyState >= 1) video.currentTime = t;
  else video.addEventListener('loadedmetadata',()=>{ video.currentTime = t; }, {once:true});
  drawState();
}
function showCase(i, scroll=false, autoplay=false) {
  if (!cases[i]) return;
  selected = i; previousWindow = null;
  const c = cases[i];
  video.pause(); video.poster = c.poster; video.src = c.media;
  video.setAttribute('aria-label',c.title + ', film excerpt');
  $('case-title').textContent = c.title;
  $('case-domain').textContent = c.domain + ' / ' + (c.sourceType === 'real' ? 'Real-world' : 'Synthetic');
  [...$('case-menu').children].forEach((btn,n)=>btn.setAttribute('aria-pressed',String(n===i)));
  $('scrub').max = c.duration;
  $('bands').replaceChildren(...c.timeline.map(w=>{
    const b = make('span',null,'band '+w.action);
    b.style.left = (100*w.start/c.duration)+'%'; b.style.width = (100*(w.end-w.start)/c.duration)+'%'; b.title=w.action; return b;
  }));
  for (const k of c.keyframes) {
    const tick = make('i',null,'frame-tick');
    tick.style.left = (100*k.time/c.duration)+'%'; tick.setAttribute('aria-hidden','true');
    $('bands').append(tick);
  }
  $('keyframes').replaceChildren(...c.keyframes.map(k=>{
    const btn=make('button',null,'keyframe'), img=make('img');
    btn.type='button'; img.src=k.frame; img.alt=''; btn.append(img,make('small',seconds(k.time)+' · '+k.action),make('span',k.label,'key-label'));
    btn.title=k.label; btn.setAttribute('aria-label','Seek to '+seconds(k.time)+', '+k.label);
    btn.addEventListener('click',()=>seek(k.time)); return btn;
  }));
  updateText();
  if (scroll) $('watch').scrollIntoView({block:'start'});
  if (autoplay) video.play().catch(()=>{});
}
function highlight(i) {
  const c=cases[i]; if (!c || !$('day-query')) return;
  feature=i; $('day-query').textContent=c.task.en;
  const button=document.querySelector('.day-scene'); button.dataset.case=i;button.querySelector('img').src=c.poster;
  document.querySelectorAll('[data-highlight]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.highlight)===i)));
}
function openMedia(kind, image, label) {
  video.pause(); const body=$('modal-content');
  const item=make(kind==='film'?'video':'img'); body.replaceChildren(item);
  if (kind==='film') {
    item.controls=true; item.playsInline=true; item.src='media/overview.mp4'; item.poster='assets/film-poster.jpg'; item.setAttribute('aria-label','Omni-Life project film');
  } else {item.src=image; item.alt=label;}
  dialog.showModal();
  if (kind==='film') item.play().catch(()=>{});
}
function table() {
  const cols=['full_correct','intervention_recall','intervention_f1','silence_success','joint_qa'], sort=$('metric').value;
  $('scores').replaceChildren(...[...models].sort((a,b)=>b[sort]-a[sort]).map(m=>{
    const tr=make('tr');tr.append(make('td',m.model),...cols.map(k=>make('td',m[k].toFixed(2))));return tr;
  }));
}
document.querySelectorAll('[data-cover]').forEach(im=>{ const c=cases[Number(im.dataset.cover)]; if (c) im.src=c.poster; });
$('case-menu').replaceChildren(...cases.map((c,i)=>{
 const btn=make('button',null,'case-card'), img=make('img');
 btn.dataset.case=i; img.alt=''; img.src=c.poster; btn.append(img,make('span',c.title)); return btn;
}));
document.addEventListener('click',e=>{
 const button=e.target.closest('button');if(!button)return;
 if (button.hasAttribute('data-film')) openMedia('film');
 else if (button.hasAttribute('data-case')) showCase(Number(button.dataset.case),true,true);
 else if (button.hasAttribute('data-highlight')) highlight(Number(button.dataset.highlight));
 else if (button.hasAttribute('data-figure')) { const img=button.querySelector('img');openMedia('figure',img.src,img.alt); }
});
$('scrub').addEventListener('input',e=>seek(Number(e.target.value)));
$('lang').addEventListener('click',()=>{language=language==='en'?'zh':'en';updateText();});
['timeupdate','seeked','loadedmetadata','ended'].forEach(ev=>video.addEventListener(ev,drawState));
$('close').addEventListener('click',()=>dialog.close());dialog.addEventListener('close',()=>{const film=dialog.querySelector('video');if(film){film.pause();film.removeAttribute('src');film.load();}$('modal-content').replaceChildren();});
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
$('metric').addEventListener('change',table);table();showCase(0);highlight(0);
