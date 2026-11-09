(() => {
  'use strict';
  const explorer = document.getElementById('cases');
  if (!explorer || !document.getElementById('day-player')) return;
  const cases = JSON.parse(document.getElementById('case-data').textContent);
  const el = id => document.getElementById('dy-' + id);
  const video = el('video');
  const fieldNames = {domain:'Scenario', people:'People', viewpoint:'Viewpoint'};
  const behaviors = {
    wait: ['Keep observing', '继续观察'], warn: ['Give a warning', '发出提醒'],
    inform: ['Report the event', '告知变化'], answer: ['Answer the question', '回答提问'],
    feedback: ['Give feedback', '提供反馈']
  };
  let index = 0, language = 'en', lastWindow = null, raf = 0;
  const filters = {domain:'', people:'', viewpoint:''};
  const current = () => cases[index];
  const seconds = n => Number(n.toFixed(2)) + ' s';
  const node = (tag, content, cls) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (content !== undefined) n.textContent = content;
    return n;
  };
  const shelf = el('shelf');
  const matches = (c, exclude) => Object.entries(filters).every(([key,value]) =>
    key === exclude || !value || c[key] === value);
  const selects = [...explorer.querySelectorAll('[data-dy-filter]')];
  const allLabels = {domain:'All scenarios', people:'All people', viewpoint:'All viewpoints'};
  function updatePaging() {
    const max = shelf.scrollWidth - shelf.clientWidth;
    el('prev').disabled = shelf.scrollLeft < 2;
    el('next').disabled = shelf.scrollLeft >= max - 2;
    explorer.querySelector('.dy-paging').hidden = max < 2;
  }
  function caseTile(c, i) {
    const btn = node('button', undefined, 'dy-choice');
    btn.type = 'button';
    btn.dataset.dyCase = i;
    btn.setAttribute('aria-pressed',String(index === i));
    const shot = node('span', undefined, 'dy-shot');
    const img = node('img');
    img.src = c.poster;
    img.alt = '';
    img.loading = 'lazy';
    const play = node('span','▶','dy-tile-play');
    play.setAttribute('aria-hidden','true');
    shot.append(img,play);
    const title = node('strong',c.title,'dy-tile-title');
    btn.append(shot,node('span',c.domain,'dy-tile-domain'),title);
    return btn;
  }
  function buildLibrary() {
    for (const select of selects) {
      const key = select.dataset.dyFilter;
      const names = [...new Set(cases.map(c => c[key]))].sort((a,b)=>a.localeCompare(b));
      const options = names.map(name => {
        const option = new Option(name,name);
        option.disabled = !cases.some(c=>matches(c,key) && c[key]===name);
        return option;
      });
      select.replaceChildren(new Option(allLabels[key],''),...options);
      select.value = filters[key];
    }
    const visible = cases.map((c,i)=>({c,i})).filter(({c})=>matches(c));
    shelf.replaceChildren(...visible.map(({c,i})=>caseTile(c,i)));
    shelf.scrollLeft = 0;
    el('count').textContent = visible.length + (visible.length===1?' episode':' episodes');
    el('reset').hidden = !Object.values(filters).some(Boolean);
    updatePaging();
    return visible;
  }
  function applyFilters() {
    const visible = buildLibrary();
    if (visible.length && !matches(current())) choose(visible[0].i,false,false);
  }
  for (const select of selects) {
    select.addEventListener('change',()=>{
      filters[select.dataset.dyFilter] = select.value;
      applyFilters();
    });
  }
  function resetFilters() { for (const key of Object.keys(filters)) filters[key]=''; }
  el('reset').addEventListener('click',()=>{ resetFilters();applyFilters();selects[0].focus(); });
  function pageShelf(direction) {
    const tile = shelf.firstElementChild;
    if (tile) shelf.scrollBy({left:direction*(tile.offsetWidth+12),behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  }
  el('prev').addEventListener('click',()=>pageShelf(-1));
  el('next').addEventListener('click',()=>pageShelf(1));
  shelf.addEventListener('scroll',updatePaging,{passive:true});
  new ResizeObserver(updatePaging).observe(shelf);
  function makeTimeline() {
    const c = current(), seek = el('seek');
    seek.min = c.sourceStart; seek.max = c.sourceEnd;
    el('extent').textContent = 'Original video time · excerpt ' + seconds(c.sourceStart) + ' – ' + seconds(c.sourceEnd) + '. Select a frame to seek.';
    const midpoint = (c.sourceStart+c.sourceEnd)/2;
    el('ticks').replaceChildren(...[c.sourceStart, midpoint, c.sourceEnd].map((t,i)=>{
      const tick = node('span',seconds(t));tick.style.left=(i*50)+'%';return tick;
    }));
    el('bands').replaceChildren(...c.timeline.map(w=>{
      const band=node('span',w.action); band.dataset.action=w.action;
      band.style.left=100*w.start/c.duration+'%';band.style.width=100*(w.end-w.start)/c.duration+'%';
      band.title=w.action+' · '+seconds(w.sourceStart)+' – '+seconds(w.sourceEnd);return band;
    }));
    el('pins').replaceChildren(...c.keyframes.map(k=>{
      const pin=node('i');pin.style.left=100*k.time/c.duration+'%';return pin;
    }));
    el('frames').replaceChildren(...c.keyframes.map(k=>{
      const btn=node('button',undefined,'dy-frame'), img=node('img');
      btn.type='button';btn.dataset.time=k.time;img.src=k.frame;img.alt='';
      btn.title=k.label+' · '+seconds(k.sourceTime)+' in the original video';
      btn.setAttribute('aria-label','Seek to '+seconds(k.sourceTime)+', '+k.label);
      btn.append(img,node('strong',seconds(k.sourceTime)),node('small',k.label));
      btn.addEventListener('click',()=>{video.pause();seekClip(k.time);});return btn;
    }));
    layoutFrames();
  }
  function layoutFrames() {
    // Every connector begins at its real timestamp. Use another row when neighboring
    // thumbnails would overlap, and clamp edge cards inside the available width.
    const frameBox=el('frames'), width=el('track').clientWidth;
    if (!width || !frameBox.children.length) return;
    const c=current(), cardWidth=Math.min(124,Math.max(77,width*.15)), gap=9, connector=17;
    const rows=[], boxes=[];
    [...frameBox.children].forEach((b,i)=>{
      const x=c.keyframes[i].time/c.duration*width;
      const left=Math.max(0,Math.min(width-cardWidth,x-cardWidth/2));
      let row=0;
      while(rows[row] && left < rows[row].end+gap) row++;
      b.style.width=cardWidth+'px';b.style.left=left+'px';
      rows[row]={end:left+cardWidth,height:Math.max(rows[row]?.height||0,b.offsetHeight)};
      boxes.push({b,x,left,row});
    });
    const offsets=[]; let bottom=connector;
    for (const r of rows) {offsets.push(bottom);bottom+=r.height+10;}
    const svg=el('lines');svg.setAttribute('viewBox','0 0 '+width+' '+bottom);svg.replaceChildren();
    for(const {b,x,left,row} of boxes){
      const y=offsets[row],target=Math.max(left+7,Math.min(x,left+cardWidth-7));
      b.style.top=y+'px';
      const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');
      line.setAttribute('points',x+',0 '+x+','+(connector*.42)+' '+target+','+(connector-2)+' '+target+','+y);
      svg.append(line);
    }
    frameBox.style.height=bottom+'px';
  }
  function updateState() {
    const c=current(), t=Math.min(c.duration,Math.max(0,video.currentTime||0));
    const original=Math.min(c.sourceEnd,c.sourceStart+c.sourceRate*t);
    el('clock').textContent=seconds(original);
    el('playhead').style.left='clamp(0px,'+(100*t/c.duration)+'%,calc(100% - 2px))';
    el('seek').value=original;el('seek').setAttribute('aria-valuetext',seconds(original)+' in the original video');
    const win=c.timeline.find(w=>t>=w.start && t<w.end)||c.timeline.at(-1);
    if(win!==lastWindow){
      lastWindow=win;
      el('action').textContent=behaviors[win.action]?.[language==='en'?0:1]||win.action;
      el('response').textContent=win[language] || (c.id==='adult_gas_neg'
        ? (language==='en'?'The burner is off. The user only needs a warning if it stays on.':'炉火已关。用户只要求在炉火仍开着时提醒。')
        : (language==='en'?'The assistant should stay silent at this point.':'此时助手应保持安静。'));
      el('window').textContent='Reference window · '+seconds(win.sourceStart)+' – '+seconds(win.sourceEnd)+' in the original video';
    }
    let active=0;c.keyframes.forEach((k,i)=>{if(k.time<=t+.0005)active=i;});
    [...el('frames').children].forEach((b,i)=>b.setAttribute('aria-current',String(i===active)));
  }
  function seekClip(time) {
    const selected=index;
    function apply(){if(index===selected){video.currentTime=Math.min(time,current().duration);updateState();}}
    if(video.readyState>=1)apply();else video.addEventListener('loadedmetadata',apply,{once:true});
  }
  function changeLanguage(lang) {
    language=lang;lastWindow=null;
    explorer.querySelectorAll('[data-dy-lang]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.dyLang===lang)));
    for(const id of ['task','response','action']) el(id).lang=lang;
    el('task').textContent=current().task[lang];updateState();
  }
  function choose(i, scroll, play) {
    if(!Number.isInteger(i)||!cases[i])return;
    video.pause();index=i;lastWindow=null;
    const c=current();
    if(!matches(c)){resetFilters();buildLibrary();}
    explorer.querySelectorAll('[data-dy-case]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.dyCase)===i)));
    el('title').textContent=c.title;
    el('source').textContent=(c.sourceType==='real'?'Real-world':'Synthetic')+' film excerpt';
    el('record').href='https://github.com/todayai-labs/omni-life/blob/main/'+c.recordPath;
    el('facts').replaceChildren(...Object.entries(fieldNames).map(([key,label])=>{
      const fact=node('span');fact.append(node('b',label),document.createTextNode(c[key]));return fact;
    }));
    video.poster=c.keyframes[0].frame;video.src=c.media;video.setAttribute('aria-label',c.title+' film excerpt');
    makeTimeline();changeLanguage(language);updateControls();
    if(scroll)document.getElementById('day-player').scrollIntoView({block:'start'});
    if(play)startPlayback();
  }
  function updateControls(){
    el('play').textContent=video.paused||video.ended?'▶ Play excerpt':'Ⅱ Pause';
    el('audio').textContent=video.muted?'Sound off':'Sound on';
    el('audio').setAttribute('aria-pressed',String(!video.muted));
  }
  function startPlayback(){video.play().catch(()=>updateControls());}
  function toggle(){if(video.paused||video.ended){if(video.ended)video.currentTime=0;startPlayback();}else video.pause();}
  function tick(){updateState();if(!video.paused&&!video.ended)raf=requestAnimationFrame(tick);}
  explorer.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    if(b.hasAttribute('data-dy-case'))choose(Number(b.dataset.dyCase),true,true);
    if(b.hasAttribute('data-dy-lang'))changeLanguage(b.dataset.dyLang);
  });
  el('seek').addEventListener('input',e=>seekClip((Number(e.target.value)-current().sourceStart)/current().sourceRate));
  el('play').addEventListener('click',toggle);video.addEventListener('click',toggle);
  el('audio').addEventListener('click',()=>{video.muted=!video.muted;updateControls();});
  el('fullscreen').addEventListener('click',()=>{if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});else if(video.parentElement.requestFullscreen)video.parentElement.requestFullscreen().catch(()=>{});else if(video.webkitEnterFullscreen)video.webkitEnterFullscreen();});
  ['loadedmetadata','seeked','timeupdate'].forEach(ev=>video.addEventListener(ev,updateState));
  video.addEventListener('play',()=>{updateControls();cancelAnimationFrame(raf);raf=requestAnimationFrame(tick);});
  for(const ev of ['pause','ended'])video.addEventListener(ev,()=>{cancelAnimationFrame(raf);updateState();updateControls();});
  // The track width (rather than storyboard height) drives the collision layout.
  new ResizeObserver(layoutFrames).observe(el('track'));
  document.fonts?.ready.then(layoutFrames);
  video.controls=false;
  window.openDayCase=i=>choose(Number(i),true,true);
  buildLibrary();choose(0,false,false);
})();
