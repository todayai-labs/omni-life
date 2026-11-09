(() => {
  'use strict';
  const explorer = document.getElementById('cases');
  if (!explorer || !document.getElementById('day-player')) return;
  const cases = JSON.parse(document.getElementById('case-data').textContent);
  const el = id => document.getElementById('dy-' + id);
  const video = el('video');
  const behaviors = {
    wait: ['Keep observing', '继续观察'], warn: ['Give a warning', '发出提醒'],
    inform: ['Report the event', '告知变化'], answer: ['Answer the question', '回答提问'],
    feedback: ['Give feedback', '提供反馈']
  };
  let index = 0, language = 'en', lastWindow = null, raf = 0;
  const filters = {environment:'', domain:'', people:'', viewpoint:''};
  const current = () => cases[index];
  const seconds = n => Number(n.toFixed(2)) + ' s';
  const node = (tag, content, cls) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (content !== undefined) n.textContent = content;
    return n;
  };
  const shelf = el('shelf');
  const catalog = explorer.querySelector('.dy-catalog');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let autoEnabled = !motion.matches, autoTimer = 0, autoDirection = 1, autoFrame = 0;
  let autoPosition = 0, lastAutoFrame = null;
  let shelfVisible = false, interacting = false, keyboardMode = false, manualUntil = 0;
  const autoSpeed = 48; // CSS pixels per second, independent of display refresh rate.
  const matches = (c, exclude) => Object.entries(filters).every(([key,value]) =>
    key === exclude || !value || c[key] === value);
  const selects = [...explorer.querySelectorAll('[data-dy-filter]')];
  const allLabels = {domain:'All scenarios', people:'All groups', viewpoint:'All viewpoints'};
  const environmentButtons = [...explorer.querySelectorAll('[data-dy-environment]')];
  function updatePaging() {
    const max = shelf.scrollWidth - shelf.clientWidth;
    el('prev').disabled = shelf.scrollLeft < 2;
    el('next').disabled = shelf.scrollLeft >= max - 2;
    explorer.querySelector('.dy-paging').hidden = max < 2;
  }
  function driftShelf(now) {
    const elapsed = lastAutoFrame === null ? 0 : Math.min((now-lastAutoFrame)/1000,.05);
    lastAutoFrame = now;
    const max = Math.max(0,shelf.scrollWidth-shelf.clientWidth);
    if (max < 2) { scheduleAuto();return; }
    // Preserve fractional progress: scrollLeft can round each frame on some browsers.
    autoPosition += autoDirection*autoSpeed*elapsed;
    if (autoPosition >= max) { autoPosition=max;autoDirection=-1; }
    else if (autoPosition <= 0) { autoPosition=0;autoDirection=1; }
    shelf.scrollTo({left:autoPosition,behavior:'instant'});
    autoFrame=requestAnimationFrame(driftShelf);
  }
  function scheduleAuto() {
    clearTimeout(autoTimer);
    cancelAnimationFrame(autoFrame);
    lastAutoFrame=null;
    const button = el('auto');
    const label = autoEnabled ? 'Pause automatic scrolling' : 'Start automatic scrolling';
    button.textContent = autoEnabled ? 'Ⅱ Pause' : '▶ Auto-scroll';
    button.setAttribute('aria-label',label);button.title=label;
    if (!autoEnabled || !shelfVisible || document.hidden || interacting ||
        (keyboardMode && catalog.contains(document.activeElement) && document.activeElement!==button) || !video.paused ||
        document.querySelector('dialog[open]') || shelf.scrollWidth-shelf.clientWidth<2) return;
    const delay=manualUntil-performance.now();
    if (delay>0) { autoTimer=setTimeout(scheduleAuto,delay);return; }
    autoPosition=shelf.scrollLeft;
    autoFrame=requestAnimationFrame(driftShelf);
  }
  function holdAuto(delay=3000) {
    manualUntil=performance.now()+delay;
    scheduleAuto();
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
    btn.append(shot,node('span',c.domain,'dy-tile-domain'),title,node('span',c.people,'dy-tile-people'));
    return btn;
  }
  function buildLibrary() {
    for (const button of environmentButtons) {
      const value = button.dataset.dyEnvironment;
      button.setAttribute('aria-pressed',String(filters.environment===value));
      const count = cases.filter(c=>matches(c,'environment')&&(!value||c.environment===value)).length;
      button.disabled = count===0;
      button.replaceChildren(document.createTextNode(value||'All episodes'),node('span',String(count)));
    }
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
    autoDirection = 1;
    el('count').textContent = visible.length + (visible.length===1?' episode':' episodes');
    el('reset').hidden = !Object.values(filters).some(Boolean);
    updatePaging();
    scheduleAuto();
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
  for (const button of environmentButtons) {
    button.addEventListener('click',()=>{
      filters.environment=button.dataset.dyEnvironment;
      applyFilters();
    });
  }
  function resetFilters() { for (const key of Object.keys(filters)) filters[key]=''; }
  el('reset').addEventListener('click',()=>{ resetFilters();applyFilters();selects[0].focus(); });
  function pageShelf(direction) {
    const tile = shelf.firstElementChild;
    holdAuto();
    autoDirection=direction;
    if (tile) {
      const gap=parseFloat(getComputedStyle(shelf).columnGap)||0;
      shelf.scrollBy({left:direction*(tile.getBoundingClientRect().width+gap),behavior:motion.matches?'instant':'smooth'});
    }
  }
  el('prev').addEventListener('click',()=>pageShelf(-1));
  el('next').addEventListener('click',()=>pageShelf(1));
  el('auto').addEventListener('click',()=>{
    autoEnabled=!autoEnabled;
    if(autoEnabled)manualUntil=0;
    scheduleAuto();
  });
  // Mouse focus is retained after a click. Only deliberate keyboard focus
  // suspends the carousel indefinitely; pointer interaction gets a short hold.
  document.addEventListener('keydown',e=>{
    if(['Tab','ArrowLeft','ArrowRight','Home','End'].includes(e.key)){keyboardMode=true;scheduleAuto();}
  });
  document.addEventListener('pointerdown',()=>{keyboardMode=false;scheduleAuto();},{capture:true});
  catalog.addEventListener('focusin',scheduleAuto);
  catalog.addEventListener('focusout',()=>requestAnimationFrame(scheduleAuto));
  catalog.addEventListener('pointerdown',()=>{interacting=true;holdAuto();});
  shelf.addEventListener('pointermove',e=>{
    if(e.pointerType==='mouse' && (e.movementX || e.movementY))holdAuto(1200);
  },{passive:true});
  for(const event of ['pointerup','pointercancel'])document.addEventListener(event,()=>{
    if(interacting){interacting=false;holdAuto();}
  });
  shelf.addEventListener('wheel',()=>holdAuto(),{passive:true});
  shelf.addEventListener('keydown',e=>{
    const choices=[...shelf.children], at=choices.indexOf(e.target);
    if(at<0 || !['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
    e.preventDefault();holdAuto();
    const next=e.key==='Home'?0:e.key==='End'?choices.length-1:Math.max(0,Math.min(choices.length-1,at+(e.key==='ArrowRight'?1:-1)));
    choices[next].focus({preventScroll:true});
    choices[next].scrollIntoView({block:'nearest',inline:'nearest',behavior:motion.matches?'instant':'smooth'});
  });
  shelf.addEventListener('scroll',updatePaging,{passive:true});
  new ResizeObserver(()=>{updatePaging();scheduleAuto();}).observe(shelf);
  new IntersectionObserver(([entry])=>{
    shelfVisible=entry.isIntersecting && entry.intersectionRatio>=.2;
    scheduleAuto();
  },{threshold:[0,.2,1]}).observe(shelf);
  document.addEventListener('visibilitychange',scheduleAuto);
  motion.addEventListener('change',()=>{if(motion.matches)autoEnabled=false;scheduleAuto();});
  const dialog=document.getElementById('viewer');
  if(dialog)new MutationObserver(scheduleAuto).observe(dialog,{attributes:true,attributeFilter:['open']});
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
    // Keep one row. Nearby cards move horizontally while their connectors stay
    // anchored to the actual timestamps on the proportional video timeline.
    const frameBox=el('frames'), width=el('track').clientWidth;
    if (!width || !frameBox.children.length) return;
    const c=current(), cards=[...frameBox.children], gap=10, connector=21;
    const cardWidth=Math.min(144,(width-gap*(cards.length-1))/cards.length);
    const boxes=cards.map((b,i)=>{
      const x=c.keyframes[i].time/c.duration*width;
      const left=Math.max(0,Math.min(width-cardWidth,x-cardWidth/2));
      b.style.width=cardWidth+'px';
      return {b,x,left};
    });
    for(let i=1;i<boxes.length;i++)boxes[i].left=Math.max(boxes[i].left,boxes[i-1].left+cardWidth+gap);
    boxes.at(-1).left=Math.min(boxes.at(-1).left,width-cardWidth);
    for(let i=boxes.length-2;i>=0;i--)boxes[i].left=Math.min(boxes[i].left,boxes[i+1].left-cardWidth-gap);
    const bottom=connector+Math.max(...cards.map(b=>b.offsetHeight));
    const svg=el('lines');svg.setAttribute('viewBox','0 0 '+width+' '+bottom);svg.replaceChildren();
    for(const {b,x,left} of boxes){
      const target=Math.max(left+8,Math.min(x,left+cardWidth-8));
      b.style.left=left+'px';b.style.top=connector+'px';
      const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');
      line.setAttribute('points',x+',0 '+x+',6 '+target+','+(connector-5)+' '+target+','+connector);
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
  function choose(i, scroll, play, userSelected=false) {
    if(!Number.isInteger(i)||!cases[i])return;
    // Choosing an episode starts a reading session. Only the explicit
    // Auto-scroll control can re-enable motion after this user action.
    if(userSelected){autoEnabled=false;scheduleAuto();}
    video.pause();index=i;lastWindow=null;
    const c=current();
    if(!matches(c)){resetFilters();buildLibrary();}
    explorer.querySelectorAll('[data-dy-case]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.dyCase)===i)));
    el('title').textContent=c.title;
    el('source').textContent=c.environment+' · '+(c.sourceType==='real'?'Real-world':'Synthetic')+' video excerpt';
    el('record').href='https://github.com/todayai-labs/omni-life/blob/main/'+c.recordPath;
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
    if(b.hasAttribute('data-dy-case'))choose(Number(b.dataset.dyCase),true,true,true);
    if(b.hasAttribute('data-dy-lang'))changeLanguage(b.dataset.dyLang);
  });
  el('seek').addEventListener('pointerdown',()=>video.pause());
  el('seek').addEventListener('input',e=>{
    const time=(Number(e.target.value)-current().sourceStart)/current().sourceRate;
    video.pause();
    seekClip(time);
  });
  el('play').addEventListener('click',toggle);video.addEventListener('click',toggle);
  el('audio').addEventListener('click',()=>{video.muted=!video.muted;updateControls();});
  el('fullscreen').addEventListener('click',()=>{if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});else if(video.parentElement.requestFullscreen)video.parentElement.requestFullscreen().catch(()=>{});else if(video.webkitEnterFullscreen)video.webkitEnterFullscreen();});
  ['loadedmetadata','seeked','timeupdate'].forEach(ev=>video.addEventListener(ev,updateState));
  video.addEventListener('play',()=>{updateControls();scheduleAuto();cancelAnimationFrame(raf);raf=requestAnimationFrame(tick);});
  for(const ev of ['pause','ended'])video.addEventListener(ev,()=>{cancelAnimationFrame(raf);updateState();updateControls();scheduleAuto();});
  // Resizing preserves the single row and its timestamp connectors.
  new ResizeObserver(layoutFrames).observe(el('track'));
  document.fonts?.ready.then(layoutFrames);
  video.controls=false;
  window.openDayCase=i=>choose(Number(i),true,true,true);
  buildLibrary();choose(0,false,false);
})();
