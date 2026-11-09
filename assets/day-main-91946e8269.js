'use strict';
(() => {
  const paperFigures = JSON.parse(document.getElementById('figure-data').textContent);
  const models = JSON.parse(document.getElementById('model-data').textContent);
  const scenes = [
    {offset:138, height:405, title:'Study focus', text:'A quick phone check can pass in silence. If the child stays distracted, the assistant should prompt them to return to their homework.'},
    {offset:551, height:402, title:'Safe navigation', text:'A traveler using a white cane approaches a puddle. A warning is useful as soon as the wet floor becomes visible, while the traveler can still avoid it.'},
    {offset:961, height:400, title:'Child monitoring', text:'The caregiver has already reached and comforted the child. For the task of alerting when help is needed, the expected response is silence.'}
  ];
  const sceneButtons = [...document.querySelectorAll('[data-scene]')];
  function showScene(index) {
    const scene = scenes[index];
    const panel = document.querySelector('.fig-panel');
    panel.style.aspectRatio = '1078 / ' + scene.height;
    const illustration = panel.querySelector('img');
    illustration.style.transform = 'translateY(-' + 100 * scene.offset / 1361 + '%)';
    illustration.alt = scene.title + ', Figure 1 example';
    document.getElementById('scene-copy').textContent = scene.text;
    sceneButtons.forEach((button,i) => button.setAttribute('aria-pressed',String(i===index)));
  }
  sceneButtons.forEach((button,i)=>button.addEventListener('click',()=>showScene(i)));
  showScene(0);
  const dialog = document.getElementById('viewer');
  const body = document.getElementById('viewer-body');
  const full = document.getElementById('full-figure');
  function openMedia(button) {
    document.querySelectorAll('video').forEach(video=>video.pause());
    const isFilm = button.hasAttribute('data-film');
    const item = document.createElement(isFilm ? 'video' : 'img');
    if (isFilm) {
      item.src=button.dataset.film || 'media/overview.mp4'; item.controls=true;item.playsInline=true;
      item.setAttribute('aria-label','Omni-Life project film'); full.hidden=true;
    } else {
      const figure=paperFigures[Number(button.dataset.figure)-1];
      item.alt=figure.alt;item.src=figure.src;full.href=figure.src;full.hidden=false;
    }
    body.replaceChildren(item); dialog.showModal();
    if(isFilm)item.play().catch(()=>{});
  }
  document.querySelectorAll('[data-film],[data-figure]').forEach(button => button.addEventListener('click',()=>openMedia(button)));
  document.getElementById('close-viewer').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
  dialog.addEventListener('close',()=>{const v=body.querySelector('video');if(v){v.pause();v.removeAttribute('src');v.load();}body.replaceChildren();});
  const metric=document.getElementById('metric');
  function showTable() {
    const table=document.getElementById('scores');
    table.replaceChildren(...[...models].sort((a,b)=>b[metric.value]-a[metric.value]).map(model=>{
      const row=document.createElement('tr');
      [model.model,...['full_correct','intervention_recall','intervention_f1','silence_success','joint_qa'].map(k=>model[k].toFixed(2))].forEach(s=>{
        const cell=document.createElement('td'); cell.textContent=s;row.append(cell);
      });return row;
    }));
  }
  metric.addEventListener('change',showTable);showTable();
})();
