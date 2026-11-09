(()=>{
  const hero = document.querySelector('.dl-hero');
  const data = document.getElementById('case-data');
  if (!hero || !data) return;
  const cases = JSON.parse(data.textContent);
  const summaries = {
    elder_fall2_cam: 'Notify the caregiver if danger arises.',
    ask_school_word: 'Respond when the student asks a question.',
    adult_gas_neg: 'Alert the user if they leave the burner on.'
  };
  const choices = [...hero.querySelectorAll('[data-dl-pick]')];
  const controls = [...hero.querySelectorAll('[data-dl-play]')];
  let selectedIndex = 0;
  function select(id) {
    const index = cases.findIndex(entry => entry.id === id);
    if (index < 0) return;
    selectedIndex = index;
    const entry = cases[index];
    hero.querySelector('#dl-case-name').textContent = entry.title;
    hero.querySelector('#dl-case-summary').textContent = summaries[id] || entry.task.en;
    hero.querySelector('.dl-cover img').src = entry.poster;
    choices.forEach(choice => choice.setAttribute('aria-pressed',String(choice.dataset.dlPick === id)));
    controls.forEach(control => control.setAttribute('aria-label','Play '+entry.title));
  }
  choices.forEach(choice => choice.addEventListener('click',()=>select(choice.dataset.dlPick)));
  controls.forEach(control => control.addEventListener('click',()=>window.openDayCase(selectedIndex)));
  select('elder_fall2_cam');
})();
