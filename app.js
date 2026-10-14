(function(){
  'use strict';
  var progress=document.querySelector('.scroll-progress');
  function updateProgress(){var max=document.documentElement.scrollHeight-window.innerHeight;progress.style.width=(max>0?(window.scrollY/max)*100:0)+'%';}
  window.addEventListener('scroll',updateProgress,{passive:true}); updateProgress();

  var toggle=document.querySelector('.nav-toggle'), nav=document.querySelector('.primary-nav');
  if(toggle){toggle.addEventListener('click',function(){var open=nav.classList.toggle('open');toggle.setAttribute('aria-expanded',String(open));});}
  document.querySelectorAll('.primary-nav a').forEach(function(link){link.addEventListener('click',function(){nav.classList.remove('open');if(toggle)toggle.setAttribute('aria-expanded','false');});});

  document.querySelectorAll('.path-tab').forEach(function(tab){
    tab.addEventListener('click',function(){
      var path=tab.getAttribute('data-path');
      document.querySelectorAll('.path-tab').forEach(function(item){var active=item===tab;item.classList.toggle('active',active);item.setAttribute('aria-selected',String(active));});
      document.querySelectorAll('.path-panel').forEach(function(panel){var active=panel.id==='panel-'+path;panel.classList.toggle('active',active);panel.hidden=!active;});
    });
    tab.addEventListener('keydown',function(e){if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();var tabs=Array.from(document.querySelectorAll('.path-tab'));var i=tabs.indexOf(tab);var next=tabs[(i+(e.key==='ArrowDown'?1:-1)+tabs.length)%tabs.length];next.focus();next.click();}});
  });

  document.querySelectorAll('.workflow-step').forEach(function(step){step.tabIndex=0;function activate(){document.querySelectorAll('.workflow-step').forEach(function(x){x.classList.remove('active');});step.classList.add('active');}step.addEventListener('click',activate);step.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}});});

  function copyText(text){if(navigator.clipboard){return navigator.clipboard.writeText(text);}var area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove();return Promise.resolve();}
  document.querySelectorAll('.copy-button').forEach(function(btn){btn.addEventListener('click',function(){copyText(btn.getAttribute('data-copy')||'').then(function(){var toast=btn.closest('.code-window')?btn.closest('.code-window').querySelector('.copy-toast'):null;if(toast){toast.classList.add('show');setTimeout(function(){toast.classList.remove('show');},1400);}var label=btn.querySelector('.copy-label');if(label){var old=label.textContent;label.textContent='✓';setTimeout(function(){label.textContent=old;},1400);}});});});
})();