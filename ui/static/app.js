const state = { reports: [], project: 'all', reading: 'all', selected: null };
const readKey = 'textjepa-read-reports-v1';
const readMap = () => JSON.parse(localStorage.getItem(readKey) || '{}');
const isRead = r => readMap()[r.id] === r.hash;
function setRead(r, value) { const m=readMap(); if(value)m[r.id]=r.hash; else delete m[r.id]; localStorage.setItem(readKey,JSON.stringify(m)); }
const esc = s => String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load() {
  const data = await fetch('/api/reports').then(r=>r.json()); state.reports=data.reports;
  const sync=data.status.sync||{}; document.querySelector('#sync').textContent=sync.ok ? `Synced ${sync.at||'now'} · ${state.reports.length} reports` : `Sync problem: ${sync.message}`;
  const rounds=Object.values(data.status.controller?.rounds||{}); const active=rounds.filter(x=>x.state==='ACTIVE').length;
  document.querySelector('#controller').innerHTML=`<strong>Autonomy status</strong><br>${active} active round${active===1?'':'s'}<br>${data.status.controller?.paused?'Paused':'Controller available'}`;
  renderProjects(); renderTimeline();
}
function renderProjects(){
  const counts={all:state.reports.filter(r=>!isRead(r)).length}; state.reports.forEach(r=>counts[r.project]=(counts[r.project]||0)+(!isRead(r)));
  document.querySelector('#projects').innerHTML=Object.keys(counts).sort((a,b)=>a==='all'?-1:b==='all'?1:a.localeCompare(b)).map(p=>`<button class="project-button ${state.project===p?'active':''}" data-project="${esc(p)}"><span>${p==='all'?'All projects':esc(p.replaceAll('-',' ').replaceAll('_',' '))}</span><span class=badge>${counts[p]}</span></button>`).join('');
  document.querySelectorAll('[data-project]').forEach(b=>b.onclick=()=>{state.project=b.dataset.project;renderProjects();renderTimeline();});
}
function filtered(){return state.reports.filter(r=>(state.project==='all'||r.project===state.project)&&(state.reading==='all'||!isRead(r)));}
function renderTimeline(){
  document.querySelector('#reader').hidden=true; document.querySelector('#welcome').hidden=false; document.querySelector('#timeline').hidden=false;
  const rows=filtered(); document.querySelector('#timeline').innerHTML=rows.length?rows.map(r=>`<div class="report-card ${isRead(r)?'':'unread'}" data-id="${esc(r.id)}"><div><div class=date>${esc(new Date(r.created_at).toLocaleDateString())}</div><span class=project-pill>${esc(r.project)}</span></div><div><h3>${esc(r.title)}</h3><p>${esc(r.plain_summary)}</p></div><div><span class=status-pill>${esc(r.status)}</span><br><span class=unread-dot>${isRead(r)?'Read':'● Unread'}</span></div></div>`).join(''):'<div class=welcome><h2>You are caught up.</h2><p>No reports match this view.</p></div>';
  document.querySelectorAll('.report-card').forEach(x=>x.onclick=()=>openReport(x.dataset.id));
}
async function openReport(id){
  const data=await fetch('/api/report?id='+encodeURIComponent(id)).then(r=>r.json()); const r=data.report; state.selected=r;
  document.querySelector('#welcome').hidden=true;document.querySelector('#timeline').hidden=true;document.querySelector('#reader').hidden=false;
  const artifacts=(r.artifacts||[]).map(a=>`<a href="/files/${encodeURIComponent(r.bundle+'/'+a)}" target="_blank">Open ${esc(a.split('/').pop())}</a>`).join(' · ');
  document.querySelector('#reportMeta').innerHTML=`<span class=eyebrow>${esc(r.project)} · ${esc(r.status)}</span><h1>${esc(r.title)}</h1><p class=meta-summary>${esc(r.plain_summary)}</p><p><strong>Decision:</strong> ${esc(r.decision)}</p>${artifacts?`<p class=meta-summary>${artifacts}</p>`:''}`;
  document.querySelector('#reportBody').innerHTML=data.html; document.querySelector('#raw').href='/files/'+encodeURIComponent(r.markdown_path); updateReadButton(); window.scrollTo(0,0);
}
function updateReadButton(){document.querySelector('#readToggle').textContent=isRead(state.selected)?'Mark as unread':'Mark as read';}
document.querySelector('#back').onclick=()=>{renderProjects();renderTimeline();};
document.querySelector('#readToggle').onclick=async()=>{
  const value=!isRead(state.selected);
  const response=await fetch('/api/ack',{method:'POST',headers:{'Content-Type':'application/json','X-TextJEPA-UI':'1'},body:JSON.stringify({report_id:state.selected.id,hash:state.selected.hash,read:value})});
  const data=await response.json(); if(!data.ok){alert(data.error);return;} setRead(state.selected,value);updateReadButton();renderProjects();
};
document.querySelector('#nextUnread').onclick=()=>{const r=state.reports.slice().reverse().find(x=>!isRead(x)); if(r)openReport(r.id);};
document.querySelectorAll('input[name=reading]').forEach(x=>x.onchange=()=>{state.reading=x.value;renderTimeline();});
document.querySelector('#sendSteering').onclick=async()=>{
  const message=document.querySelector('#steeringText').value,status=document.querySelector('#steeringStatus'); status.textContent=' Sending…';
  const response=await fetch('/api/steer',{method:'POST',headers:{'Content-Type':'application/json','X-TextJEPA-UI':'1'},body:JSON.stringify({project:state.selected.project,report_id:state.selected.id,message})}); const data=await response.json(); status.textContent=data.ok?' Sent. The next project oversight will read it.':' '+data.error;
};
load(); setInterval(load,60000);
