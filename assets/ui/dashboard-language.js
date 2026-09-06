/* Display language only: IDs, API fields and reward event identifiers stay intact. */
let uiLanguage = localStorage.getItem('pkmai-language') || 'en';
const placeNames = {
 '3,0':['Pallet Town','Alabastia'], '3,1':['Viridian City','Vertania City'],
 '3,2':['Pewter City','Marmoria City'], '3,19':['Route 1','Route 1'],
 '3,20':['Route 2','Route 2'], '1,0':['Viridian Forest','Vertania-Wald'],
 '4,0':["Pallet Town · Red’s Home · F1",'Alabastia · Reds Haus · F1'],
 '4,1':["Pallet Town · Red’s Home · F2",'Alabastia · Reds Haus · F2'],
 '4,2':["Pallet Town · Rival’s House",'Alabastia · Haus des Rivalen'],
 '4,3':["Pallet Town · Professor Oak’s Lab",'Alabastia · Professor Eichs Labor'],
 '5,0':['Viridian City · House','Vertania City · Wohnhaus'],
 '5,1':['Viridian City · Gym','Vertania City · Arena'],
 '5,2':['Viridian City · School','Vertania City · Schule'],
 '5,3':['Viridian City · Poké Mart','Vertania City · Pokémon-Markt'],
 '5,4':['Viridian City · Pokémon Center · Ground floor','Vertania City · Pokémon-Center · Erdgeschoss'],
 '5,5':['Viridian City · Pokémon Center · Upstairs','Vertania City · Pokémon-Center · Obergeschoss']
};
// Map groups verified against pret/pokefirered data/maps/map_groups.json.
function placeName(bank, map) {
 const names = placeNames[`${Number(bank)},${Number(map)}`];
 if(names) return names[uiLanguage === 'de' ? 1 : 0];
 return uiLanguage === 'de' ? `Gebiet ${Number(bank)+1} · Gebäude ${Number(map)+1}` : `Area ${Number(bank)+1} · Building ${Number(map)+1}`;
}
const translations = [
 ['Noch keine Move-Telemetrie verfügbar.','No move data available yet.'],
 ['noch keine 1-Tile-Schritte','No tile movement recorded yet'],
 ['Noch keine Telemetrie','No telemetry yet'],
 ['noch kein world_depth-Event in diesem Zyklus','No world progress event in this cycle yet'],
 ['Champion: beste Full-Steps','Champion: fastest complete run'],
 ['Eigene K.O. (Party wiped)','Team defeats (party wiped)'],
 ['Gegner-K.O.','Opponent defeats'],['Schaden-HP','Damage dealt (HP)'],
 ['letzter Abbruch:','last reset:'],['im Kampf','in battle'],['Lauf-Steps','Travel steps'],
 ['Lernkurve','Learning curve'],['Spiel-Fortschritt','Game progress'],
 ['Festfahren / Anti-Loop','Stalling / loop prevention'],
 ['Erster Orden','First badge'],['Pokédex / Paket abgegeben','Pokédex / parcel delivered'],
 ['Weltstufe','World stage'],['Treppe','Stairs'],['Haus-Exit','Leaving home'],
 ['Hausausgang','Leaving home'],['abgeschlossen','completed'],['gesamte','total'],
 ['gesamt','total'],['Läufe','Runs'],['Lauf','Run'],
 ['Episoden','Episodes'],['Ø Ep-Steps','Average episode steps'],
 ['Erfolgsrate','Success rate'],['Tiefe','Depth'],['Rolle','Role'],['Rollen','Roles'],

 ['Ø Episode-Reward über echte PPO-Trainingsschritte.','Average episode reward across actual PPO training steps.'],
 ['Nur vollständige Runs vom echten Spielanfang: Intro, Treppe, Hausausgang, Schiggi und Orden.','Complete runs: intro, stairs, leaving home, starter and badges.'],
 ['Bestes Level, Orden und Maps je Modellstand.','Best level, badges and locations per model version.'],
 ['Loops pro 100 echte Beginning-Runs; Curriculum wird separat gezählt.','Loops per 100 complete runs; curriculum is counted separately.'],
 ['Welt-Tiefe (Außen-Maps)','World progress (outdoor locations)'],
 ['Bestes Party-Level','Highest team level'],['Tiefen-Durchbrüche','Progress breakthroughs'],
 ['Kämpfe ges.','Total battles'],['draußen','outside'],

 ['Noch keine Indoor-Tiles seit dem Mapping-Reset entdeckt.','No indoor areas explored yet.'],
 ['Klick auf Agent = nur diesen anzeigen · nochmal klicken = alle.','Select an agent to focus · select again to show all.'],
 ['Watcher-Telemetrie läuft immer weiter.','Watcher telemetry remains live.'],
 ['32 abgeschlossene Full-Runs · nur bessere Candidates werden Champion','32 completed runs · only better candidates become champion'],
 ['Klick auf Team-Slot für Details','Select a team member for details'],
 ['Noch kein Pokémon / keine Party-Telemetrie','No Pokémon / no team data yet'],
 ['Noch keine Daten','No data yet'], ['Agenten – antippen für Live-Stats','Agents – select for live stats'],
 ['End-to-End-Screenshot + Live-Stats jedes Agenten','Live game view and agent statistics'],
 ['Alle Rollen','All roles'],['Alle Maps','All locations'],['Alle Stages','All stages'],
 ['Starter egal','Any starter'],['hat Starter','Has starter'],['kein Starter','No starter'],
 ['Sortierung: Standard','Sort: default'],['Weitester Fortschritt','Furthest progress'],
 ['Meiste Maps','Most locations'],['Höchstes Level','Highest level'],['Höchster Reward','Highest reward'],
 ['Weitere Wege: aus','Extra paths: off'],['Weitere Wege: an','Extra paths: on'],
 ['Zeige Agenten:','Show agents:'],['Lade 35 Agenten...','Loading agents...'],
 ['Letztes Reward-Event','Latest reward event'],['Reward-Verlauf','Reward history'],
 ['Steps / Lernfortschritt','Steps / learning progress'],['Letzte 1-Tile-Schritte','Recent tile steps'],
 ['warte auf Route','Waiting for movement'],['Agent anklicken','Select an agent'],
 ['Seit Champion','Since champion'],['bestätigt','confirmed'],['nicht übernommen','not promoted'],
 ['Flotten-Status','Fleet status'],['FLOTTEN-STATUS','FLEET STATUS'],
 ['Live aus den Agenten','Live agent data'],['Welt-Tiefe','World progress'],
 ['tiefster Checkpoint','furthest checkpoint'],['Fortschritt','Progress'],
 ['Kategorien – aktuelle Episoden','Categories – current episodes'],
 ['Einzelne Agenten','Individual agents'],['Kategorien','Categories'],
 ['anklicken für Live-Stats + Reward-Events','select for live stats and reward events'],
 ['wird das Netz wirklich besser?','is the policy improving?'],
 ['Erkundung','Exploration'],['Schritte','Steps'],['Weg-Steps','Travel steps'],
 ['Kampf-Steps','Battle steps'],['Weg','Travel'],['Kämpfe','Battles'],['Kampf','Battle'],
 ['Orden','Badges'],['Felder','Tiles'],['Kanten','Edges'],['erkundet','Explored'],
 ['begehbare Kante','Walkable path'],['Gebäude','Building'],['Gebiet','Area'],
 ['Spielanfang','Game start'],['Alabastia','Pallet Town'],['Vertania-Wald','Viridian Forest'],
 ['Vertania','Viridian'],['Marmoria','Pewter'],['Eichs Paket','Oak’s parcel'],
 ['Paketabgabe','Parcel delivered'],['Paket','Parcel'],['außen','outside'],['innen','inside'],
 ['Agenten','Agents'],['Alle','All'],['Laden','Loading'],['Lade','Loading'],
 ['Gesamt','Total'],['Durchschnitt','Average'],['Letzter','Last'],['Letzte','Last'],
 ['Aktuell','Current'],['aktuell','current'],['Abbruchgrund','Reset reason'],
 ['Erfolg','Success'],['Fehler','Error'],['keine','none'],['Keine','None'],
 ['jetzt','now'],['seit vorhin','since earlier'],['Global entdeckt','Globally discovered'],
 ['Bester Live-Reward jetzt','Best live reward now'],
 ['Ø Live-Reward jetzt','Avg live reward now'],
 ['Höchstes Level jetzt','Highest level now'],
 ['Bestes Party-Level','Best team level'],['Bestes Level','Best level'],
 ['Kämpfe gesamt','Battles total'],
 ['Leer / keine Party-Telemetrie','empty / no team data'],
 ['keine Party-Telemetrie','no team data'],['Leer','Empty'],
 ['Eich-Szene','Oak scene'],['Taste','Input'],['Episode-Reward','Episode reward'],
 ['Instanzen','Instances'],['Letzte 10 Rewards','Last 10 rewards'],
 ['Schiggi','Squirtle'],['Bisasam','Bulbasaur'],['Glumanda','Charmander'],

 /* V18: UI strings that stayed German in EN mode. Full phrases first so the
    length-sorted pass replaces them before the single-word pairs above. */
 ['Zusätzliche echte Nachbarschritte der Savestate-Runner anzeigen','Show the savestate runners’ extra real neighbour steps'],
 ['Ein-/ausklappen (Karte freigeben)','Collapse / expand (free the map)'],
 ['Mapper ist angehalten oder startet gerade.','Mapper is paused or still starting up.'],
 ['Spielanfang / Alabastia-Innen','Game start / indoors'],
 ['Alabastia (außen)','Pallet Town (outdoor)'],
 ['Agents mit Stats','Agents with stats'],
 ['Battles gestartet','Battles started'],['Battles beendet','Battles finished'],
 ['Kämpfe s/f','Battles start/done'],
 ['Gesammelte Orden','Badges collected'],
 ['noch keine Events','No events yet'],
 ['neue Felder','new tiles'],['Bild-Tiles','image tiles'],['Map unbekannt','Map unknown'],
 ['Erstes Pokémon','First Pokémon'],['Erfolge','successes'],
 ['Live-Weltkarte','Live world map'],
 ['Welt-Stufe','World stage'],
 ['Netz nachgeladen','Policy reloads'],
 ['Haus-Exit','Leaving home'],['Haus Exit','Leaving home'],
 [' aktiv',' active'],
 ['Kanto Orden','Kanto Badges'],
 ['Felsorden','Boulder Badge'],['Quellorden','Cascade Badge'],['Donnerorden','Thunder Badge'],
 ['Farborden','Rainbow Badge'],['Seelenorden','Soul Badge'],['Sumpforden','Marsh Badge'],
 ['Vulkanorden','Volcano Badge'],['Erdorden','Earth Badge']
];
const englishToGerman = [
 ['Overworld Map','Weltkarte'],['Indoor Mapping','Gebäudekarten'],['Graphs','Diagramme'],
 ['LIVE WATCHER','LIVE WATCHER'],['TRAINER · LIVE','TRAINER · LIVE'],
 ['FRONTIER CHAMPION','FORTSCHRITTS-CHAMPION'],['GLOBAL AI','GEMEINSAME KI'],
 ['Selected Agent Team','Team des gewählten Agenten'],['Episode Reward','Episoden-Reward'],
 ['Known Edges','Bekannte Wege'],['Known Maps','Bekannte Orte'],['Transitions','Übergänge'],
 ['Finished','Beendet'],['Learning','Lernen'],['Maps','Orte'],['Steps','Schritte'],
 ['Battles','Kämpfe'],['Battle','Kampf'],['Reward history','Reward-Verlauf']
];
const originals = new WeakMap();
const attributeOriginals = new WeakMap();
function translated(value) {
 let text = value;
 // Replace raw RAM addresses with visitor-facing location names wherever they appear.
 text = text.replace(/Bank\s+(\d+)\s*\/?\s*Map\s+(\d+)/g,(_,b,m)=>placeName(b,m));
 const pairs = uiLanguage === 'en' ? translations : englishToGerman;
 for(const [from,to] of [...pairs].sort((a,b)=>b[0].length-a[0].length)) text = text.split(from).join(to);
 return text;
}
function translatePage() {
 observer.disconnect();
 const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
 let node;
 while((node=walker.nextNode())) {
  try {
   if(!node.parentElement || node.parentElement.closest('script,style,#language-toggle')) continue;
   const previous=originals.get(node);
   const original=previous && previous.rendered===node.nodeValue ? previous.original : node.nodeValue;
   const rendered=translated(original);
   if(rendered!==node.nodeValue) node.nodeValue=rendered;
   originals.set(node,{original,rendered});
  } catch(_) { /* never let one bad node stop the whole pass */ }
 }
 for(const el of document.querySelectorAll('[title],[aria-label],[placeholder]')) {
  if(el.id==='language-toggle')continue;
  const saved=attributeOriginals.get(el)||{};
  for(const key of ['title','aria-label','placeholder']) {
   if(!el.hasAttribute(key))continue;
   const current=el.getAttribute(key), old=saved[key];
   const original=old&&old.rendered===current ? old.original : current;
   const rendered=translated(original);
   if(current!==rendered)el.setAttribute(key,rendered);
   saved[key]={original,rendered};
  }
  attributeOriginals.set(el,saved);
 }
 document.documentElement.lang=uiLanguage;
 const tgl=document.getElementById('language-toggle');
 const tglText=uiLanguage==='en' ? 'EN / DE' : 'DE / EN';
 if(tgl && tgl.textContent!==tglText) tgl.textContent=tglText;
 observer.observe(document.body,{subtree:true,childList:true,characterData:true});
}
let inTranslate=false;
const observer=new MutationObserver(()=>{
 // Run synchronously in the observer microtask (no setTimeout / rAF): the DOM
 // is retranslated before the browser paints, so a panel re-rendered in German
 // by one of the dashboard's refreshers never flashes. translated() output is
 // stable (no de->en pair matches an English result), so setting nodeValue
 // here produces no further mutations and cannot loop. rAF was wrong (paused
 // in background tabs -> translation silently stopped); setTimeout was wrong
 // (macrotask -> a visible German flicker on every refresh).
 if(inTranslate)return;
 inTranslate=true;
 try{translatePage();}catch(_){}
 inTranslate=false;
});
// Safety net: the observer is disconnected while translatePage() runs, so a
// panel re-rendered by one of the many setInterval refreshers during that
// window can stay untranslated until the next unrelated mutation. A periodic
// pass guarantees those sections get caught even if an event is missed.
setInterval(()=>{try{translatePage();}catch(_){}}, 700);
function setLanguage(language) {
 uiLanguage=language;
 localStorage.setItem('pkmai-language',language);
 if(typeof latestInstances!=='undefined') latestInstances.forEach(i=>{i.room=placeName(i.bank,i.map);});
 if(typeof renderIndoorMapping==='function')renderIndoorMapping();
 if(typeof renderSelectedAgent==='function')renderSelectedAgent();
 if(typeof Chart!=='undefined')Object.values(Chart.instances||{}).forEach(chart=>chart.update('none'));
 translatePage();
}
window.addEventListener('DOMContentLoaded',()=>{
 document.getElementById('language-toggle').onclick=()=>setLanguage(uiLanguage==='en'?'de':'en');
 translatePage();
});
// Canvas labels are not DOM text; keep chart legends and axis titles in sync.
window.addEventListener('DOMContentLoaded',()=>{
 if(typeof Chart==='undefined')return;
 Chart.register({id:'dashboardLanguage',beforeUpdate(chart){
  for(const dataset of chart.data.datasets||[]) {
   if(!dataset.label)continue;
   const previous=dataset._languageLabel;
   const original=previous&&previous.rendered===dataset.label ? previous.original : dataset.label;
   const rendered=translated(original);
   dataset.label=rendered;
   dataset._languageLabel={original,rendered};
  }
 }});
});

// Live fleet tools use the existing uncached UI asset: no web/watch restart.
function fleetRole(agent) {
 if (Number(agent.id) === 120) return 'watcher';
 const modes = ['FULL','BRIDGE','FRONTIER','RETENTION','FIGHTER'];
 const mode = String(agent.training_mode || '').toUpperCase();
 if (modes.includes(mode)) return mode.toLowerCase();
 const named = String(agent.name || '').match(/\b(FULL|BRIDGE|FRONTIER|RETENTION|FIGHTER)\b/);
 return named ? named[1].toLowerCase() : String(agent.training_objective || agent.agent_role || 'unknown').toLowerCase();
}
function fleetHealth(agent) {
 const party = Array.isArray(agent.party) ? agent.party : [];
 const hp = party.reduce((n,m)=>n+Number(m.cur_hp||0),0);
 const max = party.reduce((n,m)=>n+Number(m.max_hp||0),0);
 return {hp,max,alive:party.filter(m=>Number(m.cur_hp)>0).length,total:party.length,
         critical:party.some(m=>Number(m.max_hp)>0 && Number(m.cur_hp)/Number(m.max_hp)<0.3)};
}
function fleetMatches(agent, filter) {
 const h = fleetHealth(agent);
 if (filter.role && fleetRole(agent)!==filter.role) return false;
 if (filter.health==='critical' && !h.critical) return false;
 if (filter.health==='battle' && !agent.in_battle) return false;
 if (filter.health==='checkpoint' && (!agent.episode_start || agent.episode_start==='beginning')) return false;
 const q=String(filter.query||'').trim().toLowerCase();
 const id=q.match(/^(?:a|agent\s*)?(\d+)$/);
 if(id) return Number(agent.id)===Number(id[1]);
 return !q || [agent.name,fleetRole(agent),agent.room,agent.episode_start,
     `${agent.bank}/${agent.map}`].join(' ').toLowerCase().includes(q);
}
window.addEventListener('DOMContentLoaded',()=>{
 if(typeof renderStatusDashboard!=='function') return;
 const filter={role:'',health:'',query:''};
 let lastState=null;
 const labels={full:'Full',bridge:'Bridge',frontier:'Frontier',retention:'Retention',fighter:'Fighter',watcher:'Watcher',scout:'Scout (Legacy)'};
 Object.assign(FLEET_ROLE_LABELS,labels);
 Object.assign(STATUS_ROLE_ICONS,{bridge:'🌉',frontier:'🔭',retention:'🔁',fighter:'⚔️'});
 const displayAgent=i=>({...i,training_objective:fleetRole(i)});
 const oldStatus=renderStatusDashboard, oldChips=agentChipsHtml, oldDetail=renderAgentDetail;
 const esc=statusEsc;
 const style=document.createElement('style');
 style.textContent='.fleet-tools{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.fleet-tools input,.fleet-tools select,.fleet-tools button{background:#171e2b;color:#eef3ff;border:1px solid #3a4965;border-radius:7px;padding:9px;font:inherit}.fleet-tools input{flex:1;min-width:140px}.fleet-health{font-size:11px;line-height:1.7;border-top:1px solid #303a4c;margin-top:8px;padding-top:6px;color:#b4c4dc}.fleet-health.critical{color:#ff8686}.fleet-results{font-size:12px;color:#9cafc7;margin:8px 0}.fleet-role-help{color:#a8bad1;font-size:12px;line-height:1.7;margin:10px 0}';
 document.head.appendChild(style);
 function refresh(){
  if(lastState) renderStatusDashboard(lastState);
  renderWatcherTab();
  document.querySelectorAll('.fleet-tools').forEach(bar=>{
   bar.querySelector('[data-key="role"]').value=filter.role;
   bar.querySelector('[data-key="health"]').value=filter.health;
   const input=bar.querySelector('input');if(input!==document.activeElement)input.value=filter.query;
  });
 }
 for(const [anchorId,scope] of [['status-agent-detail','status'],['wt-grid','watcher']]){
  const anchor=document.getElementById(anchorId);if(!anchor)continue;
  const bar=document.createElement('div');bar.className='fleet-tools';bar.id=scope+'-fleet-filters';
  bar.innerHTML='<input aria-label="Agent suchen" placeholder="ID, Name, Map oder Startpunkt" data-key="query">'
   +'<select aria-label="Rolle filtern" data-key="role"><option value="">Alle Rollen</option>'
   +Object.entries(labels).filter(([r])=>r!=='scout').map(([r,n])=>`<option value="${r}">${n}</option>`).join('')+'</select>'
   +'<select aria-label="Zustand filtern" data-key="health"><option value="">Alle Zustände</option><option value="critical">Team angeschlagen</option><option value="battle">Im Kampf</option><option value="checkpoint">Ab Savestate</option></select><button type="button">Zurücksetzen</button>';
  bar.addEventListener('input',event=>{if(event.target.dataset.key){filter[event.target.dataset.key]=event.target.value;refresh();}});
  bar.querySelector('button').onclick=()=>{Object.assign(filter,{role:'',health:'',query:''});refresh();};
  anchor.before(bar);
  const count=document.createElement('div');count.id=scope+'-fleet-results';count.className='fleet-results';anchor.before(count);
 }
 function healthHtml(i){
  const h=fleetHealth(i),start=i.episode_start_health||{};
  const startText=start.party_max_hp ? `${start.party_hp}/${start.party_max_hp} KP` : 'noch nicht gemeldet';
  return `<div class="fleet-health ${h.critical?'critical':''}">♥ Team: ${h.hp}/${h.max} KP · ${h.alive}/${h.total} einsatzfähig<br>💾 Start: ${esc(i.episode_start||'–')} · Start-Team: ${esc(startText)}</div>`;
 }
 renderStatusDashboard=function(state){
  lastState=state;
  oldStatus({...state,instances:(state.instances||[]).map(displayAgent)});
  const rows=(state.instances||[]).filter(i=>Number(i.id)!==120&&Number(i.id)>=0);
  let visible=0;
  document.querySelectorAll('#status-agent-grid [data-aid]').forEach(card=>{
   const row=rows.find(i=>Number(i.id)===Number(card.dataset.aid));if(!row)return;
   card.hidden=!fleetMatches(row,filter);if(!card.hidden)visible++;
   card.insertAdjacentHTML('beforeend',healthHtml(row));
  });
  const count=document.getElementById('status-fleet-results');if(count)count.textContent=`${visible} / ${rows.length} Agenten`;
  const cards=document.querySelectorAll('#status-role-grid .status-role');
  cards.forEach(card=>{const text=card.textContent.toLowerCase();const role=Object.keys(labels).find(r=>text.includes(labels[r].toLowerCase()));if(role){card.style.cursor='pointer';card.onclick=()=>{filter.role=role;refresh();};}});
 };
 agentChipsHtml=function(list){return oldChips(list.filter(i=>fleetMatches(i,filter)).map(displayAgent));};
 renderAgentDetail=function(id){
  oldDetail(id);
  const box=document.getElementById(id),row=(latestInstances||[]).find(i=>Number(i.id)===wtSelected);
  if(!box||box.hidden||!row)return;
  const sub=box.querySelector('.wt-sub');if(sub)sub.textContent=`${labels[fleetRole(row)]||fleetRole(row)} · ${row.room||''} · Start: ${row.episode_start||'–'}`;
  box.insertAdjacentHTML('beforeend',healthHtml(row));
 };
 const oldWatcher=renderWatcherTab;
 renderWatcherTab=function(){oldWatcher();const rows=latestInstances||[];const text=`${rows.filter(i=>fleetMatches(i,filter)).length} / ${rows.length} Agenten`;for(const id of ['wt-count','watcher-fleet-results']){const el=document.getElementById(id);if(el)el.textContent=text;}};
 const oldPass=agentPassesFilter;
 agentPassesFilter=i=>oldPass(displayAgent(i));
 const roleSelect=document.getElementById('af-role');
 if(roleSelect){const value=roleSelect.value;roleSelect.innerHTML='<option value="">Alle Rollen</option>'+Object.entries(labels).map(([r,n])=>`<option value="${r}">${n}</option>`).join('');roleSelect.value=value;}
 const help=document.createElement('div');help.className='fleet-role-help';
 help.textContent='Full: Gesamtweg · Bridge: unsicheren Übergang üben · Frontier: neue Wege entdecken · Retention: gelernte Übergänge erhalten · Fighter: Kämpfe üben. Alle Rollen trainieren dasselbe Netz.';
 document.getElementById('status-role-grid').before(help);
 if(typeof updateDashboard==='function')updateDashboard();
});
