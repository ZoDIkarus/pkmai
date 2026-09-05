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
 ['gesamt','total'],['Beste','Best'],['beste','best'],['Läufe','Runs'],['Lauf','Run'],
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
 ['Schiggi','Squirtle'],['Bisasam','Bulbasaur'],['Glumanda','Charmander']
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
  if(node.parentElement.closest('script,style,#language-toggle')) continue;
  const previous=originals.get(node);
  const original=previous && previous.rendered===node.nodeValue ? previous.original : node.nodeValue;
  const rendered=translated(original);
  if(rendered!==node.nodeValue) node.nodeValue=rendered;
  originals.set(node,{original,rendered});
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
 document.getElementById('language-toggle').textContent=uiLanguage==='en' ? 'EN / DE' : 'DE / EN';
 observer.observe(document.body,{subtree:true,childList:true,characterData:true});
}
let translationPending=false;
const observer=new MutationObserver(()=>{
 if(translationPending)return;
 translationPending=true;
 requestAnimationFrame(()=>{translationPending=false;translatePage();});
});
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
