"""Offline viewer for actual temporal predictions and five-action policy."""

import json
from pathlib import Path


def save_crossing_viewer(episode, path):
    data = json.dumps(episode).replace("<", "\\u003c")
    html = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>EFI · Predictive crossing</title>
<style>
body{margin:2rem auto;padding:0 1rem;max-width:1100px;background:#101820;color:#e4eaf0;font:16px system-ui}
h1{font-size:1.7rem}p{color:#bfcbd6;line-height:1.5}.panels{display:flex;gap:24px;flex-wrap:wrap}
canvas{width:300px;max-width:85vw;background:#1d2a36;border-radius:8px;image-rendering:pixelated}
button,input{margin:12px 12px 12px 0}button{padding:8px 15px;cursor:pointer}#scrub{width:55%}
#policy{font-family:monospace;white-space:pre-wrap;line-height:1.8}.caption{min-height:2em}
</style>
<h1 id="title">EFI · Predictive crossing</h1>
<p id="description"></p>
<div class="panels">
<div><p id="worldlabel">World · blue agent, red hazard, green goal</p><canvas id="world" width="330" height="330"></canvas></div>
<div><p id="nearlabel">Hazard probability · next step</p><canvas id="near" width="330" height="330"></canvas></div>
<div><p id="farlabel"></p><canvas id="far" width="330" height="330"></canvas></div>
<div id="dangerpanel" style="display:none"><p>Hazard probability · next step</p><canvas id="danger" width="330" height="330"></canvas></div>
</div>
<button id="play">Play</button><button id="back">Previous</button><button id="next">Next</button>
<input id="scrub" type="range" min="0" value="0" aria-label="Episode step">
<p class="caption" id="caption"></p><div id="policy"></div>
<p id="explanation">The dashed square is the agent’s observation window. Forecast panels use its discovered map;
dark cells are unobserved. They show probabilities
generated before the next movement. These are learned forecasts, including their uncertainty.
The replay is a successful transfer trial with waiting; the complete experiment includes every trial.</p>
<script>
const data=EPISODE_DATA, frames=data.frames, names=['up','down','left','right','wait'];
const interception=data.kind==='interception', objectName=interception?'Target':'Hazard';
if(interception){
 document.title=document.getElementById('title').textContent='EFI · Transferred motion';
 document.getElementById('worldlabel').textContent='World · blue agent, green reward, red hazard';
 if(frames[0].hazard_forecast)document.getElementById('dangerpanel').style.display='block';
 document.getElementById('nearlabel').textContent='Target probability · next step';
 document.getElementById('explanation').textContent='The dashed square is the observation window. Forecast panels use the agent’s discovered map; dark cells are unobserved. Motion rules were learned while avoiding hazards, then frozen and reused here to intercept a reward. This is the first obstacle trial, selected before its outcome was known. The complete evaluation includes every trial.';
}
let index=0,timer=null;
const scrub=document.getElementById('scrub');scrub.max=frames.length-1;
document.getElementById('description').textContent=`Task: ${data.result.task||data.result.phase}. Seed ${data.result.seed}, trial ${data.result.episode+1}. `+
`Success: ${data.result.success}. Return: ${data.result.return.toFixed(2)}. `+(interception?'No motion-rule updates during this trial.':'Rules learned online before and during this trial.');
function draw(id,frame,field){
 const c=document.getElementById(id),ctx=c.getContext('2d'),H=frame.walls.length,W=frame.walls[0].length;
 const s=Math.min(c.width/W,c.height/H),ox=(c.width-W*s)/2,oy=(c.height-H*s)/2;
 ctx.clearRect(0,0,c.width,c.height);
 const walls=field?frame.known_walls:frame.walls;
 for(let y=0;y<H;y++)for(let x=0;x<W;x++){
  ctx.fillStyle=walls[y][x]?'#415364':field&&!frame.seen[y][x]?'#0b1118':'#18242f';ctx.fillRect(ox+x*s,oy+y*s,s-1,s-1);
  if(field&&!walls[y][x]){ctx.fillStyle=`rgba(${interception&&id!=='danger'?'49,204,137':'255,95,85'},${Math.max(0,Math.min(1,field[y][x]))})`;ctx.fillRect(ox+x*s,oy+y*s,s-1,s-1);}
 }
 function dot(p,color,r){ctx.beginPath();ctx.arc(ox+(p[1]+.5)*s,oy+(p[0]+.5)*s,r*s,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();}
 if(!field){dot(frame.target||frame.goal,'#31cc89',.35);if(frame.hazard)dot(frame.hazard,'#ff5f55',.35);}
 dot(frame.position,'#57b3ff',.26);
 if(!field){ctx.strokeStyle='#acd3f3';ctx.setLineDash([4,3]);const half=(data.config.win-1)/2;
 ctx.strokeRect(ox+(frame.position[1]-half)*s,oy+(frame.position[0]-half)*s,data.config.win*s,data.config.win*s);ctx.setLineDash([]);}
}
function render(){const f=frames[index];scrub.value=index;draw('world',f,null);draw('near',f,f.forecast[0]);draw('far',f,f.forecast[f.forecast.length-1]);
 if(f.hazard_forecast)draw('danger',f,f.hazard_forecast[0]);
 document.getElementById('farlabel').textContent=`${objectName} probability · ${f.forecast.length} steps ahead`;
 document.getElementById('caption').textContent=`Step ${f.step} · chosen action: ${names[f.action]}`+(interception?'':` · observed motion transitions learned: ${f.transitions}`);
 document.getElementById('policy').textContent=f.policy.map((p,i)=>`${names[i].padEnd(6)} ${(100*p).toFixed(1).padStart(5)}%  ${'█'.repeat(Math.round(p*30))}`).join('\n');}
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';}
document.getElementById('play').onclick=()=>{if(timer){stop();return;}if(index===frames.length-1)index=0;render();document.getElementById('play').textContent='Pause';timer=setInterval(()=>{if(index===frames.length-1){stop();return;}index++;render();},450);};
document.getElementById('back').onclick=()=>{stop();index=Math.max(0,index-1);render();};
document.getElementById('next').onclick=()=>{stop();index=Math.min(frames.length-1,index+1);render();};
scrub.oninput=()=>{stop();index=Number(scrub.value);render();};render();
</script></html>"""
    Path(path).write_text(html.replace("EPISODE_DATA", data))
