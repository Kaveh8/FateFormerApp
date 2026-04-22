"""Flux Analysis: metabolic map with searchable side panel."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from streamlit_hf.lib import io
from streamlit_hf.lib import ui

ui.inject_app_styles()

_NAR_FATEFORMER_URL = "https://academic.oup.com/nar/article/51/W1/W180/7175334"

_HELP_MET_MAP = f"""
**Figure (paper):** Network model of key metabolic pathways linked to fate outcomes identified by the model. Important pathways and reactions are mapped onto the **scFLUX** metabolic network schema. **Arrow colour** shows the **log₂ fold change** in **scFEA**-inferred flux between **reprogramming** and **dead-end** cells: **red** = higher flux in reprogramming, **blue** = higher in dead-end. **Black** arrows = no corresponding scFEA entry or no measurable flux difference. **Triple-star** markers in the figure denote **p_adj < 0.001** (two-sample *t*-test with Benjamini–Hochberg correction). Full article: [{_NAR_FATEFORMER_URL}]({_NAR_FATEFORMER_URL})

**In this explorer:** The same schematic is **interactive**: **metabolites** on the map link to the reconstruction. The **sidebar** ranks metabolites by the **strongest associated flux reaction** in this deployment (**#1** = top). **Search** the list (every word must match somewhere in that row). **Hover** labels for a **tooltip**. **Pan** (drag background) and **zoom** (scroll or **+ / −**); **Esc** clears search. Use it as a **navigation** layer between **pathway geography** and **model-ranked reactions**, not a quantitative flux-balance diagram.
"""

st.title("Flux Analysis")
st.caption(
    "**Flux Analysis** ties inferred **reaction flux** to **pathways**, **fate contrasts**, **rankings**, and **model** metadata. "
    "For multimodal **shift**/**attention** summaries, open **Feature Insights**."
)


def _build_map_html(svg_content: str, metabolite_json: str) -> str:
    """Self-contained HTML for the map iframe."""
    return (
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #fff; color: #1f2328; height: 100vh; overflow: hidden; display: flex;
  }}
  #sidebar {{
    width: 300px; min-width: 300px; max-width: 320px; background: #f6f8fa; border-right: 1px solid #d1d9e0;
    display: flex; flex-direction: column; z-index: 10;
  }}
  #sidebar h1 {{ font-size: 14px; font-weight: 600; padding: 12px 12px 4px; color: #1f2328; }}
  #sidebar .hint {{ font-size: 10px; color: #656d76; padding: 0 12px 8px; line-height: 1.35; }}
  #search-box {{
    margin: 4px 12px 8px; padding: 6px 10px; background: #fff; border: 1px solid #d1d9e0;
    border-radius: 6px; color: #1f2328; font-size: 12px; outline: none;
  }}
  #search-box:focus {{ border-color: #0969da; }}
  #search-box::placeholder {{ color: #8c959f; }}
  .btn-row {{ padding: 0 12px 8px; }}
  .btn-row button {{
    width: 100%; padding: 6px 8px; background: #f6f8fa; border: 1px solid #d1d9e0;
    border-radius: 4px; color: #1f2328; font-size: 11px; cursor: pointer;
  }}
  .btn-row button:hover {{ background: #eaeef2; }}
  #met-list-wrap {{
    flex: 1; overflow-y: auto; border-top: 1px solid #d1d9e0; min-height: 0;
  }}
  #met-list {{ padding: 4px 0 12px; }}
  .met-item {{
    padding: 7px 12px; cursor: default; font-size: 11px; border-bottom: 1px solid #eaeef2;
    display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;
  }}
  .met-item:hover {{ background: #eaeef2; }}
  .met-item .nm {{ flex: 1; min-width: 0; word-break: break-word; }}
  .met-item .rk {{ flex-shrink: 0; font-size: 10px; color: #656d76; text-align: right; }}
  .met-item .rk strong {{ color: #0969da; font-weight: 600; }}
  .met-item.hl {{ background: #ddf4ff; }}
  #map-container {{
    flex: 1; position: relative; overflow: hidden; cursor: grab; background: #fff;
    background-image: radial-gradient(circle at 1px 1px, #e8e8e8 0.5px, transparent 0);
    background-size: 24px 24px;
  }}
  #map-container.grabbing {{ cursor: grabbing; }}
  #svg-wrap {{ position: absolute; transform-origin: 0 0; }}
  #svg-wrap svg {{ display: block; }}
  #tooltip {{
    position: fixed; background: #fff; border: 1px solid #d1d9e0; border-radius: 8px;
    padding: 10px 12px; font-size: 11px; pointer-events: none; opacity: 0;
    transition: opacity 0.12s; z-index: 100; max-width: 360px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12); line-height: 1.45;
  }}
  #tooltip.vis {{ opacity: 1; }}
  #tooltip .tn {{ font-weight: 600; color: #1f2328; margin-bottom: 4px; font-size: 12px; }}
  #tooltip .tp {{ color: #1f2328; font-size: 11px; }}
  .ctrls {{
    position: absolute; bottom: 12px; right: 12px; display: flex; gap: 3px; z-index: 10;
  }}
  .ctrls button {{
    width: 32px; height: 32px; background: #fff; border: 1px solid #d1d9e0;
    border-radius: 5px; color: #1f2328; font-size: 16px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
  }}
  .ctrls button:hover {{ background: #f6f8fa; }}
  .info-bar {{
    position: absolute; top: 8px; right: 12px; font-size: 10px; color: #8c959f; z-index: 10;
  }}
</style>
</head>
<body>
<script>window.FF_METABOLITES = """
        + metabolite_json
        + r""";</script>
<div id="sidebar">
  <h1>Metabolic map</h1>
  <p class="hint">Search with any words; every word must appear somewhere in that row (name, pathway, fate, reaction text, ranks).</p>
  <input type="text" id="search-box" placeholder="Search…" autocomplete="off"/>
  <div class="btn-row">
    <button type="button" id="btn-reset">Reset zoom</button>
  </div>
  <div id="met-list-wrap"><div id="met-list"></div></div>
</div>
<div id="map-container">
  <div id="svg-wrap">"""
        + svg_content
        + r"""</div>
  <div id="tooltip"><div class="tn"></div><div class="tp"></div></div>
  <div class="ctrls">
    <button type="button" id="z-in" title="Zoom in">+</button>
    <button type="button" id="z-out" title="Zoom out">&minus;</button>
    <button type="button" id="z-fit" title="Fit">&squf;</button>
  </div>
  <div class="info-bar">Pan · zoom</div>
</div>
<script>
let sc=1,tx=0,ty=0,drag=false,dx,dy,svgEl,wrap,ctr,tt;
let tokenMap=null;
let listHighlightKey=null;

function normLabel(s){
  return s.normalize('NFD').replace(/\p{M}/gu,'').trim().toLowerCase().replace(/\s+/g,' ');
}
function buildTokenMap(){
  const m=new Map();
  const M=window.FF_METABOLITES;
  if(!M||!M.list)return m;
  for(const row of M.list){
    for(const tok of row.tokens){
      const nt=normLabel(tok);
      if(nt&&!m.has(nt))m.set(nt,row.key);
      const b=nt.replace(/\u03b2/g,'b').replace(/\u03b1/g,'a');
      if(b!==nt&&!m.has(b))m.set(b,row.key);
    }
  }
  return m;
}
function lookupMetKey(label){
  if(!tokenMap) return null;
  const nk=normLabel(label);
  let k=tokenMap.get(nk);
  if(k) return k;
  k=tokenMap.get(nk.replace(/\u03b2/g,'b').replace(/\u03b1/g,'a'));
  if(k) return k;
  if(nk.startsWith('b-')) k=tokenMap.get('\u03b2-'+nk.slice(2));
  if(!k && nk.startsWith('\u03b2-')) k=tokenMap.get('b-'+nk.slice(2));
  return k||null;
}

function escapeHtml(s){
  const d=document.createElement('div'); d.textContent=s; return d.innerHTML;
}

function rowMatchesQuery(mrow, rawQ){
  const q=(rawQ||'').trim();
  if(!q) return true;
  const fallback=((mrow.name||'')+' '+(mrow.key||'')).toLowerCase();
  const hay=(mrow.search_text||fallback).toLowerCase();
  const toks=q.toLowerCase().split(/\s+/).filter(Boolean);
  return toks.every(t=>hay.includes(t));
}

function showTip(e,label,mKey){
  const M=window.FF_METABOLITES;
  if(!mKey||!M||!M.by_key||!M.by_key[mKey]) return;
  const tn=tt.querySelector('.tn'), tp=tt.querySelector('.tp');
  const row=M.by_key[mKey];
  tn.textContent=row.name;
  tp.innerHTML=row.blurb_html;
  tt.classList.add('vis'); posT(e);
}

function clearSidebarHl(){
  listHighlightKey=null;
  document.querySelectorAll('.met-item').forEach(x=>x.classList.remove('hl'));
}

function renderMetList(q){
  const box=document.getElementById('met-list');
  box.innerHTML='';
  const M=window.FF_METABOLITES;
  if(!M||!M.list){
    box.innerHTML='<p class="hint" style="padding:12px">No index loaded for the panel.</p>';
    return;
  }
  const items=M.list.filter(m=>rowMatchesQuery(m,q));
  const cap=500;
  let n=0;
  for(const mrow of items){
    if(n++>=cap) break;
    const div=document.createElement('div');
    div.className='met-item'+(listHighlightKey===mrow.key?' hl':'');
    const rk=mrow.importance_rank!=null?`<strong>#${mrow.importance_rank}</strong>`:'<span>-</span>';
    div.innerHTML=`<span class="nm">${escapeHtml(mrow.name)}</span><span class="rk">${rk}<br/><span style="opacity:.85">${mrow.n_reactions} rxn</span></span>`;
    div.addEventListener('mouseenter',ev=>{
      document.querySelectorAll('.met-item').forEach(x=>x.classList.remove('hl'));
      div.classList.add('hl'); listHighlightKey=mrow.key;
      showTip(ev,mrow.name,mrow.key);
    });
    div.addEventListener('mousemove',posT);
    div.addEventListener('mouseleave',()=>{ tt.classList.remove('vis'); });
    box.appendChild(div);
  }
  if(items.length>cap){
    const p=document.createElement('p');
    p.className='hint'; p.style.padding='8px 12px';
    p.textContent='Showing first '+cap+' of '+items.length+' matches.';
    box.appendChild(p);
  }
}

function init(){
  tokenMap=buildTokenMap();
  ctr=document.getElementById('map-container');
  wrap=document.getElementById('svg-wrap');
  tt=document.getElementById('tooltip');
  svgEl=wrap.querySelector('svg');
  svgEl.style.width='100%'; svgEl.style.height='100%';
  svgEl.removeAttribute('width'); svgEl.removeAttribute('height');
  const vb=svgEl.viewBox.baseVal,r=ctr.getBoundingClientRect();
  const sx=r.width/vb.width,sy=r.height/vb.height;
  sc=Math.min(sx,sy)*0.92;
  tx=(r.width-vb.width*sc)/2;ty=(r.height-vb.height*sc)/2;
  svgEl.style.width=vb.width+'px'; svgEl.style.height=vb.height+'px';
  applyT();attachDiagramHoverOnly();setupPZ();
  renderMetList('');
  document.getElementById('btn-reset').addEventListener('click',resetZoomOnly);
  document.getElementById('z-in').addEventListener('click',()=>zoomIn());
  document.getElementById('z-out').addEventListener('click',()=>zoomOut());
  document.getElementById('z-fit').addEventListener('click',resetZoomOnly);
}
function applyT(){wrap.style.transform=`translate(${tx}px,${ty}px) scale(${sc})`;}

function attachDiagramHoverOnly(){
  svgEl.querySelectorAll('text').forEach(t=>{
    const c=t.textContent.trim();
    if(!c||c.length<2||c==='***'||c==='**'||c==='*') return;
    if(c.startsWith('Metabolic Alterations')) return;
    const lc=c.toLowerCase();
    if(/^log\s*2/i.test(c)||/^log2fc/i.test(lc)) return;
    if(c.length<20&&/^log/i.test(lc)) return;
    const mKey=lookupMetKey(c);
    if(!mKey) return;
    t.style.cursor='default';
    t.addEventListener('mouseenter',e=>{ showTip(e,c,mKey); });
    t.addEventListener('mousemove',posT);
    t.addEventListener('mouseleave',()=>tt.classList.remove('vis'));
  });
}

function posT(e){
  if(!tt||!e) return;
  const OFFSET=12;
  const PAD=8;
  tt.style.visibility='hidden';
  tt.style.left='0px';
  tt.style.top='0px';
  const w=tt.offsetWidth;
  const h=tt.offsetHeight;
  tt.style.visibility='visible';

  const inMap=Boolean(e.target && e.target.closest && e.target.closest('#map-container'));
  const bounds=inMap && ctr
    ? ctr.getBoundingClientRect()
    : { left: 0, top: 0, right: window.innerWidth, bottom: window.innerHeight };

  let x=e.clientX+OFFSET;
  let y=e.clientY+OFFSET;
  if(y+h+PAD>bounds.bottom) y=e.clientY-h-OFFSET;
  if(x+w+PAD>bounds.right) x=e.clientX-w-OFFSET;
  if(x+w+PAD>bounds.right) x=Math.max(bounds.left+PAD, bounds.right-w-PAD);
  if(y+h+PAD>bounds.bottom) y=Math.max(bounds.top+PAD, bounds.bottom-h-PAD);
  if(x<bounds.left+PAD) x=bounds.left+PAD;
  if(y<bounds.top+PAD) y=bounds.top+PAD;

  tt.style.left=x+'px';
  tt.style.top=y+'px';
}

function setupPZ(){
  ctr.addEventListener('mousedown',e=>{
    if(e.target.closest('text')||e.target.closest('button'))return;
    drag=true;dx=e.clientX-tx;dy=e.clientY-ty;ctr.classList.add('grabbing');
  });
  window.addEventListener('mousemove',e=>{if(!drag)return;tx=e.clientX-dx;ty=e.clientY-dy;applyT();});
  window.addEventListener('mouseup',()=>{drag=false;ctr.classList.remove('grabbing');});
  ctr.addEventListener('wheel',e=>{
    e.preventDefault();const r=ctr.getBoundingClientRect();
    const mx=e.clientX-r.left,my=e.clientY-r.top,ps=sc;
    sc=Math.max(0.3,Math.min(sc*(e.deltaY>0?0.9:1.1),15));
    tx=mx-(mx-tx)*(sc/ps);ty=my-(my-ty)*(sc/ps);applyT();
  },{passive:false});
}
function zoomBtn(f){
  const r=ctr.getBoundingClientRect(),cx=r.width/2,cy=r.height/2,ps=sc;
  sc=Math.max(0.3,Math.min(sc*f,15));
  tx=cx-(cx-tx)*(sc/ps);ty=cy-(cy-ty)*(sc/ps);applyT();
}
function zoomIn(){zoomBtn(1.3);}
function zoomOut(){zoomBtn(1/1.3);}
function resetZoomOnly(){
  const vb=svgEl.viewBox.baseVal,r=ctr.getBoundingClientRect();
  sc=Math.min(r.width/vb.width,r.height/vb.height)*0.92;
  tx=(r.width-vb.width*sc)/2;ty=(r.height-vb.height*sc)/2;applyT();
}

const searchEl=document.getElementById('search-box');
searchEl.addEventListener('input',function(){ renderMetList(this.value); });
window.addEventListener('keydown',e=>{
  if(e.key==='Escape'){
    searchEl.value='';
    renderMetList('');
    clearSidebarHl();
    tt.classList.remove('vis');
  }
});
init();
</script>
</body></html>"""
    )


st.subheader("Metabolic map")
st.caption("This page shows the interactive metabolic map of important pathways and reactions.")
ui.plot_caption_with_help(
    "Browse metabolites tied to the reconstruction and flux layer. The number is the rank of the strongest linked reaction (1 = top).",
    _HELP_MET_MAP,
    key="flux_map_help",
)

_streamlit_hf = Path(__file__).resolve().parents[2]
_svg_path = _streamlit_hf / "static" / "metabolic_map.svg"

_meta = io.load_metabolic_model_metadata()
_df = io.load_df_features()
_flux = None
if _df is not None and not _df.empty and "modality" in _df.columns:
    _flux = _df[_df["modality"].astype(str).str.upper().eq("FLUX")].copy()

_bundle = io.build_metabolite_map_bundle(_meta, _flux)
_met_json = json.dumps(_bundle if _bundle else None)

if _svg_path.is_file():
    _svg_content = _svg_path.read_text(encoding="utf-8")
    _html_doc = _build_map_html(_svg_content, _met_json)
    _iframe_src = "data:text/html;base64," + base64.b64encode(_html_doc.encode("utf-8")).decode("ascii")
    st.iframe(_iframe_src, height=820)
else:
    st.warning("The map graphic is missing in this deployment.")
