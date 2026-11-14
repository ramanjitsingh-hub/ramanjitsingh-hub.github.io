// Homepage figure for investigation 01. Needs assets/filter.js.
/* ---------------------------------------------------------------------------
   fig. 1  Robust grid filter vs the production FemoraAI update (shared-filter.js)
--------------------------------------------------------------------------- */
(function(){
  const svg=document.getElementById("fig1"),NS="http://www.w3.org/2000/svg",W=640,H=300,L=34,R=8,T=34,B=36;
  const xmin=18,xmax=62,sx=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),ymax=.6,sy=y=>H-B-(y/ymax)*(H-B-T);
  const seq=RF.synthetic(7,22);let f,prod,k,acc=true,playing=true;
  const el=(n,a,t)=>{const e=document.createElementNS(NS,n);for(const q in a)e.setAttribute(q,a[q]);if(t!=null)e.textContent=t;svg.append(e);return e;};
  function reset(){f=RF.gridFilter({prior:28,priorSd:5,s:1.8});prod=RF.production(28);k=0;acc=true;draw();}
  function draw(){
    svg.innerHTML="";
    for(let x=20;x<=60;x+=10){el("line",{x1:sx(x),x2:sx(x),y1:T-6,y2:H-B,stroke:"#ECE9E2"});el("text",{x:sx(x),y:H-B+20,"text-anchor":"middle","font-size":11,fill:"#8A8C8F","font-family":"Geist Mono"},x);}
    el("line",{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:"#DCD8CF"});
    el("text",{x:W-R,y:H-4,"text-anchor":"end","font-size":11,fill:"#8A8C8F","font-family":"Geist Mono"},"cycle level, days");
    for(let i=0;i<k;i++){const z=seq[i].z,x=sx(Math.min(z,xmax-.3)),cur=i===k-1;
      el("line",{x1:x,x2:x,y1:H-B,y2:H-B-(cur?18:9),stroke:"#9FBCD6","stroke-width":cur?2:1.2,opacity:cur?1:.75});
      if(z>xmax&&cur)el("text",{x:x-4,y:H-B-22,"text-anchor":"end","font-size":10.5,fill:"#5B5E63","font-family":"Geist Mono"},"z = "+z+" (merged)");}
    const d=f.density;let path=`M${sx(xmin)},${sy(0)}`;
    RF.X.forEach((x,i)=>{if(x>=xmin&&x<=xmax)path+=` L${sx(x).toFixed(1)},${sy(Math.min(d[i],ymax)).toFixed(1)}`;});
    path+=` L${sx(xmax)},${sy(0)} Z`;
    el("path",{d:path,fill:"#9D93C8","fill-opacity":.22,stroke:"#9D93C8","stroke-width":1.6,"stroke-linejoin":"round"});
    const m=prod.S.mu,s=prod.sigma(),yb=T-12;
    el("line",{x1:sx(m-s),x2:sx(m+s),y1:yb,y2:yb,stroke:"#5B5E63","stroke-width":1.2,"stroke-dasharray":"3 3"});
    [m-s,m+s].forEach(q=>el("line",{x1:sx(q),x2:sx(q),y1:yb-4,y2:yb+4,stroke:"#5B5E63","stroke-width":1.2}));
    el("circle",{cx:sx(m),cy:yb,r:2.6,fill:"#5B5E63"});
    el("line",{x1:sx(m),x2:sx(m),y1:yb+6,y2:H-B,stroke:"#5B5E63","stroke-width":.8,"stroke-dasharray":"2 4",opacity:.6});
    const o=k?seq[k-1]:null;
    document.getElementById("f-k").textContent="k = "+k;
    document.getElementById("f-read").textContent=o?`z = ${o.z} d${o.missed?" merged":""} · posterior mean ${f.mean().toFixed(1)} · production ${m.toFixed(1)}${acc?"":" · gate rejected"}`:"prior, before any observation";
  }
  function step(){if(k>=seq.length){reset();return;}const z=seq[k].z;f.update(z);acc=prod.observe(z);k++;draw();}
  document.getElementById("b-step").onclick=step;document.getElementById("b-reset").onclick=reset;
  const pb=document.getElementById("b-play");pb.onclick=()=>{playing=!playing;pb.textContent=playing?"pause":"play";};
  reset();
  let vis=false;new IntersectionObserver(es=>{vis=es[0].isIntersecting;}).observe(svg);
  setInterval(()=>{if(playing&&vis&&!document.hidden)step();},1400);
})();

