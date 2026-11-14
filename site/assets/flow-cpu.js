// CPU fallback for the homepage flow when WebGPU is unavailable. Defines window.startCpuFlow.
/* ---------------------------------------------------------------------------
   fig. 0  Stable-fluids solver (Stam 1999) with vorticity confinement
   (Fedkiw, Stam, Jensen 2001). Inviscid, semi-Lagrangian advection,
   Gauss–Seidel pressure projection on a 160 x 90 cell-centred grid.
--------------------------------------------------------------------------- */
window.startCpuFlow=function(){
  const NX=160, NY=90, W=NX+2, SZ=W*(NY+2), IX=(i,j)=>i+W*j;
  let u=new Float32Array(SZ),v=new Float32Array(SZ),u0=new Float32Array(SZ),v0=new Float32Array(SZ);
  const p=new Float32Array(SZ),div=new Float32Array(SZ),curl=new Float32Array(SZ);
  const dyes=[0,1,2].map(()=>({d:new Float32Array(SZ),t:new Float32Array(SZ)}));
  const COLS=[[157,147,200],[159,188,214],[163,181,155]], BASE=[246,244,239];
  const canvas=document.getElementById("fluid"),ctx=canvas.getContext("2d");
  const off=document.createElement("canvas");off.width=NX;off.height=NY;const octx=off.getContext("2d");
  const img=octx.createImageData(NX,NY);
  const reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.getElementById("a-grid").textContent=`${NX} × ${NY}, CPU fallback`;document.getElementById("a-v1").textContent="semi-Lagrangian";document.getElementById("a-v2").textContent="20 Gauss–Seidel sweeps";document.getElementById("a-k3").textContent="step time, your CPU";

  function bnd(b,x){
    for(let i=1;i<=NX;i++){x[IX(i,0)]=b===2?-x[IX(i,1)]:x[IX(i,1)];x[IX(i,NY+1)]=b===2?-x[IX(i,NY)]:x[IX(i,NY)];}
    for(let j=1;j<=NY;j++){x[IX(0,j)]=b===1?-x[IX(1,j)]:x[IX(1,j)];x[IX(NX+1,j)]=b===1?-x[IX(NX,j)]:x[IX(NX,j)];}
    x[IX(0,0)]=.5*(x[IX(1,0)]+x[IX(0,1)]);x[IX(0,NY+1)]=.5*(x[IX(1,NY+1)]+x[IX(0,NY)]);
    x[IX(NX+1,0)]=.5*(x[IX(NX,0)]+x[IX(NX+1,1)]);x[IX(NX+1,NY+1)]=.5*(x[IX(NX,NY+1)]+x[IX(NX+1,NY)]);
  }
  function advect(b,d,d0,uu,vv,dt){
    const dt0=dt*NX;
    for(let j=1;j<=NY;j++)for(let i=1;i<=NX;i++){
      const k=IX(i,j);let x=i-dt0*uu[k],y=j-dt0*vv[k];
      x=Math.min(Math.max(x,.5),NX+.5);y=Math.min(Math.max(y,.5),NY+.5);
      const i0=x|0,j0=y|0,s1=x-i0,t1=y-j0,s0=1-s1,t0=1-t1;
      d[k]=s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j0+1)])+s1*(t0*d0[IX(i0+1,j0)]+t1*d0[IX(i0+1,j0+1)]);
    }
    bnd(b,d);
  }
  function project(){
    const h=1/NX;
    for(let j=1;j<=NY;j++)for(let i=1;i<=NX;i++){const k=IX(i,j);
      div[k]=-.5*h*(u[k+1]-u[k-1]+v[k+W]-v[k-W]);p[k]=0;}
    bnd(0,div);bnd(0,p);
    for(let it=0;it<20;it++){
      for(let j=1;j<=NY;j++)for(let i=1;i<=NX;i++){const k=IX(i,j);p[k]=(div[k]+p[k-1]+p[k+1]+p[k-W]+p[k+W])/4;}
      bnd(0,p);
    }
    for(let j=1;j<=NY;j++)for(let i=1;i<=NX;i++){const k=IX(i,j);
      u[k]-=.5*(p[k+1]-p[k-1])/h;v[k]-=.5*(p[k+W]-p[k-W])/h;}
    bnd(1,u);bnd(2,v);
  }
  function maxDiv(){let m=0;for(let j=1;j<=NY;j++)for(let i=1;i<=NX;i++){const k=IX(i,j);
    const d=Math.abs(.5*(u[k+1]-u[k-1]+v[k+W]-v[k-W]));if(d>m)m=d;}return m;}
  function confine(dt,eps){
    for(let j=1;j<=NY;j++)for(let i=1;i<=NX;i++){const k=IX(i,j);curl[k]=.5*((v[k+1]-v[k-1])-(u[k+W]-u[k-W]));}
    for(let j=2;j<NY;j++)for(let i=2;i<NX;i++){const k=IX(i,j);
      let gx=.5*(Math.abs(curl[k+1])-Math.abs(curl[k-1])),gy=.5*(Math.abs(curl[k+W])-Math.abs(curl[k-W]));
      const len=Math.hypot(gx,gy)+1e-6;gx/=len;gy/=len;
      u[k]+=dt*eps*gy*curl[k];v[k]-=dt*eps*gx*curl[k];}
  }
  function splat(cx,cy,fx,fy,ci,amt,rad){
    const r2=rad*rad,i0=Math.max(1,(cx-3*rad)|0),i1=Math.min(NX,(cx+3*rad)|0),j0=Math.max(1,(cy-3*rad)|0),j1=Math.min(NY,(cy+3*rad)|0);
    for(let j=j0;j<=j1;j++)for(let i=i0;i<=i1;i++){const w=Math.exp(-((i-cx)**2+(j-cy)**2)/r2),k=IX(i,j);
      u[k]+=fx*w;v[k]+=fy*w;dyes[ci].d[k]=Math.min(1.6,dyes[ci].d[k]+amt*w);}
  }
  // emitters on slow Lissajous paths, biased to the right of the composition
  const em=[{a:.31,b:.23,ph:0},{a:.19,b:.29,ph:2.1},{a:.27,b:.17,ph:4.2}];
  let time=0,frame=0,ema=0,pointer=null;
  function step(){
    const dt=.016,t0=performance.now();time+=dt;
    em.forEach((e,c)=>{
      const x=.70+.22*Math.sin(e.a*time+e.ph),y=.55+.32*Math.sin(e.b*time*1.3+e.ph*.7);
      const dx=.22*e.a*Math.cos(e.a*time+e.ph),dy=.32*e.b*1.3*Math.cos(e.b*time*1.3+e.ph*.7);
      splat(x*NX,y*NY,dx*.9,dy*.9,c,.07,4.0);
    });
    if(pointer&&pointer.fresh){splat(pointer.x*NX,pointer.y*NY,pointer.dx*3,pointer.dy*3,frame%3,.12,4.5);pointer.fresh=false;}
    confine(dt,.9);
    u0.set(u);v0.set(v);advect(1,u,u0,u0,v0,dt);advect(2,v,v0,u0,v0,dt);
    project();
    for(const dy of dyes){dy.t.set(dy.d);advect(0,dy.d,dy.t,u,v,dt);for(let k=0;k<SZ;k++)dy.d[k]*=.9965;}
    for(let k=0;k<SZ;k++){u[k]*=.9995;v[k]*=.9995;}
    const ms=performance.now()-t0;ema=ema?ema*.95+ms*.05:ms;frame++;
    if(frame%20===0){document.getElementById("a-ms").textContent=ema.toFixed(1)+" ms";
      document.getElementById("a-div").textContent=(maxDiv()).toExponential(1);}
  }
  function render(){
    const data=img.data;
    for(let j=0;j<NY;j++)for(let i=0;i<NX;i++){const k=IX(i+1,j+1),o=4*(i+NX*j);
      let r=BASE[0],g=BASE[1],b=BASE[2];
      for(let c=0;c<3;c++){const a=Math.min(1,dyes[c].d[k])*.78;const C=COLS[c];r=r*(1-a)+C[0]*a;g=g*(1-a)+C[1]*a;b=b*(1-a)+C[2]*a;}
      data[o]=r;data[o+1]=g;data[o+2]=b;data[o+3]=255;}
    octx.putImageData(img,0,0);
    ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality="high";
    ctx.drawImage(off,0,0,canvas.width,canvas.height);
  }
  function size(){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=r.width*d;canvas.height=r.height*d;}
  size();addEventListener("resize",size);
  const hero=document.querySelector(".demo");
  hero.addEventListener("pointermove",e=>{const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
    if(pointer){pointer.dx=x-pointer.x;pointer.dy=y-pointer.y;}pointer={x,y,dx:pointer?pointer.dx:0,dy:pointer?pointer.dy:0,fresh:true};});
  let visible=true;new IntersectionObserver(es=>{visible=es[0].isIntersecting;}).observe(hero);
  if(reduce){for(let i=0;i<240;i++)step();render();}
  else{(function loop(){if(visible&&!document.hidden){step();render();}requestAnimationFrame(loop);})();}
};

