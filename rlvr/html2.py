header = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>
"""

header2 = """
</title>
<style>
  :root { --bg:#0b0f17; --card:#121826; --fg:#e6edf3; --muted:#9aa4b2; --grid:#1f2637; --accent:#6ea8fe; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: monospace; background:var(--bg); color:var(--fg); }
    .green { color: rgb(9, 222, 9); }
    .red { color: red; }
    .yellow { color: rgb(224, 149, 8); }
    .black { color: rgb(146, 144, 144); }
    .gray { color: rgb(212, 203, 203); }
    .blue { color: rgb(18, 18, 193); }
    .bold { font-weight: bold; }
    .cyan { color: rgb(10, 208, 208); font-weight: bold; }

  .wrap { max-width: 1100px; margin: 28px auto; padding: 0 16px; }
  h1 { font-size: 22px; margin: 0 0 10px; }
  .grid { display:grid; grid-template-columns: 1fr; gap:18px; }
  @media (min-width: 980px) { .grid { grid-template-columns: 1fr 1fr; } }
  .card { background:var(--card); border-radius:16px; box-shadow: 0 10px 30px rgba(0,0,0,.25); padding:16px 18px; }
  .title { font-weight:600; margin:0 0 6px; }
  .subtle { color: var(--muted); font-size: 12px; }
  canvas { width:100%; height:auto; }
  .tooltip {
    position: fixed; pointer-events: none; background: #0f172a; color:#e2e8f0;
    padding:6px 8px; border-radius: 8px; font-size: 12px; border:1px solid #243045;
    transform: translate(-50%, calc(-100% - 12px)); display:none; z-index: 10;
    white-space: nowrap;
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="subtle">
"""

header3 = """
  </div>
  <div id="charts" class="grid"></div>
</div>
<div id="tooltip" class="tooltip"></div>

<script>
/* ===================== YOUR DATA ===================== */
/* Format:
  const charts = [
    { name: "prices",  x: x1, y: y1, xLabel: "step",  yLabel: "price ($)" },
    { name: "rewards", x: x1, y: y2, xLabel: "epoch", yLabel: "reward" }
  ];
*/
//const x1 = [0,1,2,3,4,5,6];
//const y1 = [2,3,5,6,7,9,10];
//const y2 = [1,1.5,2.5,4,6,6.5,7];

//const charts = [
//  { name: "prices",  x: x1, y: y1, xLabel: "step",  yLabel: "price ($)" },
//  { name: "rewards", x: x1, y: y2, xLabel: "epoch", yLabel: "reward" },
//];

"""
charthtml = """
;
/* ===================== HELPERS ===================== */
const BASE = {
  padding: { left: 60, right: 20, top: 16, bottom: 52 },
  gridColor: "#1f2637",
  axisColor: "#556080",
  labelColor: "#9aa4b2",
  font: "24px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
};

function extent(a){ let lo=Infinity, hi=-Infinity; for (const v of a){ if (v<lo) lo=v; if (v>hi) hi=v; } return [lo,hi]; }
function niceRange(min,max,n=6){
  if(!isFinite(min)||!isFinite(max)) return [0,1,1];
  const span = max-min || Math.abs(max)||1;
  const step0 = Math.pow(10, Math.floor(Math.log10(span/n)));
  const steps=[1,2,2.5,5,10].map(s=>s*step0);
  let best=steps[0]; for(const s of steps) if (Math.abs(span/s-n)<Math.abs(span/best-n)) best=s;
  const niceMin=Math.floor(min/best)*best, niceMax=Math.ceil(max/best)*best;
  return [niceMin,niceMax,best];
}
function scaleLinear(d0,d1,r0,r1){const d=d1-d0||1,m=(r1-r0)/d;return v=>r0+(v-d0)*m;}
function fmt(v){const a=Math.abs(v); if (a>=1e6) return (v/1e6).toFixed(1)+"M"; if (a>=1e3) return (v/1e3).toFixed(1)+"k";
  if (Number.isInteger(v)) return String(v); if (a<1 && a>0) return v.toFixed(2); return v.toFixed(2);}

/* ===================== DRAW ONE LINE CHART ===================== */
function drawLineChart(canvas, data, xLabel, yLabel) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const P = BASE.padding;
  ctx.clearRect(0,0,W,H);

  // Validate
  if (!Array.isArray(data.x) || !Array.isArray(data.y) || data.x.length !== data.y.length || data.x.length === 0) {
    ctx.fillStyle = BASE.labelColor; ctx.font = BASE.font;
    ctx.fillText("Invalid data (x and y must be same length, non-empty).", 20, 28);
    return { xScale:null, yScale:null };
  }

  // Ranges & scales
  const [xminRaw,xmaxRaw]=extent(data.x);
  const [yminRaw,ymaxRaw]=extent(data.y);
  const [xmin,xmax,xStep]=niceRange(xminRaw,xmaxRaw);
  const [ymin,ymax,yStep]=niceRange(yminRaw,ymaxRaw);

  const xScale = scaleLinear(xmin, xmax, P.left, W - P.right);
  const yScale = scaleLinear(ymin, ymax, H - P.bottom, P.top);

  // Ticks
  const xTicks=[], yTicks=[];
  for (let t=xmin; t<=xmax+1e-9; t+=xStep) xTicks.push(+t.toFixed(12));
  for (let t=ymin; t<=ymax+1e-9; t+=yStep) yTicks.push(+t.toFixed(12));

  // Axes & grid
  ctx.save();
  ctx.font = BASE.font;
  ctx.strokeStyle = BASE.gridColor;
  ctx.lineWidth = 1;

  // grid lines
  for (const t of xTicks) { const px = xScale(t); line(ctx, px, H-P.bottom, px, P.top); }
  for (const t of yTicks) { const py = yScale(t); line(ctx, P.left, py, W-P.right, py); }

  // axes
  ctx.strokeStyle = BASE.axisColor;
  line(ctx, P.left, H-P.bottom, W-P.right, H-P.bottom);
  line(ctx, P.left, H-P.bottom, P.left, P.top);

  // tick labels
  ctx.fillStyle = BASE.labelColor;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of xTicks) ctx.fillText(fmt(t), xScale(t), H-P.bottom+8);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const t of yTicks) ctx.fillText(fmt(t), P.left-8, yScale(t));

  // axis labels
  if (xLabel) { ctx.textAlign="center"; ctx.textBaseline="top"; ctx.fillText(xLabel, (P.left+W-P.right)/2, H-23); }
  if (yLabel) { ctx.save(); ctx.translate(12, (P.top + H - P.bottom)/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center"; ctx.textBaseline="top"; ctx.fillText(yLabel, 0, 0); ctx.restore(); }

  // -- compute rolling average (simple moving avg) over last 10 datapoints --
  const windowSize = 10;
  const n = data.y.length;
  const ma = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; ++i) {
    sum += data.y[i];
    if (i >= windowSize) {
      sum -= data.y[i - windowSize];
    }
    const count = Math.min(i + 1, windowSize);
    ma[i] = sum / count;
  }

  // line (original data)
  ctx.strokeStyle = "#6ea8fe";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let moved=false;
  for (let i=0;i<data.x.length;i++){
    const px=xScale(data.x[i]), py=yScale(data.y[i]);
    if (!moved){ ctx.moveTo(px,py); moved=true; } else ctx.lineTo(px,py);
  }
  ctx.stroke();

  // points (original data)
  ctx.fillStyle = "#6ea8fe";
  for (let i=0;i<data.x.length;i++){
    const px=xScale(data.x[i]), py=yScale(data.y[i]);
    ctx.beginPath(); ctx.arc(px,py,2.8,0,Math.PI*2); ctx.fill();
  }

  // overlay: rolling average line
  ctx.strokeStyle = "#ff7f0e"; // orange for contrast
  ctx.lineWidth = 2;
  ctx.setLineDash([]); // solid; change if you prefer dashed
  ctx.beginPath();
  moved = false;
  for (let i = 0; i < data.x.length; ++i) {
    const px = xScale(data.x[i]);
    const py = yScale(ma[i]);
    if (!moved) { ctx.moveTo(px, py); moved = true; } else { ctx.lineTo(px, py); }
  }
  ctx.stroke();

  // optional: small dots on the moving-average (uncomment if desired)
  // ctx.fillStyle = "#ff7f0e";
  // for (let i=0;i<data.x.length;i++){
  //   const px=xScale(data.x[i]), py=yScale(ma[i]);
  //   ctx.beginPath(); ctx.arc(px,py,2,0,Math.PI*2); ctx.fill();
  // }

  ctx.restore();

  return { xScale, yScale };
}

function line(ctx,x1,y1,x2,y2){ ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); }

/* ===================== TOOLTIP ===================== */
function attachTooltip(canvas, info, data) {
  const tip = document.getElementById("tooltip");
  const rect = () => canvas.getBoundingClientRect();

  canvas.addEventListener("mousemove", e => {
    if (!info.xScale) return;
    const r = rect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;

    // nearest point
    let best = null, bestD = Infinity, bestIdx = -1;
    for (let i=0;i<data.x.length;i++){
      const px = info.xScale(data.x[i]);
      const py = info.yScale(data.y[i]);
      const d2 = (mx-px)*(mx-px)+(my-py)*(my-py);
      if (d2 < bestD) { bestD = d2; best = {px,py}; bestIdx = i; }
    }
    if (best){
      tip.style.display = "block";
      tip.style.left = e.clientX + "px";
      tip.style.top  = e.clientY + "px";
      tip.innerHTML = `x=${fmt(data.x[bestIdx])}<br>y=${fmt(data.y[bestIdx])}`;
    }
  });
  canvas.addEventListener("mouseleave", () => tip.style.display = "none");
}

/* ===================== RENDER ALL CHARTS ===================== */
function renderCharts(charts){
  const host = document.getElementById("charts");
  host.innerHTML = "";
  charts.forEach((c, idx) => {
    // card
    const card = document.createElement("div");
    card.className = "card";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = c.name ?? `Chart ${idx+1}`;
    const subtitle = document.createElement("div");
    subtitle.className = "subtle";
    subtitle.textContent = (c.xLabel || c.yLabel) ? `x: ${c.xLabel ?? "x"}  •  y: ${c.yLabel ?? "y"}` : "";
    const canvas = document.createElement("canvas");
    canvas.width = 900; canvas.height = 420;

    card.appendChild(title);
    if (subtitle.textContent) card.appendChild(subtitle);
    card.appendChild(canvas);
    host.appendChild(card);

    // draw & tooltip
    const info = drawLineChart(canvas, {x:c.x, y:c.y}, c.xLabel, c.yLabel);
    //attachTooltip(canvas, info, {x:c.x, y:c.y});
  });
}

/* Responsive redraw (basic) */
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    // re-render by recreating canvases at current CSS width
    renderCharts(charts);
  }, 100);
});

/* boot */
renderCharts(charts);
</script>
"""