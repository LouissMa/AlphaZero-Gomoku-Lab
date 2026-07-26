"use strict";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#game-board");
const ctx = canvas.getContext("2d");
const ui = {
  form: $("#setup-form"), model: $("#model-select"), color: $("#color-select"),
  search: $("#search-select"), simulations: $("#simulations-select"),
  thinking: $("#thinking"), status: $("#status-text"), turn: $("#turn-badge"),
  moveCount: $("#move-count"), valueScore: $("#value-score"), valueLabel: $("#value-label"),
  valueBar: $("#value-bar"), latency: $("#latency"), budget: $("#budget"), rule: $("#rule"),
  heatmap: $("#heatmap-toggle"), undo: $("#undo-button"), replay: $("#replay-button"),
  replayPanel: $("#replay-panel"), replaySlider: $("#replay-slider"),
  replayOutput: $("#replay-output"), replayExit: $("#replay-exit"), error: $("#error-message"),
};
const state = { game: null, models: [], pending: false, hover: null, heatmap: true,
  replayFrames: null, replayIndex: null, animationTime: 0, geometry: null };

async function request(path, options = {}) {
  const response = await fetch(path, {headers: {"Content-Type": "application/json"}, ...options});
  let payload = null;
  try { payload = await response.json(); } catch (_) { /* response can be empty */ }
  if (!response.ok) throw new Error(payload?.detail || `请求失败 (${response.status})`);
  return payload;
}

function setPending(pending) {
  state.pending = pending; ui.thinking.hidden = !pending;
  [...ui.form.elements, ui.undo, ui.replay].forEach((element) => { element.disabled = pending; });
  if (!pending) syncControls();
}
function showError(error) { ui.error.textContent = error ? error.message : ""; }

async function loadModels() {
  state.models = await request("/api/models");
  ui.model.innerHTML = state.models.map((model) =>
    `<option value="${model.id}">${model.name}</option>`).join("");
}

async function newGame(event) {
  event?.preventDefault(); showError(null); setPending(true);
  try {
    state.game = await request("/api/games", {method: "POST", body: JSON.stringify({
      model_id: ui.model.value, human_color: ui.color.value, search: ui.search.value,
      simulations: Number(ui.simulations.value),
    })});
    state.replayFrames = null; state.replayIndex = null; ui.replayPanel.hidden = true;
    updateInterface();
  } catch (error) { showError(error); } finally { setPending(false); render(); }
}

function viewState() {
  if (state.replayFrames && state.replayIndex !== null) {
    return {...state.game, ...state.replayFrames[state.replayIndex], analysis: null};
  }
  return state.game;
}

function statusCopy(game) {
  if (!game) return "正在载入模型目录…";
  if (state.replayFrames) return `复盘 · 第 ${state.replayIndex} 手`;
  if (game.status === "draw") return "和棋 · 棋盘已满";
  if (game.status.endsWith("_won")) return `${game.winner === game.human_color ? "你赢了" : "AI 获胜"} · 可撤销后再战`;
  return game.current_color === game.human_color ? "轮到你落子" : "AI 正在思考";
}
function colorName(color) { return color === "black" ? "黑方" : "白方"; }

function updateInterface() {
  const game = viewState(); if (!game) return;
  ui.status.textContent = statusCopy(game);
  ui.turn.textContent = game.status === "playing" ? `${colorName(game.current_color)}行棋` : "对局结束";
  ui.moveCount.textContent = String(game.move_count).padStart(2, "0");
  ui.budget.textContent = `${state.game.search.toUpperCase()} · ${state.game.simulations}`;
  ui.rule.textContent = `${game.width}×${game.height} · 连 ${game.n_in_row}`;
  const analysis = game.analysis;
  if (analysis) {
    const percent = Math.round(analysis.value * 100);
    ui.valueScore.textContent = `${percent > 0 ? "+" : ""}${percent}`;
    ui.valueLabel.textContent = `${colorName(analysis.color)}视角 · ${Math.abs(percent) < 8 ? "局势均衡" : percent > 0 ? "更有利" : "承受压力"}`;
    const blackValue = analysis.color === "black" ? analysis.value : -analysis.value;
    ui.valueBar.style.width = `${Math.max(0, Math.min(100, (blackValue + 1) * 50))}%`;
    ui.latency.textContent = `${analysis.latency_ms.toFixed(1)} ms`;
  } else {
    ui.valueScore.textContent = "—"; ui.valueLabel.textContent = "终局 / 复盘";
    ui.valueBar.style.width = "50%"; ui.latency.textContent = "— ms";
  }
  syncControls(); render();
}
function syncControls() {
  const game = state.game; const replaying = Boolean(state.replayFrames);
  ui.undo.disabled = state.pending || !game?.can_undo || replaying;
  ui.replay.disabled = state.pending || !game || game.move_count === 0 || replaying;
}

async function playAt(row, column) {
  const game = state.game;
  if (!game || state.pending || state.replayFrames || game.status !== "playing" || game.current_color !== game.human_color) return;
  showError(null); setPending(true);
  try {
    state.game = await request(`/api/games/${game.id}/moves`, {
      method: "POST", body: JSON.stringify({row, column}),
    }); updateInterface();
  } catch (error) { showError(error); } finally { setPending(false); render(); }
}

async function undo() {
  if (!state.game) return; showError(null); setPending(true);
  try { state.game = await request(`/api/games/${state.game.id}/undo`, {method: "POST"}); updateInterface(); }
  catch (error) { showError(error); } finally { setPending(false); render(); }
}
async function enterReplay() {
  if (!state.game) return;
  try {
    state.replayFrames = await request(`/api/games/${state.game.id}/replay`);
    state.replayIndex = state.replayFrames.length - 1; ui.replayPanel.hidden = false;
    ui.replaySlider.max = String(state.replayFrames.length - 1); ui.replaySlider.value = String(state.replayIndex);
    updateReplayOutput(); updateInterface();
  } catch (error) { showError(error); }
}
function exitReplay() { state.replayFrames = null; state.replayIndex = null; ui.replayPanel.hidden = true; updateInterface(); }
function updateReplayOutput() { ui.replayOutput.textContent = `${state.replayIndex} / ${state.replayFrames.length - 1}`; }

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect(); const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); render();
}
function boardGeometry(game) {
  const size = canvas.getBoundingClientRect().width; const pad = size * .085;
  return {size, pad, stepX: (size - pad * 2) / (game.width - 1), stepY: (size - pad * 2) / (game.height - 1)};
}
function point(row, column, geo) { return {x: geo.pad + column * geo.stepX, y: geo.pad + row * geo.stepY}; }

function render() {
  const game = viewState(); const size = canvas.getBoundingClientRect().width || 600;
  ctx.clearRect(0, 0, size, size);
  const paper = ctx.createLinearGradient(0, 0, size, size); paper.addColorStop(0, "#d8ba80"); paper.addColorStop(.5, "#cda96d"); paper.addColorStop(1, "#b88e55");
  ctx.fillStyle = paper; ctx.fillRect(0, 0, size, size);
  ctx.globalAlpha = .12; ctx.strokeStyle = "#fff5d6";
  for (let i = 0; i < 24; i++) { const y = (i * 47 + 13) % size; ctx.beginPath(); ctx.moveTo(0, y); ctx.bezierCurveTo(size*.3,y+7,size*.7,y-9,size,y+2); ctx.stroke(); }
  ctx.globalAlpha = 1;
  if (!game) { ctx.fillStyle="#25312e"; ctx.textAlign="center"; ctx.font="20px Georgia"; ctx.fillText("LOADING KIFU…",size/2,size/2); return; }
  const geo = boardGeometry(game); state.geometry = geo;
  ctx.strokeStyle = "rgba(34,37,31,.72)"; ctx.lineWidth = Math.max(1, size/650);
  for (let c=0;c<game.width;c++){const p1=point(0,c,geo),p2=point(game.height-1,c,geo);ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke();}
  for (let r=0;r<game.height;r++){const p1=point(r,0,geo),p2=point(r,game.width-1,geo);ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke();}
  const stars = game.width >= 8 ? [[2,2],[2,game.width-3],[game.height-3,2],[game.height-3,game.width-3]] : [[1,1],[1,game.width-2],[game.height-2,1],[game.height-2,game.width-2]];
  if (game.width%2===1 && game.height%2===1) stars.push([(game.height-1)/2,(game.width-1)/2]);
  ctx.fillStyle="#28312d"; for(const [r,c] of stars){const p=point(r,c,geo);ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2,size/190),0,Math.PI*2);ctx.fill();}
  if (state.heatmap && !state.replayFrames && game.analysis) {
    const max=Math.max(...game.analysis.policy, .0001);
    game.analysis.policy.forEach((prob,index)=>{if(prob<=0)return;const r=Math.floor(index/game.width),c=index%game.width,p=point(r,c,geo),ratio=prob/max;ctx.beginPath();ctx.fillStyle=`rgba(11,109,123,${.08+.48*ratio})`;ctx.arc(p.x,p.y,geo.stepX*(.13+.28*ratio),0,Math.PI*2);ctx.fill();});
  }
  for(const move of game.moves){const p=point(move.row,move.column,geo),radius=Math.min(geo.stepX,geo.stepY)*.42;ctx.save();ctx.shadowColor="rgba(25,20,12,.35)";ctx.shadowBlur=radius*.32;ctx.shadowOffsetY=radius*.12;const grad=ctx.createRadialGradient(p.x-radius*.3,p.y-radius*.35,radius*.08,p.x,p.y,radius);if(move.color==="black"){grad.addColorStop(0,"#56605b");grad.addColorStop(.34,"#202826");grad.addColorStop(1,"#080b0a");}else{grad.addColorStop(0,"#fffdf1");grad.addColorStop(.55,"#eae2d0");grad.addColorStop(1,"#b8ad98");}ctx.fillStyle=grad;ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fill();ctx.restore();}
  if(game.last_move){const p=point(game.last_move.row,game.last_move.column,geo);ctx.strokeStyle=game.last_move.color==="black"?"#f3d6b0":"#b43b2b";ctx.lineWidth=2;ctx.strokeRect(p.x-4,p.y-4,8,8);}
  if(state.hover && !state.pending && !state.replayFrames && game.status==="playing" && game.current_color===game.human_color){const occupied=game.moves.some(m=>m.row===state.hover.row&&m.column===state.hover.column);if(!occupied){const p=point(state.hover.row,state.hover.column,geo);ctx.globalAlpha=.32;ctx.fillStyle=game.human_color==="black"?"#111":"#fff";ctx.beginPath();ctx.arc(p.x,p.y,Math.min(geo.stepX,geo.stepY)*.4,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;}}
}
function canvasCell(event) { const rect=canvas.getBoundingClientRect(),game=viewState();if(!game)return null;const geo=state.geometry||boardGeometry(game),x=event.clientX-rect.left,y=event.clientY-rect.top;const column=Math.round((x-geo.pad)/geo.stepX),row=Math.round((y-geo.pad)/geo.stepY);if(row<0||row>=game.height||column<0||column>=game.width)return null;return {row,column}; }

ui.form.addEventListener("submit",newGame); ui.undo.addEventListener("click",undo); ui.replay.addEventListener("click",enterReplay); ui.replayExit.addEventListener("click",exitReplay);
ui.heatmap.addEventListener("change",()=>{state.heatmap=ui.heatmap.checked;render();});
ui.replaySlider.addEventListener("input",()=>{state.replayIndex=Number(ui.replaySlider.value);updateReplayOutput();updateInterface();});
canvas.addEventListener("pointermove",event=>{state.hover=canvasCell(event);render();}); canvas.addEventListener("pointerleave",()=>{state.hover=null;render();});
canvas.addEventListener("click",event=>{const cell=canvasCell(event);if(cell)playAt(cell.row,cell.column);});
window.addEventListener("resize",resizeCanvas); document.addEventListener("fullscreenchange",()=>requestAnimationFrame(resizeCanvas));
document.addEventListener("keydown",async event=>{if(event.key.toLowerCase()==="f"&&!event.ctrlKey&&!event.metaKey){event.preventDefault();if(!document.fullscreenElement)await $("#board-frame").requestFullscreen();else await document.exitFullscreen();}});
window.render_game_to_text=()=>JSON.stringify({coordinate_system:"top-left origin; rows increase downward; columns increase rightward",pending:state.pending,heatmap:state.heatmap,replay_index:state.replayIndex,game:viewState()?{id:state.game.id,status:viewState().status,current_color:viewState().current_color,human_color:state.game.human_color,width:viewState().width,height:viewState().height,moves:viewState().moves,last_move:viewState().last_move,analysis:viewState().analysis}:null});
window.advanceTime=(ms)=>{state.animationTime+=Math.max(0,Number(ms)||0);render();};

(async()=>{try{await loadModels();await newGame();}catch(error){showError(error);}finally{requestAnimationFrame(resizeCanvas);}})();