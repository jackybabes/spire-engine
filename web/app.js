// The viewer and play mode (milestone 6). The engine (engine.js) holds the run; this page asks it
// for the view after every move and draws it: the run on the left, the screen in the middle with
// the moves under it, the decisions on the right. A start screen opens runs: the gallery
// (gallery.json, the files under /runs/), a file, a new run, or the run kept in this browser.
import { loadEngine, ENGINE } from "./engine.js";
import * as T from "./text.js";
import { eventAbout, eventOutcome } from "./text-events.js";
import { artFor } from "./art.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const STORE_RUN = "spire-viewer.run"; // the run being played here: every decision, its variations, and where we are
const STORE_LINES = "spire-viewer.variations:"; // + a server run's address: the variations played on it here
const linesKey = (url) => STORE_LINES + url.replace(/^\//, ""); // "runs/x.json" and "/runs/x.json" alike
const STORE_THEME = "spire-viewer.theme";
const STORE_SPEED = "spire-viewer.speed";
const STORE_HIDE = "spire-viewer.hide-advice";

let speed = 1; // decisions a second while playing

/// Sets the playback speed (−/+ step it by 10%), keeps it for next time, and carries on playing.
function setSpeed(v) {
  speed = Math.min(50, Math.max(0.1, v));
  $("speedval").textContent = `${speed < 10 ? speed.toFixed(1) : Math.round(speed)} /s`;
  storage((s) => s.setItem(STORE_SPEED, String(speed)));
  if (timer) (stop(), togglePlay());
}

let call = null; // the engine holding the open run (useEngine), as a function
let callUrl = null; // ...and its address
const engines = new Map(); // address -> {ready (a promise), call, info, net (the AI loaded in it), failed}
let INFO = { cards: {}, relics: {}, potions: {}, monsters: {} };
let GALLERY = [];
let NETS = []; // the AIs (web/networks.json): name, character, title, summary, file, bytes, win
let aiChoice = null; // the chosen AI's name (null: none); it downloads only once chosen
const downloads = new Map(); // AI name -> {status: "downloading" | "ready" | "error", progress, buffer, error}
let NET = null; // the AI loaded in the engine holding the run: its networks.json entry, with its weights file's metadata
let view = null; // the engine's view of the current position
let log = []; // every decision of the line we're on: act, floor, pos, hp, label, action, by
let timer = null;
// Where the open run came from: {url} for a run on the server (links can point at it), else
// {name} (a file, a new run or the kept one), which this browser keeps in its storage.
let source = null;

// ---- Small helpers ----

function storage(fn) {
  try {
    return fn(window.localStorage);
  } catch (e) {
    return null; // private windows and blocked storage: carry on without
  }
}

function banner(text) {
  $("bannertext").textContent = text;
  $("banner").hidden = !text;
}

function engine(cmd) {
  try {
    return call(cmd);
  } catch (e) {
    banner(e.message);
    throw e;
  }
}

/// Runs `fn` after `ms` of quiet (address bar and storage updates while stepping fast).
function debounce(fn, ms) {
  let t = null;
  return () => {
    clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}

const pretty = (s) => String(s).replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/_/g, " ");
const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const monsterName = (id) => INFO.monsters[id] || pretty(id);
const relicName = (id) => (INFO.relics[id] || {}).name || id;

// ---- Tooltips: elements with data-tip (an index into tips, refilled on every draw) ----

let tips = [];
const tip = (html) => `data-tip="${tips.push(html) - 1}" tabindex="0"`;
let tipFor = null;

function showTip(el) {
  const html = tips[Number(el.dataset.tip)];
  if (!html) return hideTip();
  const box = $("tip");
  box.innerHTML = html;
  box.hidden = false;
  tipFor = el;
  const r = el.getBoundingClientRect();
  const w = box.offsetWidth, h = box.offsetHeight;
  let x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
  let y = r.bottom + 8;
  if (y + h > window.innerHeight - 8 && r.top - h - 8 > 0) y = r.top - h - 8;
  box.style.left = `${x}px`;
  box.style.top = `${Math.max(8, y)}px`;
}

function hideTip() {
  $("tip").hidden = true;
  tipFor = null;
}

let lastTouch = false; // the last pointer was a finger or a pen

function wireTips() {
  let touch = false;
  document.addEventListener("pointerdown", (e) => {
    touch = e.pointerType === "touch" || e.pointerType === "pen";
    lastTouch = touch;
    const el = e.target.closest("[data-tip]");
    // A tap shows the tip (a second tap on the same thing hides it); a tap elsewhere hides it.
    if (touch) {
      if (el && el !== tipFor) showTip(el);
      else hideTip();
    }
  });
  document.addEventListener("pointerover", (e) => {
    if (e.pointerType !== "mouse") return;
    const el = e.target.closest("[data-tip]");
    if (el) showTip(el);
  });
  document.addEventListener("pointerout", (e) => {
    if (e.pointerType !== "mouse") return;
    const el = e.target.closest("[data-tip]");
    if (el && !el.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el && !touch) showTip(el);
  });
  document.addEventListener("focusout", () => !touch && hideTip());
  window.addEventListener("scroll", hideTip, true);
}

// ---- The modal (pile and deck lists, the map, confirmations, help) ----

let modalDone = null;

function modal(title, body, buttons = []) {
  stop();
  hideTip();
  $("modaltitle").textContent = title;
  $("modalbody").innerHTML = body;
  $("modalbuttons").innerHTML = buttons.map((b, i) => `<button data-i="${i}" class="${b.primary ? "primary" : ""}">${esc(b.label)}</button>`).join("");
  $("modal").hidden = false;
  return new Promise((resolve) => {
    modalDone = resolve;
    for (const el of $("modalbuttons").querySelectorAll("button")) el.onclick = () => closeModal(buttons[Number(el.dataset.i)].value);
    ($("modalbuttons").querySelector(".primary") || $("modalclose")).focus();
  });
}

function closeModal(value = null) {
  if ($("modal").hidden) return;
  $("modal").hidden = true;
  hideTip();
  const done = modalDone;
  modalDone = null;
  if (done) done(value);
}

// ---- Screens: the start screen and the viewer ----

function showHome() {
  stop();
  closeModal();
  $("home").hidden = false;
  $("viewer").hidden = true;
  $("controls").hidden = true;
  $("setup").hidden = true;
  document.body.classList.remove("viewing");
  $("runtitle").textContent = "";
  $("save").hidden = $("link").hidden = true;
  document.title = "Spire Viewer";
  storage((s) => {
    const kept = JSON.parse(s.getItem(STORE_RUN) || "null");
    if (kept) s.setItem(STORE_RUN, JSON.stringify({ ...kept, active: false }));
  });
  drawContinue();
  if (location.search) history.replaceState(null, "", location.pathname);
}

function showViewer() {
  $("home").hidden = true;
  $("setup").hidden = true;
  $("viewer").hidden = false;
  $("controls").hidden = false;
  document.body.classList.add("viewing");
  $("save").hidden = false;
}

/// Draws engine view `v` of a position on the line we're on (the engine may hold another line that
/// takes the same decisions up to here: the line's length and ending are ours).
function show(v) {
  view = { ...v, steps: log.length, ending: cur.end };
  render();
  remember();
  // On a phone (one column) keep the screen in view: a shorter next screen would otherwise
  // leave the page scrolled down to the run and the log.
  const top = $("screen").getBoundingClientRect().top;
  if (top < 0) window.scrollTo(0, window.scrollY + top);
  scrollMap();
}

// ---- Loading and starting runs ----

/// Opens a saved run (parsed JSON): `src` says where it came from, `at` where to stand (on line
/// `line`, one of its variations, if it has that one). It plays in the engine of the chosen AI if
/// that AI plays the run's character (engineForRun); `status`: show the status line while that
/// engine loads. `kept`: the variations this browser kept for a server run. Throws if the run
/// can't be opened.
async function openRun(data, src, at = 0, by = undefined, status = true, line = 0, kept = null) {
  if (Array.isArray(data)) data = data[0];
  if (data && data.log && !data.steps) data = data.log; // an episode summary holding its log
  if (!data || !Array.isArray(data.steps)) throw new Error("This file isn't a saved run (it has no decisions).");
  const url = engineForRun(data.character);
  const e = await needEngine(url, status);
  if (!e) throw new Error("the game engine didn't load");
  e.call({ cmd: "load", log: data, ...(by ? { by } : {}) });
  stop();
  closeModal();
  useEngine(e, url);
  source = src;
  // A run kept in this browser (opened from a file): the address no longer names a server run.
  if (!src.url && location.search) history.replaceState(null, "", location.pathname);
  startLines(kept && kept.n === data.steps.length ? kept.variations : data.variations, !!src.url);
  showViewer();
  const L = lines.find((x) => x.id === line) || lines[0];
  if (goLine(L, at)) banner("");
  else goLine(lines[0], at); // the banner says why the variation didn't open
  if (!src.url) keep();
}

async function openFile(file) {
  banner("");
  setBusy(`Opening ${file.name}…`);
  try {
    const data = JSON.parse(await file.text());
    await openRun(data, { name: file.name }, 0, undefined, false);
  } catch (e) {
    banner(`Couldn't open ${file.name}: ${e.message}`);
  } finally {
    setBusy(null);
  }
}

async function openUrl(url, at = 0, line = 0) {
  banner("");
  setBusy("Loading the run…");
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(r.status === 404 ? "it isn't there" : `the server said ${r.status}`);
    const g = GALLERY.find((x) => `runs/${x.file}` === url.replace(/^\//, ""));
    const kept = storage((s) => JSON.parse(s.getItem(linesKey(url)) || "null"));
    await openRun(await r.json(), { url, name: url.split("/").pop() }, at, g && g.by, false, line, kept);
  } catch (e) {
    showHome();
    banner(`Couldn't load the run ${url}: ${e.message}.`);
  } finally {
    setBusy(null);
  }
}

async function newRun(e) {
  e.preventDefault();
  stop();
  const seedText = $("seed").value.trim();
  const seed = seedText === "" ? Math.floor(Math.random() * 2 ** 40) : /^-?\d+$/.test(seedText) ? Number(seedText) : seedText;
  const character = document.querySelector('input[name="character"]:checked').value;
  const url = engineForRun(character);
  const eng = await needEngine(url);
  if (!eng) return;
  try {
    eng.call({ cmd: "new_run", seed, character, ascension: Number($("ascension").value) || 0 });
  } catch (err) {
    return banner(err.message); // a bad seed, for one
  }
  try {
    useEngine(eng, url);
    source = { name: "new run" };
    history.replaceState(null, "", location.pathname);
    startLines([], false);
    showViewer();
    goLine(lines[0], 0);
    banner("");
    keep();
  } catch (err) {
    // The engine's message is in the banner.
  }
}

// ---- The new-run page: character, ascension, seed ----

// What each ascension adds (each includes the ones below it), in our own words.
const ASCENSION = [
  "The standard game.",
  "More elites on the map.", "Normal enemies hit harder.", "Elites hit harder.", "Bosses hit harder.",
  "You heal less after a boss.", "You start each run hurt.", "Normal enemies have more HP.", "Elites have more HP.",
  "Bosses have more HP.", "You start with a curse, Ascender's Bane.", "One potion slot fewer.",
  "Card rewards are upgraded less often.", "Bosses drop less gold.", "Your max HP is lower.", "Events are less kind.",
  "Shops cost more.", "Normal enemies fight smarter.", "Elites fight smarter.", "Bosses fight smarter.",
  "Two bosses at the end of the third act.",
];

function showSetup(push = true) {
  stop();
  closeModal();
  banner("");
  $("home").hidden = true;
  $("viewer").hidden = true;
  $("controls").hidden = true;
  $("setup").hidden = false;
  document.body.classList.remove("viewing");
  if (push) history.pushState({ setup: true }, "", "#play");
  setAscension(Number($("ascension").value) || 0);
  renderAiChoices();
  $("embark").focus();
}

function setAscension(n) {
  n = Math.max(0, Math.min(20, Math.round(n) || 0));
  $("ascension").value = n;
  $("ascdesc").textContent = n === 0 ? ASCENSION[0] : `${ASCENSION[n]}${n > 1 ? " And everything below it." : ""}`;
}

/// Downloads the run: every decision of it, and its variations.
function save() {
  const data = savedRun();
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `run-${data.seed_string}-${data.character}-A${data.ascension}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

let busy = null;
/// The floating status line (over any page), or none.
function setBusy(text) {
  busy = text;
  $("busy").hidden = !text;
  if (text) $("loading").textContent = text;
  document.body.classList.toggle("busy", !!text);
}

// ---- The engines: none loads with the page ----
// A run plays in one engine: the current one (ENGINE), or, when its AI was made for an older
// encoding, the engine that AI was released with (networks.json `engine`), so every AI on the site
// keeps playing as released. An engine downloads when it is first needed (an AI picked, a run
// started or opened) and stays for the visit. Engines from before `info` reported an interface
// version speak version 1 (crates/sts-wasm INTERFACE); check `INFO.interface` before using a
// command or reply field added later.

const engineUrl = (n) => (n && n.engine) || ENGINE;

/// The engine at `url`, its download started the first time it is asked for (and again after a
/// failure).
function getEngine(url) {
  let e = engines.get(url);
  if (!e || e.failed) {
    e = { call: null, info: null, net: null, failed: false };
    e.ready = loadEngine(url).then((c) => {
      e.call = c;
      e.info = { interface: 1, ...c({ cmd: "info" }) };
      return e;
    });
    e.ready.catch(() => (e.failed = true));
    engines.set(url, e);
  }
  return e;
}

/// Waits for the engine at `url`, showing the status line only if it isn't there yet; null (with
/// the reason in the banner) if it can't load. `status`: show the line.
async function needEngine(url, status = true) {
  const e = getEngine(url);
  if (e.call) return e;
  if (status) setBusy("Loading the game engine…");
  try {
    return await e.ready;
  } catch (err) {
    banner(`Couldn't load the game engine (${url.split("/").pop()}): ${err.message}. Try reloading the page.`);
    return null;
  } finally {
    if (status) setBusy(null);
  }
}

/// Makes engine `e` (at `url`) the one holding the run: its card and relic text, its AI.
function useEngine(e, url) {
  call = e.call;
  callUrl = url;
  INFO = e.info;
  T.setInfo(INFO);
  NET = e.net;
  useChosenAi();
}

/// The engine a run of `character` plays in: the chosen AI's when it plays that character, else
/// the current one.
function engineForRun(character) {
  const n = chosenNet();
  return n && n.character === String(character || "ironclad").toLowerCase() ? engineUrl(n) : ENGINE;
}

/// Moves the open run into the engine at `url` (an AI on another engine was chosen): saved in
/// one, loaded in the other, at the same decision. False if it can't (the engine didn't load, or
/// the run doesn't replay the same there, after a fix to the game), and the run stays where it is.
async function moveRun(url) {
  if (!view || url === callUrl) return true;
  const e = await needEngine(url);
  if (!e || !view) return false;
  if (url === callUrl) return true;
  const data = engine({ cmd: "save", all: true }); // the line it holds
  const at = view.step;
  try {
    e.call({ cmd: "load", log: data });
    const v = e.call({ cmd: "goto", index: at });
    useEngine(e, url);
    show(v);
    return true;
  } catch (err) {
    banner(`This run can't move to that AI's game engine: ${err.message}`);
    return false;
  }
}

// ---- Keeping the run: the address bar for server runs, this browser's storage for the rest ----
// A run that isn't on the server is kept whole, with its variations; a server run keeps only the
// variations played on it here, by its address.

/// Saves every decision of a run that isn't on the server, or a server run's variations (after a
/// new decision, a load or a deletion).
function keep() {
  if (!view) return;
  const ok = storage((s) => {
    if (source && source.url) {
      if (lines.length > 1) s.setItem(linesKey(source.url), JSON.stringify({ n: lines[0].steps.length, variations: savedVariations(), line: cur.id, at: view.step, when: Date.now() }));
      else s.removeItem(linesKey(source.url));
    } else {
      s.setItem(STORE_RUN, JSON.stringify({ run: savedRun(), at: view.step, line: cur.id, name: source ? source.name : "", active: true, when: Date.now(), ai: aiChoice }));
    }
    return true;
  });
  if (!ok) $("pos").title = "This browser isn't keeping the run (storage is off): save it to keep it.";
}

const rememberPlace = debounce(() => {
  if (!view) return;
  if (source && source.url) {
    const q = new URLSearchParams({ run: source.url });
    if (view.step) q.set("at", view.step);
    if (cur !== lines[0]) q.set("line", cur.id);
    history.replaceState(null, "", `?${q.toString().replace(/%2F/g, "/")}`);
  }
  storage((s) => {
    const key = source && source.url ? linesKey(source.url) : STORE_RUN;
    const kept = JSON.parse(s.getItem(key) || "null");
    if (kept) s.setItem(key, JSON.stringify({ ...kept, at: view.step, line: cur.id, ...(key === STORE_RUN ? { active: true } : {}) }));
  });
}, 300);

function remember() {
  rememberPlace();
}

function keptRun() {
  return storage((s) => JSON.parse(s.getItem(STORE_RUN) || "null"));
}

function drawContinue() {
  const kept = keptRun();
  const el = $("continue");
  if (!kept || !kept.run) return (el.hidden = true);
  const r = kept.run;
  const n = r.steps.length;
  const steps = savedLine(r, kept.line || 0);
  const last = steps.length && kept.at ? steps[Math.min(kept.at, steps.length) - 1] : null;
  const floor = last ? last.floor + (last.action && last.action.type === "path" ? 1 : 0) : 0;
  const color = CHAR_COLOR[String(r.character || "ironclad").toLowerCase()] || "";
  const vars = (r.variations || []).length;
  el.innerHTML = `<button class="choice wide resume c-${color}" id="resume">
    <pre class="art c-${color}" aria-hidden="true">${esc(artFor(r.character || "ironclad", true))}</pre>
    <span class="resumetext"><b>Continue your run</b>
    <span>${esc(capital(r.character || "ironclad"))}, ascension ${r.ascension}, seed ${esc(r.seed_string || r.seed)}${last ? `, floor ${floor}` : ""} · ${plural(n, "decision")}${vars ? ` · ${plural(vars, "variation")}` : ""}${kept.name && kept.name !== "new run" ? ` · from ${esc(kept.name)}` : ""}</span></span>
    <span class="go">Continue ▸</span></button>
    <button id="forget" class="link">Forget it</button>`;
  el.hidden = false;
  $("resume").onclick = async () => (await loadedAis, resume(kept));
  $("forget").onclick = () => {
    storage((s) => s.removeItem(STORE_RUN));
    el.hidden = true;
  };
}

async function resume(kept) {
  if (kept.ai && NETS.some((n) => n.name === kept.ai)) chooseAi(kept.ai);
  try {
    await openRun(kept.run, { name: kept.name || "kept run" }, kept.at || 0, undefined, true, kept.line || 0);
  } catch (e) {
    banner(`Couldn't reopen the kept run: ${e.message}`);
  }
}

async function copyLink() {
  const url = location.href;
  try {
    await navigator.clipboard.writeText(url);
    flash($("link"), "Copied");
  } catch (e) {
    modal("Link to this decision", `<input class="copy" value="${esc(url)}" readonly onfocus="this.select()">`, [{ label: "Close", primary: true }]);
  }
}

function flash(el, text) {
  const old = el.textContent;
  el.textContent = text;
  setTimeout(() => (el.textContent = old), 1200);
}

// ---- Lines: the run and its variations ----
// As on a chess site, choosing differently part-way through a run starts a variation there and the
// run keeps its later decisions. lines[0] is the run itself (the main line); each variation
// branches from another line (`parent`) at one of its decisions (`at`: the variation's first
// decision replaces the parent's decision `at`) and holds its own decisions (`steps`, the engine's
// log entries: act, floor, pos, hp, label, action, by) and where it got to (`end`: act, floor, hp,
// outcome). A new decision at the end of a line extends it, except at the end of a server run's
// own decisions (the run on the server stays as it is). The engine holds one line (`held`);
// going to another replays that one from the seed (quick), unless both take the same decisions up
// to where we go.

let lines = [];
let cur = null; // the line we're on (`log` is its every decision)
let held = null; // the line in the engine: the same decisions as `cur` up to where we stand
let nextId = 1;
let fixedMain = false; // the main line is a server run's
let mainData = null; // the main line in the saved-run format (the engine's save of it)
let tree = 0; // counts changes to the lines (the decision list redraws on one)

const fulls = new Map(); // line -> its every decision: its parent's up to `at`, then its own
function full(L) {
  let f = fulls.get(L);
  if (!f) fulls.set(L, (f = L.parent ? full(L.parent).slice(0, L.at).concat(L.steps) : L.steps));
  return f;
}
const keys = new WeakMap(); // log entry -> actionKey of its action
function keyOf(d) {
  let k = keys.get(d);
  if (k === undefined) keys.set(d, (k = actionKey(d.action)));
  return k;
}
/// How many decisions lines `a` and `b` take in common from the start.
function shared(a, b) {
  const x = full(a), y = full(b);
  if (x === y) return x.length;
  let i = 0;
  while (i < x.length && i < y.length && (x[i] === y[i] || keyOf(x[i]) === keyOf(y[i]))) i++;
  return i;
}
function changed() {
  fulls.clear();
  tree++;
}
/// The line drawing decision `i` of line `L` in the list: `L`, or the line it branches from (and so on) for a decision before it.
function ownerOf(L, i) {
  while (L.parent && i < L.at) L = L.parent;
  return L;
}
/// The line that takes the decision with key `k` here after the same decisions as ours: ours
/// first, then the others in the order they were made. Null if none does.
function lineTaking(k) {
  const p = view.step;
  if (log[p] && keyOf(log[p]) === k) return cur;
  return lines.find((L) => full(L).length > p && keyOf(full(L)[p]) === k && shared(L, cur) >= p) || null;
}
/// Whether any line goes on from here (ours, or one through this decision).
const linesGoOn = () => lines.some((L) => full(L).length > view.step && shared(L, cur) >= view.step);

/// Where the last decision of a view's line left the run.
const endAt = (v) => ({ act: v.obs.act, floor: v.obs.floor, hp: v.obs.screen.type === "combat" ? v.obs.screen.combat.player.hp : v.obs.hp, outcome: v.outcome || null });

/// Starts the lines of the run just loaded into the engine (standing at its start): the run itself
/// and `variations` (saved ones: savedVariations). `fixed`: a server run, whose decisions stay as they are.
function startLines(variations, fixed) {
  const n = engine({ cmd: "view" }).steps;
  const end = endAt(engine({ cmd: "goto", index: n }));
  mainData = engine({ cmd: "save", all: true });
  lines = [{ id: 0, parent: null, at: 0, steps: engine({ cmd: "log" }).steps, end }];
  cur = held = lines[0];
  log = cur.steps;
  nextId = 1;
  fixedMain = fixed;
  changed();
  for (const s of Array.isArray(variations) ? variations : []) {
    const parent = lines.find((L) => L.id === s.from);
    if (!parent || !(s.id > 0) || lines.some((L) => L.id === s.id) || !Array.isArray(s.steps) || !s.steps.length) continue;
    if (!(s.at >= 0 && s.at <= full(parent).length) || !s.steps.every((d) => d && d.action)) continue;
    lines.push({ id: s.id, parent, at: s.at, steps: s.steps, end: s.end || null });
    nextId = Math.max(nextId, s.id + 1);
  }
  changed();
}

/// The variations as saved: each with the id of the line it branches from (0: the run itself).
const savedVariations = () => lines.slice(1).map((L) => ({ id: L.id, from: L.parent.id, at: L.at, steps: L.steps, end: L.end }));
/// The run as saved (Save run, this browser's storage): the main line, and its variations if it has any.
const savedRun = () => (lines.length > 1 ? { ...mainData, variations: savedVariations() } : mainData);
/// The decisions of line `id` of a saved run.
function savedLine(run, id, depth = 0) {
  const v = id && depth < 100 ? (run.variations || []).find((x) => x.id === id) : null;
  return v ? savedLine(run, v.from, depth + 1).slice(0, v.at).concat(v.steps) : run.steps;
}

/// Stands before decision `i` of line `L`. False if that line doesn't replay in this engine (made
/// in another one; the banner says why), and nothing moves.
function goLine(L, i) {
  i = Math.max(0, Math.min(i, full(L).length));
  if (held !== L && shared(held, L) < i) {
    try {
      engine({ cmd: "load", log: { ...mainData, steps: full(L).map((d) => ({ action: d.action, by: d.by })) } });
    } catch (e) {
      return false;
    }
    held = L;
  }
  cur = L;
  log = full(L);
  show(engine({ cmd: "goto", index: i }));
  return true;
}

/// After the engine took a new decision here (`v` its view after it, the engine's line now ours up
/// to here and then it): at the end of our line it extends it, else it starts a variation here.
function recordMove(v) {
  const p = v.step - 1;
  const entry = engine({ cmd: "log" }).steps[p];
  if (p === log.length && !(cur === lines[0] && fixedMain)) {
    cur.steps = [...cur.steps, entry];
    if (cur === lines[0]) mainData = engine({ cmd: "save", all: true });
  } else {
    // A variation of the line whose decision it replaces (the one its siblings branch from).
    let parent = cur;
    while (parent.parent && p <= parent.at) parent = parent.parent;
    cur = { id: nextId++, parent, at: p, steps: [entry], end: null };
    lines.push(cur);
  }
  cur.end = endAt(v);
  held = cur;
  changed();
  log = full(cur);
  show(v);
}

/// Deletes variation `L` and the variations branching from it, after asking; standing on one of
/// them, goes back to where it branches off.
async function deleteLine(L) {
  const gone = new Set([L]);
  for (const x of lines) if (x.parent && gone.has(x.parent)) gone.add(x);
  const n = [...gone].reduce((s, x) => s + x.steps.length, 0);
  const ok = await modal("Delete this variation?", `<p>It has <b>${plural(n, "decision")}</b>${gone.size > 1 ? `, in it and the ${plural(gone.size - 1, "variation")} branching from it` : ""}.</p>`,
    [{ label: "Delete", value: true, primary: true }, { label: "Cancel", value: false }]);
  if (!ok) return;
  const back = gone.has(cur);
  lines = lines.filter((x) => !gone.has(x));
  changed();
  if (back) goLine(L.parent, L.at);
  else render();
  keep();
}

/// Goes back to the run itself, where the variation we're on (or the one it branches from) leaves it.
function toMainLine() {
  let L = cur;
  while (L.parent && L.parent.parent) L = L.parent;
  stop();
  if (L.parent) goLine(lines[0], L.at);
}

// ---- Moving through the run ----

function goto(i) {
  if (!view) return;
  i = Math.max(0, Math.min(i, view.steps));
  if (i === view.step) return;
  goLine(cur, i);
}

function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  stopAi();
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play (space)");
}

function togglePlay() {
  if (timer) return stop();
  if (!view) return;
  if (view.step >= view.steps) return;
  $("play").textContent = "⏸";
  $("play").setAttribute("aria-label", "Pause (space)");
  timer = setInterval(() => {
    if (!view || view.step >= view.steps) return stop();
    goto(view.step + 1);
  }, 1000 / speed);
}

// A card selection's cards in any order (the engine lists Seek's by name, the network picks in index order).
const actionKey = (a) => JSON.stringify(a, (k, v) => (k === "indices" && Array.isArray(v) ? [...v].sort((x, y) => x - y) : v));

/// Takes legal action `i`: the same decision as our line's (or another line's from here) moves on
/// along that line; a new one extends our line at its end, or starts a variation part-way.
function take(i) {
  const a = view.legal[i];
  if (!a) return;
  stop();
  const L = lineTaking(actionKey(a.action));
  if (L) return goLine(L, view.step + 1);
  const verdictFor = adviceHidden() ? currentAdvice() : null;
  if (verdictFor) verdict = { step: view.step + 1, took: a, advice: verdictFor };
  recordMove(engine({ cmd: "step", index: i }));
  keep();
}

function jumpToFloor(e) {
  e.preventDefault();
  const f = Number($("floor").value);
  const i = log.findIndex((d) => d.floor >= f);
  stop();
  goto(i < 0 ? log.length : i);
}

// ---- Drawing ----

// ---- Moves as things on the screen ----
// Every legal move is something to click: a card, an enemy (to aim a card or a potion), a map
// room, an option, a reward, a shop item, a potion. Moves a screen doesn't draw (Proceed, Skip,
// End turn...) become buttons under it. The run's next decision (the AI's, when watching) glows.

let selected = null; // a card or potion waiting for its target: {hand} or {slot}, with its moves
let picks = new Set(); // the cards picked so far on a pick-several screen
let pickedAt = -1; // the decision the selection and picks belong to
let used = new Set(); // the legal moves this draw put on the screen

const act = (a) => (a.type === "combat" ? a.action : a); // a fight's own action, or the run's
const legalWhere = (pred) => view.legal.map((a, i) => (pred(a.action, act(a.action)) ? i : -1)).filter((i) => i >= 0);
const nextAction = () => (log[view.step] ? log[view.step].action : null);
const isNext = (i) => !!nextAction() && actionKey(view.legal[i].action) === actionKey(nextAction());
/// Marks `i` as drawn and returns its attributes: clickable, glowing when it is the next decision.
function actAttr(i, cls = "") {
  used.add(i);
  return `data-act="${i}" class="${cls}${isNext(i) ? " ai" : ""}" role="button" tabindex="0"`;
}
const nextCombat = () => {
  const a = nextAction();
  return a && a.type === "combat" ? a.action : null;
};

function resetSelection() {
  if (pickedAt !== view.step) {
    selected = null;
    picks = new Set();
    pickedAt = view.step;
  }
}

function render() {
  if (!view) return;
  tips = [];
  used = new Set();
  hideTip();
  resetSelection();
  const o = view.obs;
  const who = capital(view.character);
  $("runtitle").textContent = `${who} · A${view.ascension} · seed ${view.seed_string}`;
  document.title = `${who} A${view.ascension}, floor ${o.floor} · Spire Viewer`;
  // A link to a variation would work only in this browser.
  $("link").hidden = !(source && source.url) || cur !== lines[0];
  const later = view.steps - view.step;
  const nb = log[view.step] && log[view.step].by;
  // Short, to fit the header's row; what the glow means is in its hover text. In a variation, a
  // way back to the run first (a narrow header cuts the end off).
  $("pos").innerHTML = (cur !== lines[0] ? `<button id="tomain" class="varpill" title="You're in a variation. Back to the run where it branches off"><span class="vpword">variation </span>↩</button> ` : "")
    + `Decision ${view.step}/${view.steps}` + (view.outcome && !later ? ` · <b>${view.outcome === "Won" ? "won" : "lost"}</b>` : "")
    + (later ? ` <span class="dim">· ${later} ahead</span>` : "");
  if ($("tomain")) $("tomain").onclick = toMainLine;
  $("pos").title = later ? `What glows on the screen is ${nb === "network" ? "the AI's" : nb === "you" ? "your" : "the run's"} next move.` : "";
  $("back").disabled = $("first").disabled = view.step === 0;
  $("fwd").disabled = $("last").disabled = $("play").disabled = view.step >= view.steps;
  renderScreen(o);
  renderRun(o);
  renderBar();
  // The advice first: its height sets the decision list's, which scrolls to the current one.
  renderAdvice();
  renderLog();
  fitHand();
  fitMap();
}

/// On a big screen, the map screen's map takes the height the rest of the column leaves, so the
/// column doesn't scroll. (Phones scroll the page and keep the map full width.)
function fitMap() {
  const box = $("screen").querySelector(":scope.s-map .mapscroll");
  if (!box || !bigScreen()) return;
  // The map screen fills the column without scrolling it: the map's box gets the height the
  // title, the key and the buttons leave. Its scroll position is kept while the screen is redrawn.
  const col = $("screen").parentElement;
  const key = `${view.step}`;
  // The same box (a resize) keeps where it was; a redrawn one at the same decision goes back there;
  // a new map screen starts at the top and scrollMap brings the next rooms into view.
  const keep = box.scrollTop || (mapScroll.key === key ? mapScroll.top : 0);
  col.scrollTop = 0;
  // The screen may be zoomed (a phone on its side): measured lengths are on screen, set ones in
  // the screen's own units; f converts.
  box.style.height = "100px";
  const f = box.getBoundingClientRect().height / 100 || 1;
  box.style.height = "0px";
  const used = Math.max($("screen").getBoundingClientRect().bottom, $("actions").getBoundingClientRect().bottom);
  const room = col.getBoundingClientRect().bottom - parseFloat(getComputedStyle(col).paddingBottom) * f - used - 4;
  box.style.height = `${Math.max(240, Math.floor(room / f))}px`;
  box.scrollTop = keep;
  mapScroll = { key, top: box.scrollTop };
  box.onscroll = () => (mapScroll = { key, top: box.scrollTop });
}
let mapScroll = { key: null, top: 0 };

/// Overlaps the hand's cards just enough to fit its width, as the game fans them.
function fitHand() {
  const hand = $("screen").querySelector(".hand");
  if (!hand) return;
  const cards = hand.querySelectorAll(".card");
  if (cards.length < 2) return;
  const w = cards[0].offsetWidth;
  const room = hand.clientWidth - 24;
  const ov = Math.min(-8, Math.max(-w * 0.62, (room - cards.length * w) / (cards.length - 1)));
  hand.style.setProperty("--ov", `${Math.round(ov)}px`);
}
window.addEventListener("resize", () => view && (fitHand(), fitMap()));

const CHAR_COLOR = { ironclad: "Red", silent: "Green", defect: "Blue", watcher: "Purple" };

function hpBar(hp, max, block = 0, cls = "") {
  const pct = Math.max(0, Math.min(100, (100 * hp) / Math.max(max, 1)));
  return `<div class="hpbar ${cls}${block ? " blocked" : ""}" role="img" aria-label="${hp} of ${max} HP${block ? `, ${block} Block` : ""}">
    <i style="width:${pct}%"></i><span>${hp} / ${max}</span>${block ? `<em class="blockbadge" title="Block">🛡${block}</em>` : ""}</div>`;
}

function relicChip(r) {
  const d = INFO.relics[r.id] || {};
  const counter = r.counter >= 0 && T.RELIC_TEXT[r.id] && T.RELIC_TEXT[r.id].includes("{N}") ? r.counter : r.counter > 0 ? r.counter : undefined;
  const body = `<b>${esc(r.name || d.name || r.id)}</b><div class="dim">${esc(d.tier || "")} relic${r.used_up ? " · used up" : ""}</div><p>${T.relicText(r.id, r.counter)}</p>`;
  return `<span class="chip relic${r.used_up ? " used" : ""}" ${tip(body)}>${esc(r.name || d.name || r.id)}${counter !== undefined ? `<span class="n">${counter}</span>` : ""}</span>`;
}

/// A potion slot: click it to drink or throw away the potion (the moves it has now).
function potionChip(name, slot, bark) {
  if (!name) return `<span class="chip empty" ${tip("<b>An empty potion slot</b>")}>empty</span>`;
  const id = T.potionId(name);
  const d = INFO.potions[id] || {};
  const moves = legalWhere((_, a) => (a.type === "use_potion" || a.type === "discard_potion") && a.slot === slot);
  for (const i of moves) used.add(i);
  const glow = moves.some(isNext) ? " ai" : "";
  const sel = selected && selected.slot === slot ? " selected" : "";
  const body = `<b>${esc(name)}</b><div class="dim">${esc(d.rarity || "")} potion${d.target ? " · aimed at an enemy" : ""}${moves.length ? " · click to use" : ""}</div><p>${T.potionText(name, bark)}</p>`;
  return `<span class="chip potion${moves.length ? " usable" : ""}${glow}${sel}" data-potion="${slot}" ${tip(body)}>${esc(name)}</span>`;
}

/// The character's colour (the card colours' names: Red, Green, Blue, Purple).
const charColor = () => ({ ironclad: "Red", silent: "Green", defect: "Blue", watcher: "Purple" })[String(view.character).toLowerCase()] || "";

/// The bar across the top, the game's way: who, HP, gold and potions on the left; floor,
/// ascension, boss and keys, then the map and the deck on the right; the relics in a row under it.
function renderRun(o) {
  const k = o.keys;
  const bark = view.relics.some((r) => r.id === "SacredBark");
  const potions = view.potions_now.map((p, i) => potionChip(p, i, bark)).join("");
  const relics = view.relics.map(relicChip).join("");
  const key = (on, name, cls) => `<span class="key ${cls}${on ? " on" : ""}" ${tip(`<b>The ${name} key</b><p>${on ? "Taken." : "Not taken yet."} The three keys open the final act.</p>`)}>${on ? "◆" : "◇"}</span>`;
  const [hp, max] = o.screen.type === "combat" ? [o.screen.combat.player.hp, o.screen.combat.player.max_hp] : [o.hp, o.max_hp];
  const color = charColor();
  $("run").innerHTML = `<div class="topbar">
      <b class="tb-who c-${color}">${esc(capital(String(view.character).toLowerCase()))}</b>
      <span class="tb-hp" ${tip(`<b>HP</b><p>${hp} of ${max}.</p>`)}><i>♥</i> ${hp}/${max}</span>
      <span class="tb-gold" ${tip("<b>Gold</b><p>Spent at shops and some events.</p>")}><i>●</i> ${o.gold}</span>
      <span class="tb-potions">${potions || '<span class="dim small">no potion slots</span>'}</span>
      <span class="spacer"></span>
      <span class="tb-floor" ${tip(`<b>Act ${o.act}, floor ${o.floor}</b>`)}>Floor ${o.floor} <span class="dim">· act ${o.act}</span></span>
      <span class="tb-asc" ${tip(`<b>Ascension ${view.ascension}</b><p>Seed ${esc(view.seed_string)}: the same seed plays the same run.</p>`)}>A${view.ascension} <span class="dim tb-seed">· seed ${esc(view.seed_string)}</span></span>
      <span class="tb-boss" ${tip(`<b>This act's boss</b><p>${esc(monsterName(o.boss) || o.boss)}</p>`)}>Boss: ${esc(monsterName(o.boss) || o.boss)}</span>
      <span class="keys">${key(k.ruby, "Ruby", "ruby")}${key(k.emerald, "Emerald", "emerald")}${key(k.sapphire, "Sapphire", "sapphire")}</span>
      <button id="mapbtn" title="The act's map">Map</button><button id="deckbtn" title="Your deck">Deck <b>${o.deck.length}</b></button>
    </div>
    <div class="tb-relics">${relics || '<span class="dim small">no relics</span>'}</div>`;
  $("deckbtn").onclick = showDeck;
  $("mapbtn").onclick = () => modal(`Map · act ${o.act}`, renderMap(o, o.screen.type === "map" ? o.screen.next : [], false) + mapLegend(o), []);
}

function showDeck() {
  const o = view.obs;
  const order = { Attack: 0, Skill: 1, Power: 2, Status: 3, Curse: 4 };
  const deck = [...o.deck].sort((a, b) => (order[a.card_type] ?? 9) - (order[b.card_type] ?? 9) || a.name.localeCompare(b.name));
  cardList(`Deck (${deck.length})`, deck);
}

/// The upgraded copy of a card, or null when it can't be upgraded (curses, and cards already
/// upgraded, except Searing Blow, which upgrades any number of times).
function upgradedCard(c) {
  const d = INFO.cards[c.id] || {};
  const type = c.card_type || d.type;
  if (type === "Curse" || (type === "Status" && c.id !== "Burn")) return null;
  if (c.upgraded && c.id !== "Searing Blow") return null;
  // The card's own numbers plus what the upgrade adds (the upgraded copy's less the fresh one's).
  const plus = (k) => (c[`base_${k}`] === undefined ? {} : { [`base_${k}`]: c[`base_${k}`] + ((d.up || {})[k] ?? 0) - (d[k] ?? 0) });
  if (c.id === "Searing Blow") {
    // Its nth upgrade adds 4 + (n - 1) damage.
    const n = c.upgraded ? Number((c.name.match(/\+(\d+)$/) || [0, 1])[1]) : 0;
    const dmg = c.base_damage === undefined ? {} : { base_damage: c.base_damage + 4 + n };
    return { ...c, ...dmg, upgraded: true, name: `${d.name}+${n + 1}` };
  }
  // The cost the upgrade takes off (Hologram, Eruption...), from the card's cost now (a card the
  // run made cheaper stays as much cheaper).
  const drop = (d.cost ?? 0) - ((d.up || {}).cost ?? d.cost ?? 0);
  const cost = c.cost !== undefined && c.cost >= 0 && drop > 0 ? { cost: Math.max(0, c.cost - drop) } : {};
  return { ...c, ...plus("damage"), ...plus("block"), ...plus("magic"), ...cost, upgraded: true, name: `${c.name}+` };
}

/// A list of cards in the modal (the deck, a pile). Click a card to see it next to its upgrade;
/// "Show upgrades" shows every card upgraded, as the game's checkbox does.
let listCards = [];
// "Show upgrades": every card shown upgraded, as the game's checkbox does; one switch for the deck
// and the screens that offer cards (rewards, the shop, events), remembered.
let listAll = storage((ls) => ls.getItem("sts-show-upgrades") === "1") || false;
function setShowUpgrades(on) {
  listAll = on;
  storage((ls) => ls.setItem("sts-show-upgrades", on ? "1" : "0"));
}
const upgToggle = () => `<label class="upgtoggle"><input type="checkbox" class="showupg"${listAll ? " checked" : ""}> Show upgrades</label>`;
// The upgraded copy marks a cost the upgrade lowers (shown green, as the game does).
function shownCard(c) {
  const up = listAll && upgradedCard(c);
  return up ? { ...up, costDown: up.cost !== undefined && c.cost !== undefined && up.cost < c.cost } : c;
}
function cardList(title, list) {
  listCards = list;
  modal(title, cardListBody(), []);
}
function cardListBody() {
  if (!listCards.length) return '<p class="dim">Empty.</p>';
  const shown = listCards.map(shownCard);
  return `${upgToggle()}
    <p class="dim small">Click a card to see its upgrade.</p>
    <div class="cardgrid">${shown.map((c, k) => card(c, { attrs: `data-zoom="${k}" role="button" tabindex="0"`, cls: "zoomable" })).join("")}</div>`;
}
function onListClick(e) {
  if (e.target.classList.contains("showupg")) {
    setShowUpgrades(e.target.checked);
    $("modalbody").innerHTML = cardListBody();
    return;
  }
  if (e.target.closest("#backtolist")) return ($("modalbody").innerHTML = cardListBody());
  const z = e.target.closest("[data-zoom]");
  if (!z) return;
  const c = listCards[Number(z.dataset.zoom)];
  const up = upgradedCard(c);
  hideTip();
  $("modalbody").innerHTML = `<div class="compare">${card(c, { cls: "big" })}
    ${up ? `<span class="arrow" aria-label="upgrades to">➜</span>${card(up, { cls: "big" })}` : `<p class="dim">${c.upgraded ? "Already upgraded." : "This card can't be upgraded."}</p>`}</div>
    <p class="center"><button id="backtolist">Back to the list</button></p>`;
  const [a, b] = $("modalbody").querySelectorAll(".compare .card");
  markRaised(a, b);
}

/// As in the game, the numbers an upgrade raises show green on the upgraded card `b` (and its
/// cost, when it drops), against the card `a`.
function markRaised(a, b) {
  if (!a || !b) return;
  const na = a.querySelectorAll(".desc .num"), nb = b.querySelectorAll(".desc .num");
  if (na.length === nb.length) nb.forEach((n, k) => Number(n.textContent) > Number(na[k].textContent) && n.classList.add("up"));
  const ca = a.querySelector(".cost"), cb = b.querySelector(".cost");
  if (ca && cb && Number(cb.textContent) < Number(ca.textContent)) cb.classList.add("changed");
}

const TYPE_ICON = { Attack: "⚔", Skill: "⛨", Power: "✺", Status: "☁", Curse: "☠" };

/// A card drawn like the game's: the character's colour, the cost in a gem, a name banner (green
/// with a + when upgraded), an art panel with the type, and the text with its numbers as now.
/// `attrs`: what makes it clickable; `extra`: a price under it.
function card(c, { attrs = "", cls = "", extra = "", inFight = false } = {}) {
  const d = INFO.cards[c.id] || {};
  const s = T.cardStats({ ...c, inFight });
  const type = c.card_type || d.type || "";
  const cost = s.cost === -1 ? "X" : s.cost === -2 || s.cost === undefined || s.cost === null ? "" : s.cost;
  const text = T.cardText({ ...c, inFight });
  const kws = T.keywordsIn(text);
  const body = `<b class="${c.upgraded ? "upgraded" : ""}">${esc(c.name)}</b><div class="dim">${esc(type)}${d.rarity && d.rarity !== "Basic" && d.rarity !== "Special" ? ` · ${esc(d.rarity)}` : ""}${cost !== "" ? ` · costs ${cost}` : ""}</div><p>${text}</p>`
    + (kws.length ? `<dl>${kws.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>` : "");
  const costChanged = c.costDown || (inFight && d.cost !== undefined && s.cost !== (c.upgraded ? d.up.cost : d.cost) && s.cost >= 0);
  // A clickable card's attributes carry its own class; merge ours into it.
  const own = `card t-${esc(type)} c-${esc(d.color || "")} r-${esc(d.rarity || "")}${c.playable === false ? " unplayable" : ""} ${cls}`;
  const a = attrs.includes('class="') ? attrs.replace('class="', `class="${own} `) : `${attrs} class="${own}"`;
  return `<div ${a} ${tip(body)}>
    ${cost !== "" ? `<div class="cost${costChanged ? " changed" : ""}">${cost}</div>` : ""}
    <div class="banner${c.upgraded ? " upgraded" : ""}">${esc(c.name)}</div>
    <div class="art"><span>${TYPE_ICON[type] || ""}</span></div>
    <div class="type">${esc(type)}${["Uncommon", "Rare"].includes(d.rarity) ? ` · ${esc(d.rarity)}` : ""}</div>
    <div class="desc"><div>${text}</div></div>${extra}</div>`;
}

function powerChips(list) {
  if (!list || !list.length) return '<div class="powers"></div>'; // keeps the row, so HP bars line up
  return `<div class="powers">${list
    .map((p) => {
      const name = POWER_NAME[p.id] || pretty(p.id.replace(/Power$/, ""));
      const debuff = DEBUFFS.has(p.id) || (p.amount < 0 && (p.id === "Strength" || p.id === "Dexterity" || p.id === "Focus"));
      return `<span class="power${debuff ? " debuff" : ""}" ${tip(`<b>${esc(name)}</b><p>${T.powerText(p.id, p.amount)}</p>`)}>${esc(name)}${p.amount !== -1 && p.amount !== 0 ? ` <b>${p.amount}</b>` : ""}</span>`;
    })
    .join("")}</div>`;
}

const POWER_NAME = {
  Weakened: "Weak", IntangiblePlayer: "Intangible", DuplicationPower: "Duplication", DexLoss: "Dexterity Loss", NoBlockPower: "No Block",
  TheBomb: "The Bomb", "Generic Strength Up Power": "Strength Up", "Wraith Form v2": "Wraith Form", Lockon: "Lock-On", StaticDischarge: "Static Discharge",
  Hello: "Hello World", Bias: "Bias", Draw: "Draw", Repair: "Self Repair", EnergizedBlue: "Energized", BattleHymn: "Battle Hymn",
  BlockReturnPower: "Talk to the Hand", DevaForm: "Deva Form", DevotionPower: "Devotion", EndTurnDeath: "Blasphemy", EnergyDownPower: "Fasting",
  EstablishmentPower: "Establishment", FreeAttackPower: "Free Attack", LikeWaterPower: "Like Water", MasterRealityPower: "Master Reality",
  OmegaPower: "Omega", PathToVictoryPower: "Mark", WaveOfTheHandPower: "Wave of the Hand", WireheadingPower: "Foresight",
  WrathNextTurnPower: "Simmering Fury", Adaptation: "Rushdown", CorpseExplosionPower: "Corpse Explosion", BackAttack: "Back Attack",
  BeatOfDeath: "Beat of Death", Heatsink: "Heatsinks", Choked: "Choked", "Draw Card": "Draw Card", "Double Damage": "Double Damage",
};
const DEBUFFS = new Set(["Vulnerable", "Weakened", "Frail", "Poison", "Entangled", "No Draw", "Shackled", "Confusion", "DexLoss", "NoBlockPower",
  "Hex", "Constricted", "Slow", "Draw Reduction", "Choked", "Lockon", "Bias", "EndTurnDeath", "EnergyDownPower", "PathToVictoryPower",
  "CorpseExplosionPower", "Wraith Form v2"]);

function intentBadge(m) {
  if (!m.alive) return `<div class="intent dead">dead</div>`;
  const icon = T.INTENT_ICON[m.intent] || "?";
  const attack = m.intent_damage >= 0;
  const n = attack ? `${m.intent_damage}${m.intent_hits > 1 ? `×${m.intent_hits}` : ""}` : "";
  const total = attack ? m.intent_damage * Math.max(1, m.intent_hits) : 0;
  const text = `${T.INTENT_TEXT[m.intent] || pretty(m.intent)}${attack ? ` for ${m.intent_damage}${m.intent_hits > 1 ? ` damage ${m.intent_hits} times (${total} in all)` : " damage"}` : ""}.`;
  return `<div class="intent i-${esc(m.intent)}" ${tip(`<b>Intent</b><p>${esc(text)}</p>`)}><span class="icon" aria-hidden="true">${icon}</span>${n ? ` <b>${n}</b>` : ""}<span class="sr">${esc(text)}</span></div>`;
}

/// The fight: the enemies as boxes along the top, you below them, your hand at the bottom with
/// the draw pile and energy to its left and End turn and the other piles to its right.
function renderCombat(c) {
  const p = c.player;
  const nc = nextCombat();
  // The moves of each hand card and their targets; all of them count as drawn.
  const cardMoves = c.hand.map((_, h) => legalWhere((_, a) => a.type === "play_card" && a.hand_index === h));
  for (const l of cardMoves) for (const i of l) used.add(i);
  const targets = new Set(selected && selected.moves.length > 1 ? selected.moves.map((i) => act(view.legal[i].action).target) : []);
  const aiTarget = nc && (nc.type === "play_card" || nc.type === "use_potion") && nc.target !== null && nc.target !== undefined ? nc.target : null;
  // Monsters that died or escaped leave the field, as in the game; a half-dead one (a Darkling,
  // the Awakened One between forms) stays, dimmed, since it comes back.
  const shown = c.monsters.filter((m) => m.alive || m.half_dead);
  const unitOf = (m) => {
      const cls = `unit monster${m.alive ? "" : " dead"}${targets.has(m.id) ? " target" : ""}${aiTarget === m.id ? " ai-target" : ""}`;
      // As in the game: the intent over its head, the figure, its name and HP bar under it.
      return `<div class="${cls}" data-target="${m.id}">
      <div class="intentrow">${m.alive ? intentBadge(m) : '<span class="dim small">down: coming back</span>'}</div>
      <pre class="art" aria-hidden="true">${esc(artFor(m.name))}</pre>
      <b class="uname">${esc(monsterName(m.name))}</b>
      ${hpBar(m.hp, m.max_hp, m.block)}${powerChips(m.powers)}</div>`;
  };
  // The Shield and Spear fight, as in the game: the Spire Shield on the left, the Spire Spear on
  // the right, you between them, turned toward the one you face (the one behind you has Back
  // Attack: its attacks deal 50% more).
  const surrounded = c.monsters.some((m) => m.name === "SpireShield") && c.monsters.some((m) => m.name === "SpireSpear");
  const behind = shown.find((m) => (m.powers || []).some((q) => q.id === "BackAttack"));
  const facingLeft = surrounded && !!behind && behind.name !== "SpireShield";
  const facing = surrounded && behind
    ? ` <span class="facing" ${tip(`<b>Facing the ${esc(monsterName(facingLeft ? "SpireShield" : "SpireSpear"))}</b><p>The ${esc(monsterName(behind.name))} is behind you (Back Attack): its attacks deal 50% more. Attack one to turn to it.</p>`)}>${facingLeft ? "◀" : "▶"}</span>`
    : "";
  const orbs = p.orbs && p.orbs.length
    ? `<div class="orbs">${p.orbs.map((o) => `<span class="orb o-${esc(o.id)}" ${tip(`<b>${esc(o.id === "Empty" ? "Empty slot" : `${o.id} orb`)}</b><p>${T.orbText(o)}</p>`)}>${esc(o.id === "Empty" ? "○" : o.id)}${o.id !== "Empty" ? ` <b>${o.passive}</b>/<b>${o.evoke}</b>` : ""}</span>`).join("")}</div>`
    : "";
  const stance = p.stance && p.stance !== "Neutral" && p.stance !== "" ? `<span class="stance s-${esc(p.stance)}" ${tip(`<b>${esc(p.stance)}</b><p>${esc(T.STANCE_TEXT[p.stance] || "")}</p>`)}>${esc(p.stance)}</span>` : "";
  let choice = "";
  if (c.choice) {
    const opts = view.choice_options || [];
    const moves = legalWhere((_, a) => a.type === "choose");
    const one = !c.choice.any_number && c.choice.count === 1;
    const aiPicks = nc && nc.type === "choose" ? new Set(nc.indices) : new Set();
    const cards = c.choice.options.map((x, k) => {
      const pile = opts[k];
      if (one) {
        const i = moves.find((i) => act(view.legal[i].action).indices.length === 1 && act(view.legal[i].action).indices[0] === pile);
        if (i !== undefined) return card(x, { attrs: actAttr(i, "pickable"), inFight: true });
      }
      for (const i of moves) used.add(i);
      return card(x, { attrs: `data-pick="${pile}" role="button" tabindex="0"`, cls: `pickable${picks.has(pile) ? " picked" : ""}${aiPicks.has(pile) ? " ai" : ""}`, inFight: true });
    });
    const need = c.choice.any_number ? `up to ${c.choice.count}` : c.choice.count;
    // Scry says what it is for (the game: look at the top cards, discard any of them, or none).
    const head = String(c.choice.kind).toLowerCase() === "scry"
      ? `Scry ${c.choice.options.length}: pick the cards to discard <span class="dim">(any of them, or none)</span>`
      : `Choose ${need} <span class="dim">(${esc(pretty(c.choice.kind))}, from the ${esc(String(c.choice.pile).toLowerCase())})</span>`;
    choice = `<div class="choicebox"><h3>${head}</h3>
      <div class="cardrow">${cards.join("")}</div>${one ? "" : confirmButton(moves, (a) => a.indices)}</div>`;
  }
  const hand = c.hand.map((x, h) => {
    const moves = cardMoves[h];
    const glow = nc && nc.type === "play_card" && nc.hand_index === h ? " ai" : "";
    const sel = selected && selected.hand === h ? " selected" : "";
    return card(x, { attrs: `data-hand="${h}" role="button" tabindex="0"`, cls: `inhand${moves.length ? " playable" : ""}${glow}${sel}`, inFight: true });
  });
  const piles = view.piles || { draw: [], discard: [], exhaust: [] };
  const pileBtn = (key, name, help) => `<button class="pile p-${key}" data-pile="${key}" title="${esc(help)}"><b>${piles[key].length}</b><span>${name}</span></button>`;
  const end = legalWhere((_, a) => a.type === "end_turn");
  const endBtn = end.length ? `<button ${actAttr(end[0], "endturn")}>End turn</button>` : "";
  const hint = !selected ? "" : selected.moves.length === 1
    ? `<div class="hint">Tap the card again to play it</div>`
    : `<div class="hint">Choose a target: ${lastTouch ? "tap" : "click"} an enemy (Esc to cancel)</div>`;
  const you = `<div class="side-you"><div class="unit player${p.stance && p.stance !== "Neutral" ? ` in-${esc(p.stance)}` : ""}">
        ${orbs}
        <pre class="art c-${esc(charColor())}${facingLeft ? " flip" : ""}" aria-hidden="true">${esc(artFor(view.character, true))}</pre>
        <b class="uname">You${stance}${facing}</b>
        ${hpBar(p.hp, p.max_hp, p.block)}${powerChips(p.powers)}</div></div>`;
  return `<div class="battle">
    <div class="turn">Turn ${c.turn}</div>
    ${hint}
    ${surrounded
      ? `<div class="stage surrounded"><div class="enemies">${shown.filter((m) => m.name === "SpireShield").map(unitOf).join("")}</div>
        ${you}<div class="enemies">${shown.filter((m) => m.name !== "SpireShield").map(unitOf).join("")}</div></div>`
      : `<div class="stage">${you}<div class="enemies">${shown.map(unitOf).join("")}</div></div>`}
    ${choice}
    <div class="table">
      <div class="left"><div class="energy" ${tip("<b>Energy</b><p>Playing a card costs its cost in energy. It refills each turn.</p>")}>${p.energy}</div>${pileBtn("draw", "Draw", "The cards still to draw, in name order (their order is hidden)")}</div>
      <div class="hand">${hand.join("") || '<span class="dim">no cards in hand</span>'}</div>
      <div class="right">${endBtn}${pileBtn("discard", "Discard", "Played and discarded cards")}${pileBtn("exhaust", "Exhaust", "Removed until the fight ends")}</div>
    </div></div>`;
}

/// The Confirm button of a pick-several screen: on when the picks are one of the moves.
/// Picking none is a move where the game allows it (Scry: discard any number, none included), so
/// Confirm with nothing picked takes it, as the game's button does.
function confirmButton(moves, indicesOf) {
  const key = [...picks].sort((a, b) => a - b).join(",");
  const i = moves.find((i) => indicesOf(act(view.legal[i].action)).slice().sort((a, b) => a - b).join(",") === key);
  if (i === undefined) return `<div class="confirm"><button disabled>Confirm (${picks.size} picked)</button></div>`;
  return `<div class="confirm"><button ${actAttr(i, "primary")}>${key === "" ? "Confirm (none)" : "Confirm"}</button></div>`;
}

function showPile(key) {
  const names = { draw: "Draw pile (in name order)", discard: "Discard pile", exhaust: "Exhausted cards" };
  const list = (view.piles || {})[key] || [];
  cardList(`${names[key]} (${list.length})`, list);
}

// ---- The map ----

const ROOM_ICON = { Monster: "⚔", Elite: "☠", Event: "?", Rest: "♨", Shop: "$", Treasure: "▣", Boss: "♛" };

/// The map's key; `o` (the run's view) adds the burning elite when its map has one.
function mapLegend(o) {
  const burning = o && !o.keys.emerald && o.map.some((n) => n.emerald);
  return `<div class="legend">${Object.entries(ROOM_ICON)
    .map(([k, i]) => `<span><i class="r-${k}">${i}</i> ${esc(T.ROOM_TEXT[k])}</span>`)
    .join("")}${burning ? '<span><i class="r-Elite burning">☠</i> Burning elite: has the Emerald key</span>' : ""}<span><i class="here-dot"></i> You are here</span><span><i class="next-dot"></i> Can go next</span></div>`;
}

/// The act's map (bottom to top): where the run is, the path it took, the rooms it can go to
/// next (clickable when `clickable`, the run's next one glowing).
function renderMap(o, next, clickable) {
  // Rows far enough apart that a path straight up shows between its rooms (R: a room's radius).
  const W = 72, H = 52, pad = 46, R = 14, RB = 20;
  const px = (x) => pad + x * W;
  const py = (y) => pad + 58 + (14 - y) * H;
  const boss = [px(3), pad + 14];
  const where = (x, y) => (y >= 15 ? boss : [px(x), py(y)]);
  const visited = new Set(log.slice(0, view.step).filter((d) => d.act === o.act && d.pos).map((d) => `${d.pos[0]},${d.pos[1]}`));
  if (o.pos) visited.add(`${o.pos[0]},${o.pos[1]}`);
  const nextSet = new Set((next || []).map((n) => `${n.x},${n.y}`));
  const nextBoss = (next || []).some((n) => n.y >= 15);
  const na = nextAction();
  const aiX = clickable && na && na.type === "path" ? na.x : null;
  const pathMove = (x) => legalWhere((a) => a.type === "path" && a.x === x)[0];
  let lines = "", nodes = "";
  for (const n of o.map) {
    for (const [ex, ey] of n.edges) {
      const [ax, ay] = where(n.x, n.y), [bx, by] = where(ex, ey);
      const taken = visited.has(`${n.x},${n.y}`) && (visited.has(`${ex},${ey}`) || (ey >= 15 && o.pos && o.pos[1] >= 15));
      // From one room's edge to the other's, so the line isn't hidden under the rooms.
      const len = Math.hypot(bx - ax, by - ay) || 1, ux = (bx - ax) / len, uy = (by - ay) / len;
      const a = R + 3, b = (ey >= 15 ? RB : R) + 3;
      const [x1, y1, x2, y2] = [ax + ux * a, ay + uy * a, bx - ux * b, by - uy * b].map((v) => v.toFixed(1));
      lines += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${taken ? "taken" : ""}"/>`;
    }
  }
  const here = (x, y) => o.pos && o.pos[0] === x && o.pos[1] === y;
  for (const n of o.map) {
    const key = `${n.x},${n.y}`;
    const isNx = nextSet.has(key);
    const cls = here(n.x, n.y) ? "here" : isNx ? "next" : visited.has(key) ? "visited" : n.y < (o.pos ? o.pos[1] : 0) ? "past" : "";
    const room = n.room || "Monster";
    // The burning elite (the Emerald key's), until the key is taken.
    const burning = n.emerald && !o.keys.emerald;
    const label = `${T.ROOM_TEXT[room] || room}${burning ? " with the Emerald key" : ""}, floor ${n.y + 1}${here(n.x, n.y) ? " (you are here)" : isNx ? " (can go next)" : ""}`;
    const i = clickable && isNx ? pathMove(n.x) : undefined;
    if (i !== undefined) used.add(i);
    const click = i !== undefined ? ` data-act="${i}" role="button" tabindex="0"` : "";
    const ai = isNx && aiX === n.x && n.y < 15 ? " ai" : "";
    nodes += `<g class="node ${cls}${ai} r-${room}${burning ? " burning" : ""}"${click} aria-label="${esc(label)}"><title>${esc(label)}</title>${click ? `<circle class="hit" cx="${px(n.x)}" cy="${py(n.y)}" r="${R + 6}"/>` : ""}<circle cx="${px(n.x)}" cy="${py(n.y)}" r="${R}"/><text x="${px(n.x)}" y="${py(n.y)}">${ROOM_ICON[room] || "·"}</text>${burning ? `<text class="flame" x="${px(n.x) + R - 1}" y="${py(n.y) - R + 1}">◆</text>` : ""}</g>`;
  }
  const atBoss = o.pos && o.pos[1] >= 15;
  const bossCls = atBoss ? "here" : nextBoss ? "next" : "";
  const bi = clickable && nextBoss ? pathMove((next.find((n) => n.y >= 15) || {}).x) : undefined;
  if (bi !== undefined) used.add(bi);
  const bossClick = bi !== undefined ? ` data-act="${bi}" role="button" tabindex="0"` : "";
  const bossAi = bi !== undefined && isNext(bi) ? " ai" : "";
  nodes += `<g class="node boss ${bossCls}${bossAi} r-Boss"${bossClick} aria-label="The boss: ${esc(monsterName(o.boss))}"><title>The boss: ${esc(monsterName(o.boss))}</title><circle cx="${boss[0]}" cy="${boss[1]}" r="${RB}"/><text x="${boss[0]}" y="${boss[1]}">${ROOM_ICON.Boss}</text></g>`;
  const w = pad * 2 + 6 * W, h = py(0) + pad;
  return `<svg class="map" viewBox="0 0 ${w} ${h}" role="img" aria-label="The map of act ${o.act}">${lines}${nodes}</svg>`;
}

// ---- Screens ----

const EVENT_NAME = { Transmorgrifier: "Transmogrifier", FaceTrader: "Face Trader", MindBloom: "Mind Bloom", NoteForYourself: "Note For Yourself",
  WeMeetAgain: "We Meet Again!", SecretPortal: "Secret Portal", SensoryStone: "Sensory Stone" };

const GRID_TITLE = {
  Upgrade: "Choose a card to upgrade", Smith: "Choose a card to upgrade", Remove: "Choose a card to remove", ShopPurge: "Choose a card to remove",
  Toke: "Choose a card to remove", Transform: "Choose a card to transform", EventTransform: "Choose a card to transform",
  Astrolabe: "Choose cards to transform and upgrade", Bonfire: "Choose a card to offer", Duplicate: "Choose a card to copy",
  RemoveAndUpgrade: "Choose a card to remove", Bottle: "Choose a card for the bottle",
};

function relicBox(id, attrs = "", extra = "") {
  const d = INFO.relics[id] || {};
  const a = attrs.includes('class="') ? attrs.replace('class="', 'class="box relicbox ') : `${attrs} class="box relicbox"`;
  return `<div ${a} ${tip(`<b>${esc(d.name || id)}</b><div class="dim">${esc(d.tier || "")} relic</div><p>${T.relicText(id)}</p>`)}><b>${esc(d.name || id)}</b>${extra}<p>${T.relicText(id)}</p></div>`;
}

function potionBox(id, name, attrs = "", extra = "") {
  const a = attrs.includes('class="') ? attrs.replace('class="', 'class="box relicbox ') : `${attrs} class="box relicbox"`;
  return `<div ${a}><b>${esc(name)}</b>${extra}<p>${T.potionText(id, view.relics.some((r) => r.id === "SacredBark"))}</p></div>`;
}

/// An event-like screen: a title, a line, and its options as buttons, the game's way.
/// `list`: [move, label, enabled, html]; `html` (an event option's name and what it does) is
/// shown in place of the label when given.
function options(list) {
  return `<div class="options">${list.map(([i, label, enabled = true, html]) => (i === undefined
    ? `<button class="option" disabled>${html || esc(label)}</button>`
    : `<button ${actAttr(i, "option")}${enabled ? "" : " disabled"}>${html || esc(label)}</button>`)).join("")}</div>`;
}

/// Match and Keep's board, the game's way: 12 cards face down on a 4 x 3 grid, each in its place
/// (matched pairs leave gaps), the tries left above. The card turned first this try is face up and
/// lit; cards seen before are face up but dim (the run's view shows them, as the network sees
/// them). `move(k)`: the legal move that turns card k.
function matchAndKeep(s, move) {
  const slots = Array(12).fill(null);
  s.options.forEach((x, k) => {
    const c = s.cards[k];
    const picked = c && !x.enabled;
    const face = c ? card(c, { cls: picked ? "picked" : "seen" }) : '<div class="card back"><span>?</span></div>';
    const i = x.enabled ? move(k) : undefined;
    const at = i === undefined ? `class="mkslot${picked ? " picked" : ""}"` : actAttr(i, "mkslot");
    const pos = s.board && s.board[k] !== undefined ? s.board[k] : k;
    slots[pos] = `<div ${at}>${face}</div>`;
  });
  const tries = s.n ? `<p class="mktries">Tries left: <b>${s.n[0]}</b></p>` : "";
  return `<div class="event"><h2 class="title">Match and Keep!</h2>
    <p class="lore">Turn two cards: a matching pair goes into your deck.</p>${tries}
    <div class="mkgrid">${slots.map((x) => x || '<div class="mkslot gone"></div>').join("")}</div></div>`;
}

/// Match and Keep's end: the tries left (when the engine gives them) and the way out; no cards
/// (the board's leftovers aren't what you got).
function matchAndKeepDone(s, opts) {
  const tries = s.n ? `<p class="mktries">Tries left: <b>${s.n[0]}</b></p>` : "";
  return `<div class="event"><h2 class="title">Match and Keep!</h2><p class="lore">The game is over: every pair you matched is in your deck.</p>${tries}${options(opts)}</div>`;
}

/// An event option the game's way: its name in brackets, then what it does (the engine's label
/// after its colon when there is no text for it).
function eventOption(s, label, o) {
  const [name, rest] = String(label).split(/:\s*(.*)/s);
  const does = eventOutcome(s, label, o, view.ascension) || (rest ? `${capital(rest)}.` : "");
  return `<span class="optname">[${esc(capital(name))}]</span>${does ? ` <span class="optdoes">${esc(does)}</span>` : ""}`;
}
const optionLabel = (s) => capital(String(s).replace(/^[^:]*: /, ""));

function renderScreen(o) {
  const s = o.screen;
  let html = "";
  const first = (pred) => legalWhere(pred)[0];
  switch (s.type) {
    case "neow": {
      const texts = view.neow || s.options;
      html = `<div class="event"><h2 class="title">Neow</h2><p class="lore">Neow offers a blessing before the climb. Choose one.</p>
        ${options(texts.map((t, k) => [first((a) => a.type === "neow" && a.option === k), t]))}</div>`;
      break;
    }
    case "map":
      // The title above and the key below stay put; the map between them scrolls (fitMap).
      html = `<div class="mapscreen"><div class="maphead"><h2 class="title">Map <small>act ${o.act} · boss: ${esc(monsterName(o.boss))}</small></h2>
        <p class="dim">Click a ringed room to go there.</p></div>
        <div class="mapscroll">${renderMap(o, s.next, true)}</div>${mapLegend(o)}</div>`;
      break;
    case "combat":
      html = renderCombat(s.combat);
      break;
    case "rewards":
      html = `<div class="event"><h2 class="title">Rewards</h2><p class="lore">Click a reward to take it.</p><div class="rewards">${s.items
        .map((r, k) => {
          const i = first((a) => a.type === "take_reward" && a.index === k);
          const at = i === undefined ? `class="reward"` : actAttr(i, "reward");
          if (r.type === "cards") return `<div ${at}><b>Add a card to your deck</b><span class="dim">${r.cards.map((c) => esc(c.name)).join(", ")}</span></div>`;
          if (r.type === "gold" || r.type === "stolen_gold") return `<div ${at}><b class="gold">${r.amount} gold</b>${r.type === "stolen_gold" ? ' <span class="dim">(stolen back)</span>' : ""}</div>`;
          if (r.type === "relic") return relicBox(r.id, at.replace('class="', 'class="reward '));
          if (r.type === "potion") return potionBox(r.id, r.name, at.replace('class="', 'class="reward '), ' <span class="dim">potion</span>');
          if (r.type === "emerald_key") return `<div ${at}><b class="key emerald on">◆</b> The Emerald key</div>`;
          if (r.type === "sapphire_key") return `<div ${at}><b class="key sapphire on">◆</b> The Sapphire key <span class="dim">(instead of the relic)</span></div>`;
          return `<div ${at}>${esc(capital(pretty(r.type)))}</div>`;
        })
        .join("")}</div></div>`;
      break;
    case "card_reward":
      html = `<h2 class="title center">Choose a card</h2><div class="cardrow big">${s.cards
        .map((c, k) => {
          const i = first((a) => a.type === "pick_card" && a.index === k);
          return card(shownCard(c), { attrs: i === undefined ? "" : actAttr(i, "pickable") });
        })
        .join("")}</div><p class="center">${upgToggle()}</p>`;
      break;
    case "grid": {
      const purpose = typeof s.purpose === "string" ? s.purpose : Object.keys(s.purpose)[0];
      const moves = legalWhere((a) => a.type === "select");
      const na = nextAction();
      const aiPicks = na && na.type === "select" ? new Set(na.indices) : new Set();
      // Choosing a card to upgrade: hovering over one shows it upgraded, as in the game.
      const upgrading = ["Upgrade", "Smith"].includes(purpose);
      const drawn = (c, opts) => {
        const up = upgrading && upgradedCard(c);
        return up ? `<div class="upgflip">${card(c, opts)}${card({ ...up, costDown: up.cost < c.cost }, { ...opts, cls: `${opts.cls || ""} up` })}</div>` : card(c, opts);
      };
      const cards = s.options.map((d) => {
        if (s.count === 1) {
          const i = moves.find((i) => view.legal[i].action.indices.length === 1 && view.legal[i].action.indices[0] === d);
          if (i !== undefined) return drawn(o.deck[d], { attrs: actAttr(i, "pickable") });
        }
        return drawn(o.deck[d], { attrs: `data-pick="${d}" role="button" tabindex="0"`, cls: `pickable${picks.has(d) ? " picked" : ""}${aiPicks.has(d) ? " ai" : ""}` });
      });
      if (s.count !== 1) for (const i of moves) if (view.legal[i].action.indices.length) used.add(i);
      html = `<h2 class="title center">${esc(GRID_TITLE[purpose] || pretty(purpose))}${s.count > 1 ? ` <small>pick ${s.count}</small>` : ""}</h2>
        <div class="cardgrid">${cards.join("")}</div>${s.count === 1 ? "" : confirmButton(moves, (a) => a.indices)}`;
      break;
    }
    case "rest": {
      const moves = legalWhere((a) => !["use_potion", "discard_potion"].includes(a.type));
      html = `<div class="event"><h2 class="title">Rest site</h2><p class="lore">A campfire. Rest to heal 30% of your max HP, or upgrade a card.</p>
        ${hpBar(o.hp, o.max_hp, 0, "big")}${options(moves.map((i) => [i, view.legal[i].label]))}</div>`;
      break;
    }
    case "boss_relic":
      html = `<div class="event"><h2 class="title">Choose a boss relic</h2><div class="rewards">${s.relics
        .map((r, k) => {
          const i = first((a) => a.type === "boss_relic" && a.index === k);
          return relicBox(r, i === undefined ? "" : actAttr(i, "reward"));
        })
        .join("")}</div></div>`;
      break;
    case "shop": {
      const price = (p) => `<div class="price">${p} gold</div>`;
      const buy = (type, k) => {
        const i = first((a) => a.type === type && a.index === k);
        return i === undefined ? `class="cant"` : actAttr(i, "buy");
      };
      const purge = first((a) => a.type === "purge");
      html = `<h2 class="title">Shop <small>you have <span class="gold">${o.gold} gold</span> · click an item to buy it</small></h2>
        <div class="cardrow">${s.cards.map((it, k) => (it ? card(shownCard(it.card || { id: it.id, name: it.name }), { attrs: buy("buy_card", k), extra: price(it.price) }) : "")).join("")}</div>
        <p>${upgToggle()}</p>
        <div class="rewards">${s.relics.map((it, k) => (it ? relicBox(it.id, buy("buy_relic", k), ` <span class="gold">${it.price}</span>`) : "")).join("")}
        ${s.potions.map((it, k) => (it ? potionBox(it.id, it.name, buy("buy_potion", k), ` <span class="gold">${it.price}</span>`) : "")).join("")}
        ${s.purge_cost !== null && s.purge_cost !== undefined ? `<div ${purge === undefined ? 'class="box cant"' : actAttr(purge, "box buy")}><b>Card removal</b> <span class="gold">${s.purge_cost}</span><p class="dim">Remove a card from your deck.</p></div>` : ""}</div>`;
      break;
    }
    case "event": {
      if (s.options.length && s.options.every((x) => x.label === "turn a card")) {
        html = matchAndKeep(s, (k) => first((a) => a.type === "event_choice" && a.index === k));
        break;
      }
      // Its end (only Leave is left): the tries, no cards (in any engine, old ones included).
      if (s.id === "Match and Keep!" && s.options.length && s.options.every((x) => x.label === "leave")) {
        html = matchAndKeepDone(s, s.options.map((x, k) => [first((a) => a.type === "event_choice" && a.index === k), optionLabel(x.label), x.enabled, eventOption(s, x.label, o)]));
        break;
      }
      const opts = s.options.map((x, k) => [first((a) => a.type === "event_choice" && a.index === k), optionLabel(x.label), x.enabled, eventOption(s, x.label, o)]);
      const about = eventAbout(s);
      html = `<div class="event"><h2 class="title">${esc(EVENT_NAME[s.id] || pretty(s.id))}</h2>${about ? `<p class="lore">${esc(about)}</p>` : ""}
        ${s.cards && s.cards.some((c) => c) ? `<div class="cardrow">${s.cards.map((c) => (c ? card(shownCard(c)) : "")).join("")}</div><p>${upgToggle()}</p>` : ""}
        ${options(opts)}</div>`;
      break;
    }
    case "over": {
      const won = view.outcome === "Won";
      html = `<div class="event"><h2 class="title ${won ? "won" : "lost"}">${won ? "Victory" : "Defeat"}</h2>
        <p>${won ? "The run is won" : "The run ends"} on act ${o.act}, floor ${o.floor}, with ${o.hp} of ${o.max_hp} HP, ${o.gold} gold, ${plural(o.deck.length, "card")} and ${plural(view.relics.length, "relic")}.</p>
        <div class="options"><button class="option" id="overdeck">See the deck</button><button class="option" id="overhome">Back to the start</button></div></div>`;
      break;
    }
    default:
      html = `<h2 class="title">${esc(pretty(s.type))}</h2><pre>${esc(JSON.stringify(s, null, 1))}</pre>`;
  }
  $("screen").innerHTML = html;
  $("screen").className = `s-${s.type}`;
  for (const f of $("screen").querySelectorAll(".upgflip")) markRaised(...f.querySelectorAll(":scope > .card"));
  if ($("overdeck")) $("overdeck").onclick = showDeck;
  if ($("overhome")) $("overhome").onclick = showHome;
  // A new map screen: bring the rooms to choose from (or where the run is) into view, since the
  // map is taller than a phone's screen.
  mapToScroll = s.type === "map" && lastScreen !== `map${view.step}`;
  lastScreen = `${s.type}${view.step}`;
}
let lastScreen = null;
let mapToScroll = false;

function scrollMap() {
  if (!mapToScroll) return;
  mapToScroll = false;
  const target = $("screen").querySelector("svg.map .node.next, svg.map .node.here");
  // A big screen scrolls the map's own box (fitMap); a phone scrolls the page.
  const scroller = bigScreen() ? $("screen").querySelector(".mapscroll") : null;
  if (!target) return;
  const r = target.getBoundingClientRect();
  if (scroller) {
    const box = scroller.getBoundingClientRect();
    if (r.bottom > box.bottom || r.top < box.top) scroller.scrollTop += r.top - box.top - box.height / 2;
  } else if (r.bottom > window.innerHeight - 110 || r.top < 0) window.scrollBy(0, r.top - window.innerHeight / 2);
}

/// The moves this screen didn't draw (Proceed, Skip, a pick of nothing...), as buttons.
function renderBar() {
  const rest = view.legal.map((_, i) => i).filter((i) => !used.has(i));
  $("actions").innerHTML = (rest.length ? `<div class="bar">${rest.map((i) => `<button ${actAttr(i, "option small")}>${esc(barLabel(view.legal[i]))}</button>`).join("")}</div>` : "") + forks();
}

/// The other lines that take another decision here (the run, where we're in a variation; a
/// variation, where it branches off): each a button that follows it from here instead, with where
/// it got to, to compare.
function forks() {
  const p = view.step;
  const seen = new Set(log[p] ? [keyOf(log[p])] : []);
  const other = [];
  for (const L of lines) {
    const f = full(L);
    if (f.length <= p || shared(L, cur) < p || seen.has(keyOf(f[p]))) continue;
    seen.add(keyOf(f[p]));
    other.push(L);
  }
  if (!other.length) return "";
  return `<div class="forks"><span class="dim small">⑂ Also played here:</span>${other.map((L) => `<button class="fork small" data-fork="${L.id}" title="Follow this line from here">
    ${L === lines[0] ? "<b>The run</b>: " : ""}${esc(full(L)[p].label)} <span class="dim">→ ${endText(L.end)}</span></button>`).join("")}</div>`;
}
const barLabel = (a) => {
  const t = act(a.action).type;
  if (t === "choose" || t === "select") return a.label.replace(/^.*: nothing$/, "Choose nothing").replace(/^Choose nothing$/, "Choose nothing");
  return a.label;
};

// ---- Clicks on the screen ----

function onScreenClick(e) {
  if (!view) return;
  if (e.target.classList.contains("showupg")) return setShowUpgrades(e.target.checked), render();
  const t = e.target.closest("[data-act], [data-hand], [data-target], [data-pick], [data-potion], [data-pile], [data-fork]");
  if (!t) {
    if (selected && !e.target.closest("#run")) (selected = null), render();
    return;
  }
  if (t.dataset.pile) return showPile(t.dataset.pile);
  if (t.dataset.fork) {
    const L = lines.find((x) => x.id === Number(t.dataset.fork));
    return L && (stop(), (selected = null), goLine(L, view.step));
  }
  if (t.dataset.act !== undefined) {
    if (t.disabled) return;
    selected = null;
    return take(Number(t.dataset.act));
  }
  if (t.dataset.hand !== undefined) return selectCard(Number(t.dataset.hand), t);
  if (t.dataset.target !== undefined) return aim(Number(t.dataset.target));
  if (t.dataset.pick !== undefined) {
    const k = Number(t.dataset.pick);
    if (picks.has(k)) picks.delete(k);
    else picks.add(k);
    return render();
  }
  if (t.dataset.potion !== undefined) return usePotion(Number(t.dataset.potion));
}

/// Clicking a hand card: played at once when it has one move (no target, or one enemy);
/// otherwise picked up, waiting for an enemy to be clicked.
function selectCard(h, el) {
  const moves = legalWhere((_, a) => a.type === "play_card" && a.hand_index === h);
  if (!moves.length) {
    if (el) (el.classList.remove("shake"), void el.offsetWidth, el.classList.add("shake"));
    return;
  }
  const again = selected && selected.hand === h;
  // On a touch screen the first tap picks the card up (to read it), a second tap plays it.
  if (moves.length === 1 && (again || !lastTouch)) return (selected = null), take(moves[0]);
  if (again) return (selected = null), render();
  selected = { hand: h, moves };
  render();
}

function aim(target) {
  if (!selected) return;
  const i = selected.moves.find((i) => act(view.legal[i].action).target === target);
  if (i === undefined) return;
  selected = null;
  take(i);
}

/// Clicking a potion: drink it (aimed like a card when it takes a target) or throw it away.
async function usePotion(slot) {
  const drink = legalWhere((_, a) => a.type === "use_potion" && a.slot === slot);
  const toss = legalWhere((_, a) => a.type === "discard_potion" && a.slot === slot);
  if (!drink.length && !toss.length) return;
  const name = view.potions_now[slot];
  const choice = await modal(name, `<p>${T.potionText(name, view.relics.some((r) => r.id === "SacredBark"))}</p>`, [
    ...(drink.length ? [{ label: "Drink", value: "drink", primary: true }] : []),
    ...(toss.length ? [{ label: "Throw away", value: "toss" }] : []),
    { label: "Cancel", value: null },
  ]);
  if (choice === "toss") return take(toss[0]);
  if (choice !== "drink") return;
  if (drink.length === 1) return take(drink[0]);
  selected = { slot, moves: drink };
  render();
}

// ---- The decisions: the run's grouped by floor, each floor collapsible; a variation under the
// decision it replaces, its decisions in a row with their floors marked, as chess sites show moves ----

const BY = { network: "AI", you: "You" };
let logDrawn = -1; // the `tree` the list was drawn for
let openFor = null; // the innermost floor or variation opened for the current decision
const varOpen = new Map(); // variation id -> opened (or closed) by hand
const LONG_VARIATION = 40; // decisions; longer ones start closed unless we're in them

/// Where a line got to, in words: "floor 14 · 32 HP · lost".
function endText(e) {
  if (!e) return "";
  const out = e.outcome ? ` · <b class="out-${e.outcome === "Won" ? "won" : "lost"}">${e.outcome === "Won" ? "won" : "lost"}</b>` : "";
  return `floor ${e.floor} · ${e.hp} HP${out}`;
}

function drawLog() {
  const main = lines[0];
  const n = main.steps.length;
  const branches = new Map(); // "line id:decision" -> the variations replacing that decision
  for (const L of lines.slice(1)) {
    const k = `${L.parent.id}:${L.at}`;
    if (!branches.has(k)) branches.set(k, []);
    branches.get(k).push(L);
  }
  const varsAt = (L, i) => (branches.get(`${L.id}:${i}`) || []).map(variation).join("");
  function variation(L) {
    const parts = [];
    let floor = null, by = null;
    L.steps.forEach((d, j) => {
      const i = L.at + j;
      if (d.floor !== floor) parts.push(`<span class="vf">${d.floor === 0 ? "Start" : `F${d.floor}`}</span>`);
      if (d.by !== by) parts.push(`<span class="who by-${esc(d.by)}">${esc(BY[d.by] || "")}</span>`);
      else if (d.floor === floor) parts.push('<i class="sep">·</i>');
      parts.push(`<span class="mv by-${esc(d.by)}" data-l="${L.id}" data-i="${i}" role="button" tabindex="0">${esc(d.label)}</span>`);
      [floor, by] = [d.floor, d.by];
      const sub = varsAt(L, i);
      if (sub) (parts.push(sub), (floor = by = null));
    });
    const open = varOpen.has(L.id) ? varOpen.get(L.id) : L.steps.length <= LONG_VARIATION;
    return `<details class="var" data-v="${L.id}"${open ? " open" : ""}>
      <summary><span class="vlabel">Variation</span> <span class="dim">${plural(L.steps.length, "decision")}${L.end ? " →" : ""}</span> <span class="vend">${endText(L.end)}</span>
        <button class="vardel" data-del="${L.id}" aria-label="Delete this variation" title="Delete this variation">×</button></summary>
      <div class="vmoves">${parts.join(" ")}</div></details>`;
  }
  const groups = [];
  for (let i = 0; i < n; i++) {
    const d = main.steps[i];
    const g = groups[groups.length - 1];
    if (g && g.act === d.act && g.floor === d.floor) g.items.push(i);
    else groups.push({ act: d.act, floor: d.floor, hp: d.hp, items: [i] });
  }
  const row = (i) => {
    const d = main.steps[i];
    return `<li data-l="0" data-i="${i}" class="by-${esc(d.by)}"><span class="who" title="${esc(BY[d.by] || "Saved run")}">${esc(BY[d.by] || "")}</span>${esc(d.label)}</li>`;
  };
  const vars = (i) => {
    const html = varsAt(main, i) + (i === n - 1 ? varsAt(main, n) : "");
    return html ? `<li class="vars">${html}</li>` : "";
  };
  const html = groups.map((g) => {
    const last = g.items[g.items.length - 1];
    const count = lines.filter((L) => L.parent === main && L.at >= g.items[0] && (L.at <= last || (L.at === n && last === n - 1))).length;
    return `<details class="floor">
      <summary><span class="f">${g.floor === 0 ? "Start" : `Floor ${g.floor}`}</span> <span class="dim">act ${g.act} · ${g.hp} HP · ${plural(g.items.length, "decision")}</span>${count ? ` <span class="vcount" title="${plural(count, "variation")} branch${count === 1 ? "es" : ""} off on this floor">⑂ ${count}</span>` : ""}</summary>
      <ol>${g.items.map((i) => row(i) + vars(i)).join("")}</ol></details>`;
  }).join("");
  // A server run with no decisions yet: its variations from the start.
  const fromStart = n === 0 ? varsAt(main, 0) : "";
  const ended = main.end && (main.end.outcome || lines.length > 1);
  $("log").innerHTML = html + (fromStart ? `<ol><li class="vars">${fromStart}</li></ol>` : "")
    + (ended ? `<p class="lineend"><span class="vlabel">The run</span> <span class="dim">${plural(n, "decision")} →</span> ${endText(main.end)}</p>` : "");
  const kinds = new Set(lines.flatMap((L) => L.steps.map((d) => d.by)));
  $("log").classList.toggle("onekind", kinds.size === 1);
  $("logkey").innerHTML = (kinds.size > 1 || kinds.has("you")
    ? `<p class="dim small"><span class="who by-network">AI</span> the network's decisions · <span class="who by-you">You</span> yours</p>`
    : kinds.has("network") ? `<p class="dim small">Every decision here is the network's.</p>` : "")
    + (fixedMain && lines.length === 1 && n ? `<p class="dim small">Choose something else at any decision to play a variation from there; it shows here, under the decision it replaces.</p>` : "");
  logDrawn = tree;
  openFor = null;
}

function renderLog() {
  const el = $("log");
  if (logDrawn !== tree) drawLog();
  for (const old of el.querySelectorAll(".now, .last")) old.classList.remove("now", "last");
  const vars = lines.length - 1;
  $("logcount").textContent = `(${lines[0].steps.length}${vars ? ` · ${plural(vars, "variation")}` : ""})`;
  // The current decision (or the last one, at the end), and the floor and variations holding it, opened.
  const at = (i) => (i >= 0 ? el.querySelector(`[data-l="${ownerOf(cur, i).id}"][data-i="${i}"]`) : null);
  const i = Math.min(view.step, log.length - 1);
  const now = view.step < log.length ? at(view.step) : null;
  if (now) now.classList.add("now");
  const target = at(i);
  // At the end of a line, its last decision is marked: the line we're on.
  if (!now && target) target.classList.add("last");
  const inner = target ? target.parentElement.closest("details") : null;
  if (inner !== openFor) {
    const chain = [];
    for (let d = inner; d && el.contains(d); d = d.parentElement.closest("details")) chain.push(d);
    for (const d of el.querySelectorAll("details[data-auto]")) {
      if (!chain.includes(d)) (d.open = false), delete d.dataset.auto;
    }
    for (const d of chain) if (!d.open) (d.open = true), (d.dataset.auto = "1");
    openFor = inner;
  }
  if (!$("logbox").open || !target) return;
  fitLog();
  // At the run's newest decision the list stays at its bottom (fitLog); otherwise the current one
  // is brought into view. Scroll the list only (scrollIntoView would move the whole page on a phone).
  if (bigScreen() && atNewest()) return;
  const box = bigScreen() ? $("log") : $("side");
  if (box.scrollHeight > box.clientHeight) {
    const top = target.offsetTop, bottom = top + target.offsetHeight;
    if (top < box.scrollTop || bottom > box.scrollTop + box.clientHeight) box.scrollTop = top - box.clientHeight / 3;
  }
}
/// At the end of the run itself (a variation's end is part-way down the list).
const atNewest = () => view && view.step >= view.steps && cur === lines[0];

/// A click in the list: a decision goes there (on its line), × deletes a variation.
function onLogClick(e) {
  const del = e.target.closest("[data-del]");
  if (del) {
    e.preventDefault(); // not a toggle of the variation
    const L = lines.find((x) => x.id === Number(del.dataset.del));
    return L && deleteLine(L);
  }
  const head = e.target.closest("details.var > summary");
  if (head) {
    // Opened or closed by hand: it stays so when the list is redrawn.
    const d = head.parentElement;
    varOpen.set(Number(d.dataset.v), !d.open);
    delete d.dataset.auto;
    return;
  }
  const m = e.target.closest("[data-i]");
  const L = m && lines.find((x) => x.id === Number(m.dataset.l));
  if (L) (stop(), goLine(L, Number(m.dataset.i)));
}

// The upright-phone layout: a narrow window that is tall too (style.css); anything else, a phone on
// its side included, has the big screen's side panel and map box.
const PHONE = "(max-width: 820px) and (min-height: 541px)";
const bigScreen = () => !matchMedia(PHONE).matches;

/// On a big screen the advice stays at the top of the side panel and only the decisions scroll:
/// the list gets the height the panel has left (the advice's height changes with the position),
/// and at the newest decision it stays scrolled to its bottom.
function fitLog() {
  const list = $("log");
  if (!bigScreen() || !$("logbox").open) return void (list.style.maxHeight = "");
  const room = $("side").getBoundingClientRect().bottom - list.getBoundingClientRect().top - 12;
  list.style.maxHeight = `${Math.max(120, Math.floor(room))}px`;
  if (atNewest()) list.scrollTop = list.scrollHeight;
}

// ---- The AI's advice ----
// web/networks.json lists the AIs (one per character and version). One is chosen on the new-run
// page or in the panel; only then do its weights (and its engine, if not the current one)
// download, and the engine holding the run loads them. For a run
// of that AI's character it shows its likeliest decisions at the current position with their
// probabilities, and its estimate of the chance to win; its top choice is outlined on the screen. It can take one
// decision, play to the next floor or to the end. With "Hide until I choose" the advice waits
// until you have chosen, then says what the AI would have done.

let advice = null; // {step, choices: [{index, label, probability}], value} at the current position
let verdict = null; // after a move with the advice hidden: {step, took, advice}
let aiRun = null; // the AI playing: {mode, floor, act, handle}

let loadedAis = null; // the AIs' list loading (a promise)

/// The list of AIs (small); their weights download only when one is chosen.
function loadAiList() {
  return (loadedAis = fetchAiList());
}

async function fetchAiList() {
  try {
    NETS = await (await fetch("networks.json")).json();
  } catch (e) {
    NETS = [];
  }
  renderAiChoices();
  drawAis();
  if (view) renderAdvice();
}

/// The home page's AIs: each with its character's colour, how well it plays, and a button to play
/// a run with it (the new-run page with its character and it chosen).
function drawAis() {
  $("aisbox").hidden = !NETS.length;
  $("ais").innerHTML = NETS.map((n) => `<div class="aicard c-${CHAR_COLOR[n.character] || ""}">
    <pre class="art c-${CHAR_COLOR[n.character] || ""}" aria-hidden="true">${esc(artFor(n.character, true))}</pre>
    <div><b>${esc(n.title || n.name)}</b><p>${esc(n.summary || "")}</p>
    <button data-ai="${esc(n.name)}" data-char="${esc(n.character)}">Play a run with it ▸</button></div></div>`).join("");
  for (const b of $("ais").querySelectorAll("button[data-ai]")) {
    b.onclick = () => {
      const r = document.querySelector(`input[name="character"][value="${b.dataset.char}"]`);
      if (r) r.checked = true;
      chooseAi(b.dataset.ai);
      showSetup();
    };
  }
}

const netsFor = (character) => NETS.filter((n) => n.character === String(character || "").toLowerCase());
const chosenNet = () => NETS.find((n) => n.name === aiChoice) || null;
const mb = (bytes) => (bytes ? `${(bytes / 1e6).toFixed(1)} MB` : "");

/// Chooses an AI by name, or none (null): its weights download the first time, then go into the
/// engine. The choice holds for every run of that AI's character.
function chooseAi(name) {
  aiChoice = name || null;
  advice = null;
  verdict = null;
  const n = chosenNet();
  const d = n && downloads.get(n.name);
  if (n && (!d || d.status === "error")) fetchWeights(n);
  if (n) getEngine(engineUrl(n)); // downloads alongside the weights
  if (n && view && n.character === view.character.toLowerCase() && engineUrl(n) !== callUrl) {
    if (aiRun) stopAi();
    moveRun(engineUrl(n)).then(() => view && renderAdvice());
  }
  useChosenAi();
  renderAiChoices();
  if (view) (renderAdvice(), keep());
}

/// Downloads an AI's weights, showing how far it has got.
async function fetchWeights(n) {
  const d = { status: "downloading", progress: 0 };
  downloads.set(n.name, d);
  try {
    const r = await fetch(n.file);
    if (!r.ok) throw new Error(r.status === 404 ? "it isn't on this server" : `the server said ${r.status}`);
    const total = Number(r.headers.get("content-length")) || n.bytes || 0;
    if (r.body && total) {
      const reader = r.body.getReader();
      const parts = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        got += value.length;
        d.progress = Math.min(got / total, 1);
        showAiProgress(n.name);
      }
      const all = new Uint8Array(got);
      let at = 0;
      for (const p of parts) (all.set(p, at), (at += p.length));
      d.buffer = all.buffer;
    } else {
      d.buffer = await r.arrayBuffer();
    }
    d.status = "ready";
  } catch (e) {
    d.status = "error";
    d.error = e.message;
  }
  useChosenAi();
  renderAiChoices();
  if (view) renderAdvice();
}

/// Puts the chosen AI into the engine holding the run, once both have arrived and if it is that
/// AI's engine.
function useChosenAi() {
  const n = chosenNet();
  const d = n && downloads.get(n.name);
  const e = callUrl && engines.get(callUrl);
  if (!e || !e.call || !n || engineUrl(n) !== callUrl || !d || d.status !== "ready" || (e.net && e.net.name === n.name)) return;
  try {
    e.net = { ...n, meta: e.call.loadNet(d.buffer) };
  } catch (err) {
    d.status = "error";
    d.error = `it doesn't run in this game engine (${err.message})`;
  }
  NET = e.net;
  advice = null;
  renderAiChoices();
  if (view) renderAdvice();
}

/// How a chosen AI's download is going, in words.
function aiStatus(n) {
  const d = downloads.get(n.name);
  if (!d) return mb(n.bytes) ? `${mb(n.bytes)} download` : "";
  if (d.status === "downloading") return `Downloading… ${Math.round(100 * d.progress)}%`;
  if (d.status === "error") return `Couldn't load: ${d.error}`;
  return "Downloaded";
}

function showAiProgress(name) {
  const n = NETS.find((x) => x.name === name);
  for (const el of document.querySelectorAll(`[data-aiprog="${name}"]`)) el.textContent = aiStatus(n);
}

/// The new-run page's AI choice: none, or each AI for the chosen character.
function renderAiChoices() {
  const el = $("aichoices");
  const picked = document.querySelector('input[name="character"]:checked');
  if (!el || !picked) return;
  const nets = netsFor(picked.value);
  if (!nets.length) {
    el.innerHTML = `<p class="dim">${NETS.length ? `No AI plays the ${esc(capital(picked.value))} yet.` : "No AI is available here."}</p>`;
    return;
  }
  const current = nets.some((n) => n.name === aiChoice) ? aiChoice : "";
  const opt = (value, title, line, status) => `<label class="aiopt"><input type="radio" name="ai" value="${esc(value)}"${value === current ? " checked" : ""}>
    <b>${esc(title)}</b><span>${esc(line)}</span>${status !== null ? `<em data-aiprog="${esc(value)}">${esc(status)}</em>` : ""}</label>`;
  el.innerHTML = opt("", "No AI", "Play on your own.", null) + nets.map((n) => opt(n.name, n.title || n.name, n.summary || "", aiStatus(n))).join("");
  for (const r of el.querySelectorAll('input[name="ai"]')) r.onchange = () => chooseAi(r.value);
}

const forNetwork = () => !!(NET && aiChoice === NET.name && view && view.character.toLowerCase() === NET.character);
const adviceHidden = () => $("advhide").checked && view && view.step === view.steps;

/// The advice at the current position (computed once per position).
function currentAdvice() {
  if (!forNetwork() || view.outcome) return null;
  if (!advice || advice.step !== view.step || advice.at !== view) {
    try {
      advice = { ...call({ cmd: "advise", k: 3 }), at: view };
    } catch (e) {
      advice = null;
      banner(`The AI couldn't advise here: ${e.message}`);
    }
  }
  return advice;
}

/// The chance to win from the run value, or null: a logistic fit per network (networks.json `win`,
/// made by python/release/win_calibration.py, whose TERMS these are).
function winChance(value) {
  const w = NET && NET.win;
  if (!w) return null;
  const f = view.obs.floor, a = view.ascension, v = value;
  const terms = { bias: 1, value: v, floor: f, ascension: a, ascension2: a * a, value_ascension: v * a, value_floor: v * f };
  const z = Object.entries(terms).reduce((sum, [k, x]) => sum + (w[k] || 0) * x, 0);
  return 1 / (1 + Math.exp(-z));
}

const pct = (p) => (p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(100 * p)}%`);

/// The advice's rows; `taken` is the run's next decision when watching. The AI's own move (its
/// likeliest single action) is tagged when it isn't the first row: in a close call, copies of a
/// card counted together can outweigh it.
function choiceRows(a, taken = null) {
  const own = a.pick === null || a.pick === undefined ? null : view.legal[a.pick].label;
  return a.choices.map((c, k) => `<li class="advrow${k === 0 ? " top" : ""}" data-advice="${c.index}" role="button" tabindex="0" title="Take this">
    <span class="bar"><i style="width:${(100 * c.probability).toFixed(1)}%"></i></span>
    <span class="lbl">${esc(capital(c.label))}${k > 0 && c.label === own ? ' <span class="tag own" title="What the AI plays when it plays: its likeliest single move">AI\'s move</span>' : ""}${taken === c.index ? ' <span class="tag">taken</span>' : ""}</span><b>${pct(c.probability)}</b></li>`).join("");
}

function renderAdvice() {
  drawAdvice();
  fitLog();
}

function drawAdvice() {
  const box = $("advice");
  if (!view) return;
  box.hidden = false;
  const body = $("advbody");
  const buttons = ["aione", "aifloor", "aiend"];
  const setButtons = (on) => buttons.forEach((id) => ($(id).disabled = !on));
  for (const el of document.querySelectorAll(".advised, .advised-target")) el.classList.remove("advised", "advised-target");
  // The picker: no AI, or each AI for this run's character.
  const nets = netsFor(view.character);
  const pick = $("aipick");
  pick.hidden = !nets.length;
  pick.innerHTML = `<option value="">No AI</option>` + nets.map((n) => `<option value="${esc(n.name)}">${esc(n.title || n.name)}</option>`).join("");
  pick.value = nets.some((n) => n.name === aiChoice) ? aiChoice : "";
  $("advhide").parentElement.hidden = !forNetwork();
  if (!nets.length) {
    body.innerHTML = `<p class="dim small">${NETS.length ? `No AI plays the ${esc(capital(view.character))} yet.` : "No AI is available here."}</p>`;
    return setButtons(false);
  }
  const n = chosenNet();
  if (!n || !nets.includes(n)) {
    const got = downloads.get(nets[0].name);
    const size = got && got.status === "ready" ? "" : mb(nets[0].bytes);
    body.innerHTML = `<p class="dim small">Pick an AI above to see what it would do here and let it play${size ? ` (a ${size} download)` : ""}.</p>`;
    return setButtons(false);
  }
  const d = downloads.get(n.name);
  if (d && d.status === "error") {
    body.innerHTML = `<p class="dim small">Couldn't load ${esc(n.title || n.name)}: ${esc(d.error)}. Pick it again to try again.</p>`;
    return setButtons(false);
  }
  if (!forNetwork()) {
    body.innerHTML = `<p class="dim small loading">${d && d.status === "downloading" ? `${esc(n.title || n.name)}: <span data-aiprog="${esc(n.name)}">${esc(aiStatus(n))}</span>` : "Starting the AI…"}</p>`;
    return setButtons(false);
  }
  if (aiRun) {
    body.innerHTML = `<p class="small"><span class="loading">The AI is playing</span> · floor ${view.obs.floor}</p>`;
    buttons.forEach((id) => ($(id).disabled = id !== `ai${aiRun.mode}`));
    $(`ai${aiRun.mode}`).textContent = "stop";
    return;
  }
  $("aione").textContent = "this move";
  $("aifloor").textContent = "to the next floor";
  $("aiend").textContent = "to the end";
  if (view.outcome) {
    body.innerHTML = `<p class="dim small">The run is over.</p>`;
    return setButtons(false);
  }
  setButtons(true);
  let html = "";
  if (verdict && verdict.step === view.step) {
    const pick = verdict.advice.pick ?? verdict.advice.choices[0].index;
    const top = verdict.advice.choices.find((c) => c.label === verdict.advice.at.legal[pick].label) || { label: verdict.advice.at.legal[pick].label, probability: 0 };
    const same = verdict.took.label === top.label;
    const mine = verdict.advice.choices.find((c) => actionKey(verdict.advice.at.legal[c.index].action) === actionKey(verdict.took.action));
    html += `<div class="verdict ${same ? "same" : "differ"}"><p class="small">${same
      ? `<b>Same as the AI</b> (${pct(top.probability)}): ${esc(verdict.took.label)}`
      : `You chose <b>${esc(verdict.took.label)}</b>${mine ? ` (the AI: ${pct(mine.probability)})` : " (not in the AI's top three)"}. The AI would have chosen <b>${esc(top.label)}</b> (${pct(top.probability)}).`}</p></div>`;
  }
  if (adviceHidden()) {
    body.innerHTML = html + `<p class="dim small">Hidden until you choose.</p>`;
    return;
  }
  if (timer && speed > 4) {
    body.innerHTML = html + `<p class="dim small">Paused while playing fast.</p>`;
    return;
  }
  const a = currentAdvice();
  if (!a) return void (body.innerHTML = html);
  // When watching, which of them the run took next.
  const next = log[view.step];
  const taken = next ? a.choices.find((c) => actionKey(view.legal[c.index].action) === actionKey(next.action)) : null;
  const chance = winChance(a.value);
  html += `<ol class="advlist">${choiceRows(a, taken ? taken.index : null)}</ol>`;
  if (next && !taken) html += `<p class="dim small">The run took <b>${esc(next.label)}</b>, not one of these.</p>`;
  html += chance === null
    ? `<p class="dim small" title="The network's estimate of how the run will go (in its own reward units)">Run value ${a.value.toFixed(2)}</p>`
    : `<p class="small" title="From the network's value of the run, calibrated on its own runs">Chance to ${NET.win.heart ? "beat the Heart" : "win"}: <b>${pct(chance)}</b></p>`;
  body.innerHTML = html;
  for (const li of body.querySelectorAll("[data-advice]")) li.onclick = () => take(Number(li.dataset.advice));
  markOnScreen(a.pick ?? a.choices[0].index);
}

/// Outlines legal move `i` where the screen draws it (a card and its target, a potion, the
/// cards it picks, a room, an option or a button).
function markOnScreen(i) {
  const a = act(view.legal[i].action);
  const els = [...document.querySelectorAll(`#screen [data-act="${i}"], #actions [data-act="${i}"], #run [data-act="${i}"]`)];
  if (a.type === "play_card") els.push(...$("screen").querySelectorAll(`[data-hand="${a.hand_index}"]`));
  if ((a.type === "use_potion" || a.type === "discard_potion") && a.slot !== undefined) els.push(...document.querySelectorAll(`[data-potion="${a.slot}"]`));
  if ((a.type === "choose" || a.type === "select") && Array.isArray(a.indices)) for (const k of a.indices) els.push(...$("screen").querySelectorAll(`[data-pick="${k}"]`));
  for (const el of els) el.classList.add("advised");
  if ((a.type === "play_card" || a.type === "use_potion") && a.target !== null && a.target !== undefined) {
    for (const el of $("screen").querySelectorAll(`[data-target="${a.target}"]`)) el.classList.add("advised-target");
  }
}

/// The AI takes the decision here. Where a line goes on from here, its choice (its advice's pick)
/// moves on along the line that takes it, if one does; otherwise it is a new decision (take).
function aiMove() {
  if (linesGoOn()) {
    const a = currentAdvice();
    const pick = a && (a.pick ?? a.choices[0].index);
    const L = a && view.legal[pick] ? lineTaking(actionKey(view.legal[pick].action)) : null;
    if (L) return goLine(L, view.step + 1);
  }
  recordMove(engine({ cmd: "net_step", count: 1 }));
}

/// The AI plays: "one" decision, to the next "floor", or to the "end" of the run.
function aiPlay(mode) {
  if (!forNetwork() || view.outcome || aiRun) return;
  if (timer) stop();
  aiRun = { mode, floor: view.obs.floor, act: view.obs.act };
  verdict = null;
  const tick = () => {
    if (!aiRun) return;
    const before = view;
    try {
      aiMove();
    } catch (e) {
      return stopAi();
    }
    const v = view;
    if (v === before) return stopAi(); // it couldn't move (the banner says why)
    const done = mode === "one" || v.outcome || (mode === "floor" && (v.obs.floor !== aiRun.floor || v.obs.act !== aiRun.act));
    if (done) aiRun = null;
    if (aiRun) aiRun.handle = setTimeout(tick, 0);
    else finishAi();
  };
  renderAdvice();
  aiRun.handle = setTimeout(tick, 0);
}

function finishAi() {
  keep();
  renderAdvice();
}

function stopAi() {
  if (!aiRun) return;
  clearTimeout(aiRun.handle);
  aiRun = null;
  finishAi();
}

// ---- The start screen's gallery ----

function drawGallery(names) {
  const have = new Set(names);
  const entries = GALLERY.filter((g) => have.has(g.file));
  // Saved runs the gallery doesn't describe still get a line.
  for (const n of names) if (!GALLERY.some((g) => g.file === n) && !/fights/.test(n)) entries.push({ file: n, title: n.replace(/\.json$/, ""), line: "A saved run." });
  $("gallery").innerHTML = entries
    .map((g) => `<a class="run c-${CHAR_COLOR[g.character] || ""}" href="?run=runs/${esc(g.file)}">
      <b>${esc(g.title)}</b><span>${esc(g.line)}</span>
      ${g.character ? `<em>${esc(capital(g.character))} · A${g.ascension}</em>` : ""}</a>`)
    .join("");
  for (const a of $("gallery").querySelectorAll("a")) {
    a.onclick = (e) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey) return;
      e.preventDefault();
      openUrl(new URL(a.href).searchParams.get("run"));
    };
  }
  $("gallerybox").hidden = entries.length === 0;
  const featured = GALLERY.find((g) => g.featured && have.has(g.file));
  $("watch").disabled = !featured;
  $("watch").onclick = () => featured && openUrl(`runs/${featured.file}`);
  if (!featured) $("watch").title = "The featured run isn't on this server.";
}

// ---- Help and keys ----

/// The Controls panel: the keyboard shortcuts, by what they're for, and how to play with the mouse.
function showHelp() {
  const groups = [
    ["Moving through a run", [
      ["← →", "Back or forward one decision"], ["Home End", "To the start or the end"], ["Space", "Play or pause"],
      ["− +", "Play slower or faster (10% a press)"], ["U", "Undo: back one decision (nothing is dropped: choosing differently starts a variation)"],
    ]],
    ["Playing", [
      ["1–9", "In a fight: pick up that card in your hand; elsewhere: that option"], ["E", "End turn"],
      ["A", "Let the AI take this decision (when an AI plays this character)"],
    ]],
    ["Looking", [["D", "Your deck"], ["M", "The map"], ["Esc", "Put a card back; close a panel or a list"], ["?", "These controls"]]],
  ];
  const keys = (list) => `<dl class="keys">${list.map(([k, v]) => `<dt>${k.split(" ").map((x) => `<kbd>${esc(x)}</kbd>`).join(" ")}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
  modal("Controls", `<div class="controls">${groups.map(([title, list]) => `<section><h3>${esc(title)}</h3>${keys(list)}</section>`).join("")}</div>
    <h3>With the mouse</h3>
    <p class="dim">Play as in the game: click a card to play it, then the enemy to aim it at; click a potion to drink it;
    click an option, a reward, a shop item or a room on the map. Hover over (or tap) anything to see what it does; when
    choosing a card to upgrade, hover over it to see the upgrade. When you watch a run, its next move glows; doing something
    else starts a variation there, as on a chess site, and you play on from there (the AI can too). The decisions list shows
    each variation under the decision it replaces, with where it got to; click any decision, in the run or a variation, to go
    there, and × to delete a variation. Runs you play are kept in this browser until you start another, and the variations
    you play on a run with it; Save run saves them too.</p>`, [{ label: "Close", primary: true }]);
}

/// 1-9: in a fight, the hand's cards (as in the game); elsewhere the screen's clickable things.
function pressNumber(k) {
  if (view.obs.screen.type === "combat" && !view.obs.screen.combat.choice) {
    const el = $("screen").querySelector(`[data-hand="${k}"]`);
    if (el) selectCard(k, el);
    return;
  }
  const el = $("screen").querySelectorAll("[data-act], [data-pick]")[k];
  if (el) el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

function onKey(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA" || e.metaKey || e.altKey) return;
  if (e.key === "Escape") {
    document.body.classList.remove("side-open");
    if ($("modal").hidden && selected) (selected = null), render();
    return closeModal(), hideTip();
  }
  if (e.key === "?") return e.preventDefault(), showHelp();
  if (!$("modal").hidden || !view || $("viewer").hidden) return;
  if (e.ctrlKey && e.key !== "z") return;
  if (e.key === "ArrowLeft" || e.key === "u" || e.key === "U" || (e.ctrlKey && e.key === "z")) (stop(), goto(view.step - 1));
  else if (e.key === "ArrowRight") (stop(), goto(view.step + 1));
  else if (e.key === "Home") (stop(), goto(0));
  else if (e.key === "End") (stop(), goto(view.steps));
  else if (e.key === " " && e.target.tagName !== "BUTTON") togglePlay();
  else if (/^[1-9]$/.test(e.key)) pressNumber(Number(e.key) - 1);
  else if ((e.key === "e" || e.key === "E") && $("screen").querySelector(".endturn")) $("screen").querySelector(".endturn").click();
  else if (e.key === "+" || e.key === "=") setSpeed(speed * 1.1);
  else if (e.key === "-" || e.key === "_") setSpeed(speed / 1.1);
  else if (e.key === "d" || e.key === "D") showDeck();
  else if ((e.key === "a" || e.key === "A") && !$("aione").disabled && !$("advice").hidden) aiPlay("one");
  else if (e.key === "m" || e.key === "M") $("mapbtn") && $("mapbtn").click();
  else return;
  e.preventDefault();
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const t = dark ? "light" : "dark";
  document.documentElement.dataset.theme = t;
  storage((s) => s.setItem(STORE_THEME, t));
}

// ---- Wiring ----

async function main() {
  wireTips();
  // The home page's banner: the four characters facing the Heart.
  $("herobanner").innerHTML = ["ironclad", "silent", "defect", "watcher"].map((c) => `<pre class="art c-${CHAR_COLOR[c]}">${esc(artFor(c, true))}</pre>`).join("")
    + `<span class="vs">vs</span><pre class="art heart">${esc(artFor("CorruptHeart"))}</pre>`;
  $("bannerclose").onclick = () => banner("");
  $("modalclose").onclick = () => closeModal();
  $("modalbody").addEventListener("click", onListClick);
  // A click on the backdrop closes it. (A handler that returns false cancels the click's default,
  // which stopped a click on "Show upgrades"'s words from ticking its box.)
  $("modal").onclick = (e) => {
    if (e.target === $("modal")) closeModal();
  };
  $("theme").onclick = toggleTheme;
  $("help").onclick = showHelp;
  $("homelink").onclick = (e) => {
    if (e.metaKey || e.ctrlKey) return;
    e.preventDefault();
    showHome();
  };
  document.addEventListener("keydown", onKey);
  // Nothing big loads with the page: an engine when a run starts or opens, or when an AI is chosen
  // (with its weights). The AIs' list is small; a kept run waits for it (its AI picks its engine).
  const aiList = loadAiList();
  $("aipick").onchange = () => chooseAi($("aipick").value);
  for (const r of document.querySelectorAll('input[name="character"]')) r.addEventListener("change", renderAiChoices);
  $("advhide").checked = storage((s) => s.getItem(STORE_HIDE)) === "1";
  $("advhide").onchange = () => (storage((s) => s.setItem(STORE_HIDE, $("advhide").checked ? "1" : "0")), (verdict = null), renderAdvice());
  $("aione").onclick = () => (aiRun ? stopAi() : aiPlay("one"));
  $("aifloor").onclick = () => (aiRun ? stopAi() : aiPlay("floor"));
  $("aiend").onclick = () => (aiRun ? stopAi() : aiPlay("end"));
  // A phone starts with the decision log collapsed.
  if (matchMedia(PHONE).matches) $("logbox").open = false;
  // A phone on its side: the AI's advice and the decisions in a panel that slides in (style.css).
  const sidePanel = (open) => (document.body.classList.toggle("side-open", open), $("sidetoggle").setAttribute("aria-expanded", String(open)), open && fitLog());
  $("sidetoggle").onclick = () => sidePanel(!document.body.classList.contains("side-open"));
  $("sideclose").onclick = () => sidePanel(false);
  $("upright").onclick = () => document.body.classList.add("upright");
  // iOS Safari pinch-zooms despite touch-action; while playing, the page keeps its size.
  document.addEventListener("gesturestart", (e) => document.body.classList.contains("viewing") && e.preventDefault());
  // The home page's Add to Home Screen steps, unless this already is the home-screen app.
  const installed = navigator.standalone === true || matchMedia("(display-mode: standalone), (display-mode: fullscreen)").matches;
  $("install").hidden = installed;
  $("logbox").ontoggle = () => view && renderLog();
  $("log").addEventListener("click", onLogClick);
  $("log").addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.matches("[data-i]")) (e.preventDefault(), e.stopPropagation(), onLogClick(e));
  });
  window.addEventListener("resize", () => view && fitLog());
  for (const id of ["file", "file2"]) $(id).onchange = (e) => e.target.files[0] && (openFile(e.target.files[0]), (e.target.value = ""));
  $("newrun").onsubmit = newRun;
  $("playchoice").onclick = () => showSetup();
  $("setupback").onclick = () => (history.state && history.state.setup ? history.back() : showHome());
  $("ascdown").onclick = () => setAscension(Number($("ascension").value) - 1);
  $("ascup").onclick = () => setAscension(Number($("ascension").value) + 1);
  $("ascension").oninput = () => setAscension(Number($("ascension").value));
  window.addEventListener("popstate", () => {
    if (location.hash === "#play") showSetup(false);
    else if (!$("setup").hidden) showHome();
  });
  $("save").onclick = save;
  $("link").onclick = copyLink;
  $("first").onclick = () => (stop(), goto(0));
  $("back").onclick = () => (stop(), goto(view.step - 1));
  $("fwd").onclick = () => (stop(), goto(view.step + 1));
  $("last").onclick = () => (stop(), goto(view.steps));
  $("play").onclick = togglePlay;
  $("slower").onclick = () => setSpeed(speed / 1.1);
  $("faster").onclick = () => setSpeed(speed * 1.1);
  setSpeed(Number(storage((s) => s.getItem(STORE_SPEED))) || 1);
  $("jump").onsubmit = jumpToFloor;
  for (const id of ["screen", "run", "actions"]) $(id).addEventListener("click", onScreenClick);
  $("screen").addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.matches("[data-act], [data-hand], [data-target], [data-pick]")) {
      e.preventDefault();
      e.stopPropagation();
      e.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }
  });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) openFile(f);
  });
  // The gallery: runs/index.json (web/serve.py) lists the saved runs, gallery.json says what they are.
  let names = [];
  try {
    const [g, r] = await Promise.all([fetch("gallery.json"), fetch("runs/index.json")]);
    if (g.ok) GALLERY = await g.json();
    if (r.ok) names = await r.json();
  } catch (e) {
    // Served without the runs: no gallery.
  }
  drawGallery(names);
  // ?run=<url>[&at=<decision>]: that run, at that decision. Else the run this browser kept, if
  // it was open when the page was left. Else the start screen.
  const q = new URLSearchParams(location.search);
  const url = q.get("run");
  if (url) return openUrl(url, Number(q.get("at")) || 0, Number(q.get("line")) || 0);
  if (location.hash === "#play") return showSetup(false);
  const kept = keptRun();
  if (kept && kept.active && kept.run) return (await aiList, resume(kept));
  drawContinue();
}

main();
