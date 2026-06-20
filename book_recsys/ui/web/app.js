const seeds = [];
let sessionId = null;
let queue = [];
let swiping = false;  // lock: ignore input while a /swipe is in flight

const $ = (id) => document.getElementById(id);
const card = $("card");

// ---------- onboarding: search + pick seeds ----------
$("search").addEventListener("input", async (e) => {
  const q = e.target.value.trim();
  if (!q) { $("results").innerHTML = ""; return; }
  const res = await fetch(`/search?q=${encodeURIComponent(q)}`);
  const books = await res.json();
  $("results").innerHTML = "";
  for (const b of books) {
    const li = document.createElement("li");
    li.textContent = b.label;
    li.onclick = () => addSeed(b);
    $("results").appendChild(li);
  }
});

function addSeed(b) {
  if (seeds.some((s) => s.book_id === b.book_id)) return;
  seeds.push(b);
  const span = document.createElement("span");
  span.className = "seed";
  span.textContent = b.label.split(" — ")[0];   // title only on the chip
  $("seeds").appendChild(span);
  $("search").value = "";
  $("results").innerHTML = "";
  $("start").disabled = false;
}

// populate the recommender toggle from /methods (falls back to the server default if empty)
(async () => {
  try {
    const methods = await (await fetch("/methods")).json();
    for (const m of methods) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      $("method").appendChild(opt);
    }
  } catch (e) { /* /methods unavailable -> empty select, session uses the default */ }
})();

// populate the "load saved user" dropdown from /users
async function loadUsers() {
  try {
    const users = await (await fetch("/users")).json();
    $("user").innerHTML = '<option value="">— none —</option>';
    for (const u of users) {
      const opt = document.createElement("option");
      opt.value = u; opt.textContent = u;
      $("user").appendChild(opt);
    }
  } catch (e) { /* /users unavailable -> just the empty option */ }
}
loadUsers();
$("user").onchange = () => { if ($("user").value) $("start").disabled = false; };

// save the current picks (+ any loaded user) under a name, so they can be reloaded later
$("save-user").onclick = async () => {
  const name = $("save-name").value.trim();
  if (!name) return;
  const body = await (await fetch("/session", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked: seeds.map((s) => s.book_id), user: $("user").value }),
  })).json();
  await fetch(`/users/${encodeURIComponent(name)}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: body.session_id }),
  });
  $("save-name").value = "";
  await loadUsers();
  $("user").value = name;
};

$("start").onclick = async () => {
  const res = await fetch("/session", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      liked: seeds.map((s) => s.book_id), user: $("user").value,
      lam: 1.0, k: 10, method: $("method").value,
    }),
  });
  const body = await res.json();
  sessionId = body.session_id;
  queue = body.cards;
  $("onboarding").hidden = true;
  $("modes").hidden = false;
  $("swipe").hidden = false;
  $("actions").hidden = false;
  renderCard();
};

// ---------- mode toggle: swipe <-> chat ----------
function setMode(mode) {
  const chat = mode === "chat";
  $("mode-chat").classList.toggle("active", chat);
  $("mode-swipe").classList.toggle("active", !chat);
  $("swipe").hidden = chat;
  $("actions").hidden = chat;
  $("chat").hidden = !chat;
}
$("mode-swipe").onclick = () => setMode("swipe");
$("mode-chat").onclick = () => setMode("chat");

// ---------- chat: steered AI-overview ----------
async function askChat() {
  const message = $("chat-input").value.trim();
  if (!message) return;
  const out = $("chat-output");
  out.innerHTML = `<p class="thinking">📚 Steering…</p>`;
  try {
    const res = await fetch("/steer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    if (res.status === 503) {
      out.innerHTML = `<p class="thinking">⚠️ AI steering is offline (start Ollama to enable it).</p>`;
      return;
    }
    const data = await res.json();
    sessionId = data.session_id;          // thread the session across turns
    renderSteerState(data.state);
    renderOverview({ intro: data.reply, categories: [{ header: "Picks", items: data.cards }] });
  } catch (e) {
    out.innerHTML = `<p class="thinking">⚠️ Something went wrong — try again.</p>`;
  }
}

function renderSteerState(state) {
  const panel = $("steer-state");
  if (!state) { panel.hidden = true; return; }
  const w = Math.round((state.history_weight ?? 1) * 100);
  const chips = [];
  if (state.topic) chips.push(`topic: ${state.topic}`);
  if (state.genre) chips.push(`genre: ${state.genre}`);
  if (state.anchor_book) chips.push(`like: ${state.anchor_book}`);
  (state.avoid || []).forEach((a) => chips.push(`avoid: ${a}`));
  panel.hidden = false;
  panel.innerHTML =
    `<div class="blend">past reads <b>${w}%</b> ▮ <b>${100 - w}%</b> topic</div>` +
    chips.map((c) => `<span class="chip">${c}</span>`).join("");
}
$("chat-send").onclick = askChat;
$("chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") askChat(); });

function renderOverview(data) {
  const out = $("chat-output");
  if (!data.categories || !data.categories.length) {
    out.innerHTML = `<p class="thinking">No matches found — try rephrasing your request.</p>`;
    return;
  }
  let html = data.intro ? `<p class="intro">${escapeHtml(data.intro)}</p>` : "";
  for (const cat of data.categories) {
    html += `<h3 class="cat">${escapeHtml(cat.header)}</h3>`;
    for (const b of cat.items) {
      const cover = b.image_url
        ? `<img class="ov-cover" alt="" src="${escapeHtml(b.image_url)}" onerror="this.remove()" />`
        : `<div class="ov-cover ov-none">📖</div>`;
      const author = b.author ? ` <span class="ov-author">by ${escapeHtml(b.author)}</span>` : "";
      html += `<div class="ov-item">${cover}<div class="ov-text">` +
        `<div class="ov-title">${escapeHtml(b.title)}${author}</div>` +
        `<div class="ov-reason">${escapeHtml(b.reason)}</div></div></div>`;
    }
  }
  out.innerHTML = html;
}

// ---------- card rendering ----------
function renderCard() {
  resetCard();
  const b = queue[0];
  const body = card.querySelector(".card-body");
  if (!b) {
    body.innerHTML = `<div class="empty"><div class="big">🎉</div>` +
      `<p>That's all for now — swipe more or refine your taste!</p></div>`;
    return;
  }
  const author = b.author ? `<p class="author">by ${escapeHtml(b.author)}</p>` : "";
  const desc = b.description
    ? `<p class="desc">${escapeHtml(b.description)}</p>`
    : `<p class="desc"><em>No synopsis available.</em></p>`;
  const cover = b.image_url
    ? `<div class="cover-wrap"><span class="glow"></span>` +
      `<img class="cover" alt="" src="${escapeHtml(b.image_url)}" ` +
      `onerror="this.closest('.cover-wrap').remove()" draggable="false" /></div>`
    : "";
  body.innerHTML = `${cover}<h2>${escapeHtml(b.title)}</h2>${author}${desc}`;
  card.dataset.bookId = b.book_id;
  window.scrollTo({ top: 0 });   // show the cover for each new card
}

function resetCard() {
  card.style.transition = "transform .3s ease, opacity .3s ease";
  card.style.transform = "";
  card.style.opacity = "1";
  setStamps(0);
}

// ---------- swipe + fly-off ----------
const VEC = { like: [-1, 0], want: [1, 0], dislike: [0, 1], skip: [0, 0.3] };

async function swipe(action) {
  const b = queue[0];
  if (!b || swiping) return;
  swiping = true;
  flyOff(action);
  const res = await fetch("/swipe", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, book_id: b.book_id, action }),
  });
  const body = await res.json();
  queue = body.cards;
  renderReadingList(body.reading_list);
  setTimeout(() => { renderCard(); swiping = false; }, 330);
}

function flyOff(action) {
  const [sx, sy] = VEC[action] || [0, 0];
  card.style.transition = "transform .33s ease, opacity .33s ease";
  card.style.transform = `translate(${sx * 700}px, ${sy * 400}px) rotate(${sx * 18}deg)`;
  card.style.opacity = "0";
}

// ---------- swipe stamps (horizontal drag -> like / want) ----------
const clamp = (v) => Math.max(0, Math.min(1, v));
function setStamps(dx) {
  card.querySelector(".stamp-like").style.opacity = clamp(-dx / 90);
  card.querySelector(".stamp-want").style.opacity = clamp(dx / 90);
  card.querySelector(".stamp-nope").style.opacity = 0;
}

// ---------- buttons + keyboard ----------
document.querySelectorAll("#actions .act").forEach((btn) => {
  btn.onclick = () => swipe(btn.dataset.action);
});
document.addEventListener("keydown", (e) => {
  if ($("swipe").hidden) return;
  const map = { ArrowLeft: "like", ArrowRight: "want", ArrowDown: "dislike" };
  if (map[e.key]) swipe(map[e.key]);
});

// ---------- horizontal drag-to-swipe (vertical drag scrolls the page) ----------
(() => {
  let startX = 0, startY = 0, dragging = false, engaged = false;
  card.addEventListener("pointerdown", (e) => {
    if (swiping) return;
    dragging = true; engaged = false; startX = e.clientX; startY = e.clientY;
  });
  window.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX, dy = e.clientY - startY;
    if (!engaged) {
      if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 6) { dragging = false; return; }
      if (Math.abs(dx) > 6) { engaged = true; card.style.transition = "none"; }
    }
    if (engaged) {
      card.style.transform = `translateX(${dx}px) rotate(${dx * 0.04}deg)`;
      setStamps(dx);
    }
  });
  window.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    dragging = false;
    if (!engaged) return;
    engaged = false;
    if (dx < -90) swipe("like");
    else if (dx > 90) swipe("want");
    else resetCard();   // not far enough -> spring back
  });
})();

// ---------- reading list ----------
function renderReadingList(list) {
  const ul = $("reading-list").querySelector("ul");
  ul.innerHTML = "";
  for (const b of list) {
    const li = document.createElement("li");
    li.textContent = b.label;
    ul.appendChild(li);
  }
  $("rl-count").textContent = list.length;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
