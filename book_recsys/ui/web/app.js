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

$("start").onclick = async () => {
  const res = await fetch("/session", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked: seeds.map((s) => s.book_id), lam: 1.0, k: 10 }),
  });
  const body = await res.json();
  sessionId = body.session_id;
  queue = body.cards;
  $("onboarding").hidden = true;
  $("swipe").hidden = false;
  renderCard();
};

// ---------- card rendering ----------
function renderCard() {
  resetCard();
  const b = queue[0];
  const body = card.querySelector(".card-body");
  if (!b) {
    body.innerHTML = `<div class="empty"><div class="big">🎉</div>` +
      `<p>That's all for now — swipe more or refine your taste!</p></div>`;
    setPeeks(false);
    return;
  }
  const author = b.author ? `<p class="author">by ${escapeHtml(b.author)}</p>` : "";
  const desc = b.description
    ? `<p class="desc">${escapeHtml(b.description)}</p>`
    : `<p class="desc"><em>No synopsis available.</em></p>`;
  const cover = b.image_url
    ? `<img class="cover" alt="" src="${escapeHtml(b.image_url)}" ` +
      `onerror="this.remove()" draggable="false" />`
    : "";
  body.innerHTML = `${cover}<h2>${escapeHtml(b.title)}</h2>${author}${desc}`;
  card.dataset.bookId = b.book_id;
  card.scrollTop = 0;
  setPeeks(queue.length > 1);
}

function setPeeks(show) {
  document.querySelectorAll(".peek").forEach((p) => { p.style.opacity = show ? "" : "0"; });
}

function resetCard() {
  card.style.transition = "transform .3s ease, opacity .3s ease";
  card.style.transform = "";
  card.style.opacity = "1";
  setStamps(0, 0);
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
  card.style.transform = `translate(${sx * 620}px, ${sy * 620}px) rotate(${sx * 22}deg)`;
  card.style.opacity = "0";
}

// ---------- swipe stamps (driven by drag distance) ----------
const clamp = (v) => Math.max(0, Math.min(1, v));
function setStamps(dx, dy) {
  card.querySelector(".stamp-like").style.opacity = clamp(-dx / 90);
  card.querySelector(".stamp-want").style.opacity = clamp(dx / 90);
  card.querySelector(".stamp-nope").style.opacity = clamp(dy / 90);
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

// ---------- drag-to-swipe (pointer events) ----------
(() => {
  let startX = 0, startY = 0, dragging = false;
  card.addEventListener("pointerdown", (e) => {
    if (swiping) return;
    dragging = true; startX = e.clientX; startY = e.clientY;
    card.style.transition = "none";
    card.setPointerCapture(e.pointerId);
  });
  card.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX, dy = e.clientY - startY;
    card.style.transform = `translate(${dx}px, ${dy}px) rotate(${dx * 0.05}deg)`;
    setStamps(dx, dy);
  });
  card.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    dragging = false;
    const dx = e.clientX - startX, dy = e.clientY - startY;
    if (dy > 90 && dy > Math.abs(dx)) swipe("dislike");
    else if (dx < -90) swipe("like");
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
