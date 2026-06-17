const seeds = [];
let sessionId = null;
let queue = [];
let swiping = false;  // lock: ignore taps/keys/drags while a /swipe is in flight

const $ = (id) => document.getElementById(id);

// --- onboarding: search + pick seeds ---
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
  span.textContent = b.label;
  $("seeds").appendChild(span);
  $("search").value = "";
  $("results").innerHTML = "";
  $("start").disabled = seeds.length === 0;
}

// --- start session ---
$("start").onclick = async () => {
  const res = await fetch("/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked: seeds.map((s) => s.book_id), lam: 1.0, k: 10 }),
  });
  const body = await res.json();
  sessionId = body.session_id;
  queue = body.cards;
  $("onboarding").hidden = true;
  $("swipe").hidden = false;
  renderCard();
};

// --- swipe loop ---
function renderCard() {
  const card = $("card");
  if (!queue.length) {
    card.innerHTML = "<p>No more recommendations — swipe more or refine your taste!</p>";
    return;
  }
  const b = queue[0];
  card.className = "";
  const author = b.author ? `<p class="author">by ${escapeHtml(b.author)}</p>` : "";
  const desc = b.description
    ? `<p class="desc">${escapeHtml(b.description)}</p>`
    : `<p class="desc"><em>No synopsis available.</em></p>`;
  card.innerHTML = `<h2>${escapeHtml(b.title)}</h2>${author}${desc}`;
  card.dataset.bookId = b.book_id;
}

async function swipe(action) {
  const b = queue[0];
  if (!b || swiping) return;
  swiping = true;
  const dir = { like: "swipe-left", want: "swipe-right", dislike: "swipe-down", skip: "swipe-down" }[action];
  $("card").classList.add(dir);
  const res = await fetch("/swipe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, book_id: b.book_id, action }),
  });
  const body = await res.json();
  queue = body.cards;
  renderReadingList(body.reading_list);
  setTimeout(() => { renderCard(); swiping = false; }, 250);
}

function renderReadingList(list) {
  const ul = $("reading-list").querySelector("ul");
  ul.innerHTML = "";
  for (const b of list) {
    const li = document.createElement("li");
    li.textContent = b.label;
    ul.appendChild(li);
  }
}

document.querySelectorAll("#buttons button").forEach((btn) =>
  (btn.onclick = () => swipe(btn.dataset.action)));

document.addEventListener("keydown", (e) => {
  const map = { ArrowLeft: "like", ArrowRight: "want", ArrowDown: "dislike" };
  if ($("swipe").hidden) return;
  if (map[e.key]) swipe(map[e.key]);
});

// drag-to-swipe (pointer events)
(() => {
  let startX = 0, startY = 0, dragging = false;
  const card = $("card");
  card.addEventListener("pointerdown", (e) => { dragging = true; startX = e.clientX; startY = e.clientY; });
  card.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    dragging = false;
    const dx = e.clientX - startX, dy = e.clientY - startY;
    if (Math.abs(dx) < 60 && Math.abs(dy) < 60) return;
    if (dy > 60 && Math.abs(dy) > Math.abs(dx)) swipe("dislike");
    else if (dx < -60) swipe("like");
    else if (dx > 60) swipe("want");
  });
})();

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
