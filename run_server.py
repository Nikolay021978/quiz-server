#!/usr/bin/env python3
# run_server.py
#
# Quiz server with topics stored in ./topics/*.json
# - HTTP + WebSocket (aiohttp)
# - Client UI (embedded fallback if client.html missing)
# - Supports image questions: payload.image (full) and payload.image_thumb (generated thumbnail)
# - Generates cached thumbnails with EXIF orientation fix (Pillow)
# - Endpoint POST /api/reload-topics to reload topics at runtime
#
# Usage:
#   python run_server.py --host 0.0.0.0 --http-port 8000
#
# Requires: aiohttp, pillow
# Install: python -m pip install aiohttp pillow

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import glob
import pathlib
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple

from aiohttp import web, WSMsgType
from PIL import Image, ImageOps

# ---------------------------
# Configuration
# ---------------------------
TURN_TIME = 12
ROUND_DELAY = 0.8
RESTART_DELAY = 2.0
NUM_QUESTIONS = 5

BASE_DIR = os.path.dirname(__file__) or "."
STATIC_DIR = os.path.join(BASE_DIR, "static")  # serve static files from ./static
TOPICS_DIR = os.path.join(BASE_DIR, "topics")
RATINGS_FILE = os.path.join(BASE_DIR, "ratings.json")
THUMBS_DIR = os.path.join(STATIC_DIR, "thumbs")
DEFAULT_WS_PATH = "/ws"
DEFAULT_INDEX = "client.html"

# Scoring
BASE_POINTS = 1000
MAX_SPEED_BONUS = 1000
MAX_STREAK_MULT = 3

# Ensure folders exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TOPICS_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quizserver")

# ---------------------------
# Topics loading helpers
# ---------------------------
def _ensure_topics_dir():
    try:
        os.makedirs(TOPICS_DIR, exist_ok=True)
    except Exception:
        log.exception("Failed to ensure topics dir %s", TOPICS_DIR)

def _load_single_topic_file(path: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """
    Parse topic file and return (topic_name, list[(term, definition_or_image)]).
    Accepts:
      - {"name":"...", "items":[ ["term","def"], {"term":"...","image":"..."} ]}
      - legacy arrays
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        log.exception("Failed to read/parse topic file %s", path)
        return None

    if isinstance(doc, dict):
        name = doc.get("name") or os.path.splitext(os.path.basename(path))[0]
        items_raw = doc.get("items", [])
    elif isinstance(doc, list):
        name = os.path.splitext(os.path.basename(path))[0]
        items_raw = doc
    else:
        return None

    parsed: List[Tuple[str, str]] = []
    for it in items_raw:
        if isinstance(it, list) and len(it) >= 2:
            term = str(it[0])
            definition = "" if it[1] is None else str(it[1])
            if term:
                parsed.append((term, definition))
        elif isinstance(it, dict):
            # normalize fields and prefer explicit image field if present
            term = it.get("term") or it.get("name") or ""
            image = it.get("image") or it.get("img")
            definition = it.get("definition") or it.get("def") or ""
            if image and isinstance(image, str) and image.strip():
                parsed.append((str(term), str(image)))
            else:
                if term:
                    parsed.append((str(term), str(definition)))
        else:
            continue

    if not parsed:
        return None
    return (str(name), parsed)

def load_topics_from_dir() -> Dict[str, List[Tuple[str, str]]]:
    _ensure_topics_dir()
    topics: Dict[str, List[Tuple[str, str]]] = {}
    pattern = os.path.join(TOPICS_DIR, "*.json")
    for path in sorted(glob.glob(pattern), key=os.path.basename):
        res = _load_single_topic_file(path)
        if res:
            tname, items = res
            if tname in topics:
                log.warning("Duplicate topic name %s from file %s — skipping", tname, path)
                continue
            topics[tname] = items
            sample_img_count = sum(1 for _, d in items if isinstance(d, str) and d.startswith('/static/'))
            log.info("Loaded topic '%s' (%d items, %d image-paths) from %s", tname, len(items), sample_img_count, path)
    return topics

# ---------------------------
# Ratings persistence helpers
# ---------------------------
ratings_lock = asyncio.Lock()

def _ensure_ratings_file():
    if not os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except Exception:
            log.exception("Failed to create ratings file")

async def load_ratings() -> Dict[str, Dict[str, Dict[str, int]]]:
    async with ratings_lock:
        _ensure_ratings_file()
        try:
            with open(RATINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            log.exception("Failed to load ratings.json")
            return {}

async def save_ratings(ratings: Dict[str, Dict[str, Dict[str, int]]]):
    async with ratings_lock:
        try:
            with open(RATINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(ratings, f, ensure_ascii=False, indent=2)
        except Exception:
            log.exception("Failed to save ratings.json")

async def update_ratings_after_game(topic: str, per_player_final: List[Dict]):
    ratings = await load_ratings()
    topic_map = ratings.get(topic, {})
    for r in per_player_final:
        name = r.get("name")
        if not name:
            continue
        entry = topic_map.get(name, {"total_points": 0, "games": 0, "total_correct": 0})
        entry["total_points"] = entry.get("total_points", 0) + int(r.get("points", 0))
        entry["games"] = entry.get("games", 0) + 1
        entry["total_correct"] = entry.get("total_correct", 0) + int(r.get("correct", 0))
        topic_map[name] = entry
    ratings[topic] = topic_map
    await save_ratings(ratings)

async def get_ratings_for_topic(topic: str, top_n: Optional[int] = None) -> List[Dict]:
    ratings = await load_ratings()
    topic_map = ratings.get(topic, {})
    items = []
    for name, stats in topic_map.items():
        items.append({
            "name": name,
            "total_points": int(stats.get("total_points", 0)),
            "games": int(stats.get("games", 0)),
            "total_correct": int(stats.get("total_correct", 0))
        })
    items.sort(key=lambda x: -x["total_points"])
    if top_n:
        return items[:top_n]
    return items

# ---------------------------
# Game state & logic
# ---------------------------
class Player:
    def __init__(self, name: str, ws: web.WebSocketResponse):
        self.name = name
        self.ws = ws
        self.score = 0
        self.points = 0
        self.streak = 0
        self.join_ts = time.time()

class QuizServer:
    def __init__(self, topics_map: Dict[str, List[Tuple[str, str]]]):
        self.players: "OrderedDict[str, Player]" = OrderedDict()
        self.lock = asyncio.Lock()
        self.topics_map = topics_map or {}
        self.available_topics = list(self.topics_map.keys())
        self.current_topic: str = self.available_topics[0] if self.available_topics else ""
        self.questions: List[Tuple[str, str]] = self._prepare_questions(self.current_topic)
        self.current_qidx: int = 0
        self.current_choices: Optional[List[str]] = None
        self.current_correct: Optional[int] = None
        self.answers: Dict[str, Tuple[int, float]] = {}
        self.game_running: bool = False
        self.admin: Optional[str] = None
        self._game_task: Optional[asyncio.Task] = None
        self.current_question_start_ts: Optional[float] = None

        self._all_terms: List[str] = []
        for items in self.topics_map.values():
            for t, _ in items:
                if t not in self._all_terms:
                    self._all_terms.append(t)

    def _prepare_questions(self, topic: str) -> List[Tuple[str, str]]:
        items = self.topics_map.get(topic, [])
        if not items:
            return []
        k = min(len(items), NUM_QUESTIONS)
        return random.sample(items, k=k)

    async def send(self, player: Player, msg: dict):
        try:
            await player.ws.send_str(json.dumps(msg))
        except Exception:
            log.debug("send failed to %s", player.name)

    async def broadcast(self, msg: dict):
        js = json.dumps(msg)
        dead = []
        async with self.lock:
            players = list(self.players.items())
        for name, pl in players:
            try:
                await pl.ws.send_str(js)
            except Exception:
                dead.append(name)
        if dead:
            async with self.lock:
                for d in dead:
                    if d in self.players:
                        try:
                            await self.players[d].ws.close()
                        except Exception:
                            pass
                        del self.players[d]

    async def broadcast_lobby(self):
        async with self.lock:
            players = list(self.players.keys())
            admin = self.admin
            current_topic = self.current_topic
            topics = self.available_topics
        await self.broadcast({
            "type": "lobby",
            "players": players,
            "admin": admin,
            "topic": current_topic,
            "topics": topics,
            "num_questions": NUM_QUESTIONS
        })

    async def add_player(self, name: str, ws: web.WebSocketResponse) -> str:
        async with self.lock:
            base = name
            i = 1
            while name in self.players:
                name = f"{base}_{i}"; i += 1
            pl = Player(name, ws)
            self.players[name] = pl
            if not self.admin:
                self.admin = name
                log.info("Assigned admin: %s", name)
            log.info("Player joined: %s (total %d)", name, len(self.players))
        await self.send(pl, {
            "type": "joined",
            "name": name,
            "is_admin": (name == self.admin),
            "topic": self.current_topic,
            "topics": self.available_topics,
            "num_questions": NUM_QUESTIONS
        })
        await self.broadcast_lobby()
        return name

    async def remove_player(self, name: str):
        async with self.lock:
            if name in self.players:
                try:
                    await self.players[name].ws.close()
                except Exception:
                    pass
                del self.players[name]
                log.info("Player left: %s", name)
                if name == self.admin:
                    self.admin = next(iter(self.players.keys()), None)
                    log.info("New admin: %s", self.admin)
        await self.broadcast_lobby()

    async def handle_message(self, name: str, msg: dict):
        t = msg.get("type")
        if t == "choose_topic":
            topic = msg.get("topic")
            async with self.lock:
                if name != self.admin:
                    return
                if self.game_running:
                    return
                if topic not in self.available_topics:
                    return
                self.current_topic = topic
                self.questions = self._prepare_questions(self.current_topic)
                self.current_qidx = 0
            await self.broadcast_lobby()
            return

        if t == "start_game":
            async with self.lock:
                if name != self.admin:
                    return
                if self.game_running:
                    return
                if not self.questions:
                    self.questions = self._prepare_questions(self.current_topic)
                self.game_running = True
                self._game_task = asyncio.create_task(self._run_game())
            return

        if t == "answer":
            async with self.lock:
                if not self.game_running:
                    return
                if name in self.answers:
                    return
                choice = msg.get("choice")
                try:
                    self.answers[name] = (int(choice), time.time())
                except Exception:
                    pass
            return

        if t == "get_ratings":
            topic = self.current_topic
            ratings = await get_ratings_for_topic(topic)
            if name in self.players:
                await self.send(self.players[name], {"type": "ratings", "topic": topic, "ratings": ratings})
            return

    async def _run_game(self):
        async with self.lock:
            self.current_qidx = 0
            for p in self.players.values():
                p.score = 0
                p.points = 0
                p.streak = 0
        await self.broadcast({"type": "game_started", "topic": self.current_topic, "num_questions": NUM_QUESTIONS})

        total = len(self.questions)
        try:
            while True:
                async with self.lock:
                    if not self.players:
                        self.game_running = False
                        break
                    if self.current_qidx >= total:
                        break
                    term, definition = self.questions[self.current_qidx]

                    # --- Формирование вариантов только из текущей темы ---
                    same_topic_terms = [t for t, _ in self.topics_map.get(self.current_topic, []) if t != term]
                    random.shuffle(same_topic_terms)

                    # возьмём до 3 неверных вариантов из той же темы
                    wrong_choices = same_topic_terms[:3]

                    # Если в теме меньше 3 других терминов, дополним повторами случайных элементов той же темы
                    if len(wrong_choices) < 3 and same_topic_terms:
                        i = 0
                        while len(wrong_choices) < 3:
                            wrong_choices.append(same_topic_terms[i % len(same_topic_terms)])
                            i += 1

                    # Если тема содержит только текущий термин (нет других), создаём плейсхолдеры
                    while len(wrong_choices) < 3:
                        wrong_choices.append("—")

                    choices = wrong_choices[:3] + [term]
                    random.shuffle(choices)

                    self.current_choices = choices
                    self.current_correct = choices.index(term)
                    self.answers = {}
                    qidx = self.current_qidx

                payload = {
                    "type": "new_question",
                    "question_idx": qidx,
                    "choices": self.current_choices,
                    "turn_time": TURN_TIME,
                    "server_ts": time.time()
                }
                # If definition looks like an internal static path, send as image and thumbnail
                if isinstance(definition, str) and definition.startswith("/static/"):
                    payload["image"] = definition
                    basename = os.path.basename(definition)
                    # thumbnail URL: width 420, preserve aspect (height 0)
                    payload["image_thumb"] = f"/thumbs/420x0/{basename}"
                else:
                    payload["term"] = term
                    payload["definition"] = definition

                self.current_question_start_ts = time.time()
                await self.broadcast(payload)

                for sec in range(TURN_TIME, 0, -1):
                    await self.broadcast({"type": "tick", "remaining": sec, "turn_time": TURN_TIME})
                    await asyncio.sleep(1)

                results = []
                question_start = self.current_question_start_ts or (time.time() - TURN_TIME)
                async with self.lock:
                    for pname, p in list(self.players.items()):
                        ans = self.answers.get(pname)
                        if ans:
                            choice_idx, ans_ts = int(ans[0]), float(ans[1])
                            ok = (choice_idx == self.current_correct)
                        else:
                            choice_idx, ans_ts = None, None
                            ok = False

                        points_earned = 0
                        if ok:
                            p.score += 1
                            p.streak += 1
                            multiplier = min(MAX_STREAK_MULT, p.streak)
                            elapsed = max(0.0, ans_ts - question_start)
                            remaining = max(0.0, TURN_TIME - elapsed)
                            speed_bonus = int((remaining / TURN_TIME) * MAX_SPEED_BONUS)
                            points_earned = int((BASE_POINTS + speed_bonus) * multiplier)
                            p.points += points_earned
                        else:
                            p.streak = 0

                        results.append({
                            "name": pname,
                            "choice": choice_idx,
                            "correct": ok,
                            "correct_count": p.score,
                            "points_earned": points_earned,
                            "points": p.points,
                            "streak": p.streak
                        })

                leader_by_points = sorted(
                    [{"name": pname, "points": p.points, "correct": p.score} for pname, p in self.players.items()],
                    key=lambda x: -x["points"]
                )

                await self.broadcast({
                    "type": "turn_result",
                    "question_idx": qidx,
                    "correct_index": self.current_correct,
                    "results": results,
                    "leaderboard": leader_by_points
                })

                self.current_qidx += 1
                await asyncio.sleep(ROUND_DELAY)
        finally:
            async with self.lock:
                final_lb = sorted(
                    [{"name": n, "correct": p.score, "points": p.points} for n, p in self.players.items()],
                    key=lambda x: -x["points"]
                )
                self.game_running = False
                self._game_task = None
                self.current_qidx = 0
                self.current_choices = None
                self.current_correct = None
                self.answers = {}
                self.current_question_start_ts = None

            try:
                await update_ratings_after_game(self.current_topic, final_lb)
            except Exception:
                log.exception("Failed to update ratings after game")

            await self.broadcast({"type": "game_over", "leaderboard": final_lb})
            await self.broadcast_lobby()

# ---------------------------
# Client HTML: embedded fallback (full UI)
# ---------------------------
def create_embedded_client_html() -> str:
    return r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quiz client</title>
<style>
:root{
  --bg:#0b0b0b; --card:#161616; --accent:#0066ff;
  --ok:#00c853; --bad:#ff1744; --muted:#bdbdbd; --glass: rgba(255,255,255,0.03);
  --gutter:12px;
  --control-height:44px;
}
*{box-sizing:border-box}
html, body {
  overscroll-behavior: none;
  touch-action: manipulation;
}
body{margin:0;font-family:system-ui,Segoe UI,Roboto,Arial;background:var(--bg);color:#fff}
.container{max-width:1100px;margin:12px auto;padding:16px;display:flex;flex-direction:column;gap:16px}

/* TOP: two columns */
.top-grid{
  display:grid;
  grid-template-columns: 1fr 200px;
  gap:var(--gutter);
  align-items:center;
  width:100%;
}
.left-stack{display:flex;flex-direction:column;gap:var(--gutter)}
.right-stack{display:flex;flex-direction:column;gap:var(--gutter);align-items:stretch;justify-content:center}

/* fields */
.field input, .field select{
  height: var(--control-height);
  line-height: normal;
  width:100%;padding:0 12px;border-radius:10px;border:0;background:var(--glass);color:#fff;font-size:15px;
  display:flex;align-items:center;
}

/* buttons */
.btn{
  height: var(--control-height);
  border-radius:10px;border:0;font-size:15px;cursor:pointer;padding:0 14px;
  display:inline-flex;align-items:center;justify-content:center;
}
.btn.full{width:100%}
#join{background:var(--accent);color:#fff}
#start{background:#22c55e;color:#fff}
#rating{background:#ffb300;color:#111}

/* Card / question area */
.card{padding:18px;background:var(--card);border-radius:12px;margin-top:8px;display:flex;flex-direction:column;gap:12px}
.qtext{font-size:20px;text-align:center;margin:6px 0}
.meta{font-size:13px;color:var(--muted);text-align:center}
.image-wrap{text-align:center}
.image-wrap img{max-width:100%;height:auto;border-radius:8px}
.progress{height:12px;background:rgba(255,255,255,0.04);border-radius:8px;overflow:hidden;margin:8px auto 12px auto;width:100%}
.progress>i{display:block;height:100%;background:linear-gradient(90deg,#ef5350,#29b6f6);width:0%;transition:width 300ms linear}

.choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:0 auto;width:100%;max-width:920px}
.choice{padding:14px;border-radius:12px;font-weight:800;color:#fff;text-align:center;cursor:pointer;user-select:none;outline:none;box-shadow:0 6px 18px rgba(0,0,0,0.6)}
.choice.disabled{opacity:0.7;pointer-events:none}.choice.pressed{transform:translateY(3px)}
.choice.correct{border:4px solid var(--ok)}.choice.wrong{border:4px solid var(--bad)}
.a{background:#ef5350}.b{background:#ffd54f;color:#111}.c{background:#66bb6a}.d{background:#29b6f6;color:#111}

/* modals */
.modal{position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,0.6);display:none;align-items:center;justify-content:center;z-index:9999}
.box{background:#101010;padding:18px;border-radius:12px;max-width:760px;width:95%}
.results-table{width:100%;border-collapse:collapse}
.results-table th,.results-table td{padding:10px;border-bottom:1px solid rgba(255,255,255,0.03);text-align:left}
.result-1{background:linear-gradient(90deg,rgba(255,215,0,0.08),transparent)}

/* Responsive adjustments */
@media (max-width:920px){
  .top-grid{grid-template-columns: minmax(140px, 0.62fr) minmax(80px, 120px);gap:10px}
  .container{padding:12px}
  :root{--control-height:42px}
  .btn{font-size:14px}
  .field input,.field select{font-size:14px}
}
@media (max-width:720px){
  .top-grid{grid-template-columns: minmax(120px, 1fr) minmax(70px, 100px);gap:8px;align-items:center}
  .right-stack{gap:8px}
  :root{--control-height:40px}
  .btn{font-size:13px;padding:0 10px}
  .field input,.field select{font-size:13px}
  .container{padding:10px}
}
</style>
</head>
<body>
<div class="container">
  <div class="top-grid">
    <div class="left-stack">
      <div class="field"><input id="name" placeholder="Ваше имя" aria-label="Ваше имя"></div>
      <div class="field"><select id="topic" aria-label="Выбор темы"><option value="" disabled selected>Выбор темы</option></select></div>
      <div class="field"><select id="format" aria-label="Выбор формата"><option value="game" selected>Выбор формата</option><option value="study">Обучение</option></select></div>
    </div>

    <div class="right-stack">
      <button id="join" class="btn full">Connect</button>
      <button id="start" class="btn full">Старт</button>
      <button id="rating" class="btn full">Рейтинг</button>
    </div>
  </div>

  <div class="card" id="gameCard" aria-live="polite">
    <div class="qtext" id="qtext">Ожидание игры...</div>
    <div class="meta" id="metaInQ">Вопросов в игре: 0</div>
    <div class="image-wrap" id="imageWrap" style="display:none"></div>

    <div class="progress" id="progress" style="display:none"><i id="progressBar"></i></div>

    <div class="choices" id="choices" aria-hidden="true">
      <div class="choice a" id="c0">A</div>
      <div class="choice b" id="c1">B</div>
      <div class="choice c" id="c2">C</div>
      <div class="choice d" id="c3">D</div>
    </div>

    <div id="playersBox" style="margin-top:12px;color:var(--muted)"></div>
  </div>
</div>

<!-- Ratings modal -->
<div id="ratingsModal" class="modal"><div class="box">
  <div style="display:flex;justify-content:space-between;align-items:center"><h3 id="ratingsTitle" style="margin:0">Рейтинг</h3><button id="closeRatings" class="btn" style="height:36px">Закрыть</button></div>
  <table class="results-table" id="ratingsTable"><thead><tr><th>#</th><th>Игрок</th><th>Очки</th><th>Верных</th></tr></thead><tbody id="ratingsBody"></tbody></table>
</div></div>

<!-- Final modal -->
<div id="finalModal" class="modal"><div class="box">
  <div style="display:flex;justify-content:space-between;align-items:center"><h3 style="margin:0">Итоги игры</h3><button id="closeFinal" class="btn" style="height:36px">Закрыть</button></div>
  <table class="results-table" id="finalTable"><thead><tr><th>#</th><th>Игрок</th><th>Очки</th><th>Верных</th></tr></thead><tbody id="finalBody"></tbody></table>
</div></div>

<script>
/* Prevent pull-to-refresh on iOS Safari */
(function disablePullToRefresh() {
  let startY = 0;
  let maybePrevent = false;

  document.addEventListener('touchstart', function(e) {
    if (e.touches.length !== 1) return;
    startY = e.touches[0].screenY;
    maybePrevent = (window.scrollY === 0);
  }, {passive: true});

  document.addEventListener('touchmove', function(e) {
    if (!maybePrevent) return;
    const curY = e.touches[0].screenY;
    const deltaY = curY - startY;
    if (deltaY > 10) {
      e.preventDefault();
    }
  }, {passive: false});
})();

// Prevent common keyboard refresh shortcuts (F5, Ctrl+R / Cmd+R)
window.addEventListener('keydown', function(e) {
  if (e.key === 'F5' || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'r')) {
    e.preventDefault();
  }
});

/* Full client JS: WebSocket, events, UI updates */
const TURN_TIME = 12;
const nameInput = document.getElementById('name');
const joinBtn = document.getElementById('join');
const startBtn = document.getElementById('start');
const ratingBtn = document.getElementById('rating');
const topicSelect = document.getElementById('topic');
const formatSelect = document.getElementById('format');
const qtext = document.getElementById('qtext');
const metaInQ = document.getElementById('metaInQ');
const imageWrap = document.getElementById('imageWrap');
const progress = document.getElementById('progress');
const progressBar = document.getElementById('progressBar');
const choicesEls = [document.getElementById('c0'),document.getElementById('c1'),document.getElementById('c2'),document.getElementById('c3')];
const playersBox = document.getElementById('playersBox');

const ratingsModal = document.getElementById('ratingsModal');
const ratingsBody = document.getElementById('ratingsBody');
const closeRatings = document.getElementById('closeRatings');

const finalModal = document.getElementById('finalModal');
const finalBody = document.getElementById('finalBody');
const closeFinal = document.getElementById('closeFinal');

let ws = null;
let showPlayers = false;
let amIAdmin = false;
let lastTurnTime = TURN_TIME;

function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }
function setAdminVisible(v){ amIAdmin = !!v; startBtn.style.display = v ? 'inline-block' : 'none'; ratingBtn.style.display = v ? 'inline-block' : 'none'; }

function populateTopics(topics, current){
  const prev = topicSelect.value || "";
  topicSelect.innerHTML = '<option value="" disabled>Выбор темы</option>';
  topics.forEach(t => {
    const opt = document.createElement('option'); opt.value = t; opt.textContent = t;
    if(t === current) opt.selected = true;
    topicSelect.appendChild(opt);
  });
  if(prev) topicSelect.value = prev;
}

function showPlayersList(names){
  if(!showPlayers) return;
  playersBox.innerHTML = '<strong>Присоединившиеся:</strong><ul>' + (names||[]).map(n => '<li>' + escapeHtml(n) + '</li>').join('') + '</ul>';
  playersBox.style.display = 'block';
}
function hidePlayersList(){ playersBox.style.display='none'; playersBox.innerHTML=''; }

function enableChoices(){
  choicesEls.forEach(el => { el.classList.remove('disabled','pressed','correct','wrong'); el.style.pointerEvents='auto'; el.style.opacity='1'; });
  progress.style.display = 'block';
  progressBar.style.width = '100%';
}
function disableChoices(){
  choicesEls.forEach(el => { el.classList.add('disabled'); el.style.pointerEvents='none'; });
  progress.style.display = 'none';
}

function onChoiceClick(idx){
  if(!ws || ws.readyState !== 1) return;
  choicesEls.forEach(el=>el.classList.remove('pressed'));
  choicesEls[idx].classList.add('pressed');
  ws.send(JSON.stringify({type:'answer', choice: idx}));
  disableChoices();
}

joinBtn.addEventListener('click', () => {
  const name = nameInput.value.trim(); if(!name){ alert('Введите имя'); return; }
  if(ws && ws.readyState === 1){ alert('Уже подключены'); return; }
  try{ ws = new WebSocket(location.origin.replace(/^http/, 'ws') + '/ws'); } catch(e){ alert('Ошибка WebSocket'); return; }
  showPlayers = true;
  ws.onopen = ()=> { ws.send(JSON.stringify({type:'join', name})); };
  ws.onmessage = ev => { try{ const m = JSON.parse(ev.data); handleMessage(m); }catch(e){} };
  ws.onclose = ()=> { showPlayers=false; hidePlayersList(); ws=null; };
  ws.onerror = ()=> {};
});

startBtn.addEventListener('click', () => {
  if(!ws || ws.readyState !== 1){ alert('Нет соединения с сервером'); return; }
  if(!amIAdmin){ alert('Только админ может стартовать игру'); return; }
  ws.send(JSON.stringify({type:'start_game', format: formatSelect.value || 'game'}));
  metaInQ.textContent = '';
  showPlayers = false;
  hidePlayersList();
});

ratingBtn.addEventListener('click', () => {
  if(!ws || ws.readyState !== 1){ alert('Нет соединения с сервером'); return; }
  ws.send(JSON.stringify({type:'get_ratings'}));
});

closeRatings.addEventListener('click', ()=>{ ratingsModal.style.display = 'none'; });
closeFinal.addEventListener('click', ()=>{ finalModal.style.display = 'none'; });

topicSelect.addEventListener('change', () => {
  if(!ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({type:'choose_topic', topic: topicSelect.value}));
});

choicesEls.forEach((el, idx) => el.addEventListener('click', () => onChoiceClick(idx)));

function updateProgress(remaining, total){
  const t = total || lastTurnTime || TURN_TIME;
  const pct = Math.max(0, (remaining / t) * 100);
  progressBar.style.width = pct + '%';
  if(remaining <= 0) progress.style.display = 'none'; else progress.style.display = 'block';
}

function setQuestion(textOrDefinition, choices, turn_time, imageSrc){
  enableChoices();
  if(imageSrc){
    imageWrap.innerHTML = '<img src="'+escapeHtml(imageSrc)+'" alt="img">';
    imageWrap.style.display = 'block';
    qtext.textContent = '';
  } else {
    imageWrap.style.display = 'none';
    qtext.textContent = textOrDefinition || '';
  }
  for(let i=0;i<4;i++){
    const text = choices && choices[i] ? choices[i] : '';
    choicesEls[i].textContent = String.fromCharCode(65+i) + ': ' + text;
  }
  lastTurnTime = turn_time || TURN_TIME;
  progressBar.style.width = '100%';
  progress.style.display = 'block';
}

function applyTurnResult(correctIndex, results){
  for(let i=0;i<4;i++){
    choicesEls[i].classList.remove('pressed','wrong','correct');
    choicesEls[i].style.opacity = (i === correctIndex ? '1' : '0.6');
  }
  if(Number.isInteger(correctIndex) && choicesEls[correctIndex]){
    choicesEls[correctIndex].classList.add('correct');
  }
  const pressed = document.querySelector('.choice.pressed');
  if(pressed){
    const pressedIdx = choicesEls.indexOf(pressed);
    if(pressedIdx !== -1 && pressedIdx !== correctIndex){
      choicesEls[pressedIdx].classList.add('wrong');
    }
  }
  disableChoices();
}

function renderFinalTable(finalList){
  finalBody.innerHTML = '';
  (finalList || []).forEach((item, idx) => {
    const tr = document.createElement('tr');
    let cls = '';
    let medal = '';
    if(idx === 0){ cls = 'result-1'; medal = '🏆 '; }
    else if(idx === 1){ medal = '🥈 '; }
    else if(idx === 2){ medal = '🥉 '; }
    tr.className = cls;
    tr.innerHTML = '<td>' + (idx+1) + '</td>'
                 + '<td>' + medal + escapeHtml(item.name || '') + '</td>'
                 + '<td>' + (item.points||0) + '</td>'
                 + '<td>' + (item.correct||0) + '</td>';
    finalBody.appendChild(tr);
  });
  finalModal.style.display = 'flex';
}

function renderRatings(ratings, topic){
  ratingsBody.innerHTML = '';
  (ratings || []).forEach((r, idx) => {
    const tr = document.createElement('tr');
    let medal = '';
    if(idx === 0) medal = '🏆 ';
    else if(idx === 1) medal = '🥈 ';
    else if(idx === 2) medal = '🥉 ';
    tr.innerHTML = '<td>' + (idx+1) + '</td>'
                 + '<td>' + medal + escapeHtml(r.name || '') + '</td>'
                 + '<td>' + (r.total_points||0) + '</td>'
                 + '<td>' + (r.total_correct||0) + '</td>';
    ratingsBody.appendChild(tr);
  });
  ratingsModal.style.display = 'flex';
}

function handleMessage(msg){
  if(!msg || typeof msg !== 'object') return;
  if(msg.type === 'joined'){
    setAdminVisible(!!msg.is_admin);
    if(msg.topics) populateTopics(msg.topics, msg.topic || '');
    if(msg.num_questions != null) metaInQ.textContent = 'Вопросов в игре: ' + msg.num_questions;
    return;
  }
  if(msg.type === 'lobby'){
    if(msg.topics) populateTopics(msg.topics, msg.topic || '');
    if(msg.num_questions != null) metaInQ.textContent = 'Вопросов в игре: ' + msg.num_questions;
    if(showPlayers) showPlayersList(msg.players || []);
    return;
  }
  if(msg.type === 'game_started'){
    metaInQ.textContent = '';
    hidePlayersList();
    enableChoices();
    return;
  }
  if(msg.type === 'new_question'){
    const imageSrc = msg.image_thumb || msg.image || null;
    setQuestion(msg.definition || msg.term || '', msg.choices || [], msg.turn_time || TURN_TIME, imageSrc);
    return;
  }
  if(msg.type === 'tick'){
    updateProgress(msg.remaining != null ? msg.remaining : 0, msg.turn_time || lastTurnTime);
    return;
  }
  if(msg.type === 'turn_result'){
    applyTurnResult(msg.correct_index, msg.results || []);
    return;
  }
  if(msg.type === 'game_over'){
    renderFinalTable(msg.leaderboard || []);
    return;
  }
  if(msg.type === 'ratings'){
    renderRatings(msg.ratings || [], msg.topic || '');
    return;
  }
}

window.addEventListener('beforeunload', () => { try{ if(ws) ws.close(); } catch(e){} });
</script>
</body>
</html>
"""

# ---------------------------
# HTTP + WebSocket handlers (complete web part)
# ---------------------------
quiz: Optional[QuizServer] = None

async def ws_handler(request):
    global quiz
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    name: Optional[str] = None
    try:
        # First message must be join
        join_msg = await ws.receive()
        if join_msg.type != WSMsgType.TEXT:
            await ws.close()
            return ws
        try:
            j = json.loads(join_msg.data)
        except Exception:
            await ws.close()
            return ws
        if j.get("type") != "join" or not j.get("name"):
            await ws.close()
            return ws
        name = await quiz.add_player(j["name"], ws)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                await quiz.handle_message(name, data)
            elif msg.type == WSMsgType.ERROR:
                log.warning("ws connection closed with exception %s", ws.exception())
                break
    except Exception as e:
        log.exception("ws handler exception: %s", e)
    finally:
        if name:
            await quiz.remove_player(name)
        try:
            await ws.close()
        except Exception:
            pass
    return ws

def get_index_path() -> Optional[str]:
    p = os.path.join(BASE_DIR, DEFAULT_INDEX)
    return p if os.path.exists(p) else None

async def index_handler(request):
    idx_path = get_index_path()
    if idx_path:
        resp = web.FileResponse(idx_path)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    else:
        return web.Response(text=create_embedded_client_html(), content_type="text/html")

async def static_handler(request):
    rel = request.match_info.get('path', '') or request.match_info.get('name', '')
    if '..' in rel or rel.startswith('/'):
        raise web.HTTPNotFound()
    candidates = [
        os.path.join(STATIC_DIR, rel),
        os.path.join(BASE_DIR, rel),
        os.path.join(STATIC_DIR, "uploads", rel),
        os.path.join(BASE_DIR, "uploads", rel)
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            resp = web.FileResponse(path)
            resp.headers['Cache-Control'] = 'no-store'
            return resp
    raise web.HTTPNotFound()

# Thumbnail handler with EXIF orientation fix
async def thumb_handler(request):
    """
    GET /thumbs/{size}/{name}
    size: WIDTHxHEIGHT e.g. 420x0 (0 = preserve aspect)
    name: basename of file under static/uploads or uploads
    """
    size = request.match_info.get('size', '')
    name = request.match_info.get('name', '')
    if '..' in name or name.startswith('/'):
        raise web.HTTPNotFound()
    try:
        if 'x' not in size:
            raise ValueError()
        w_s, h_s = size.split('x', 1)
        w = int(w_s) if w_s.isdigit() else 0
        h = int(h_s) if h_s.isdigit() else 0
        if w <= 0 and h <= 0:
            raise ValueError()
    except Exception:
        raise web.HTTPBadRequest(text="Bad size")

    # prefer ./static/uploads, fallback to ./uploads
    src_path = os.path.join(STATIC_DIR, "uploads", name)
    if not os.path.isfile(src_path):
        src_path = os.path.join(BASE_DIR, "uploads", name)
        if not os.path.isfile(src_path):
            raise web.HTTPNotFound()

    target_dir = os.path.join(THUMBS_DIR, f"{w}x{h}")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, name)

    if os.path.isfile(target_path):
        return web.FileResponse(target_path)

    try:
        with Image.open(src_path) as im:
            # apply EXIF-based transpose so orientation matches camera/viewer expectation
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass

            # convert transparency to white background for JPEG
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255,255,255))
                bg.paste(im, mask=im.split()[-1])
                im = bg

            orig_w, orig_h = im.size
            if w == 0:
                ratio = h / orig_h
                w = max(1, int(orig_w * ratio))
            elif h == 0:
                ratio = w / orig_w
                h = max(1, int(orig_h * ratio))

            im.thumbnail((w, h), Image.LANCZOS)

            ext = pathlib.Path(name).suffix.lower()
            tmp = target_path + ".tmp"
            if ext in ('.jpg', '.jpeg'):
                im.save(tmp, format="JPEG", quality=78, optimize=True)
            elif ext == '.png':
                im.save(tmp, format="PNG", optimize=True)
            else:
                im.save(tmp)
            os.replace(tmp, target_path)
    except Exception:
        log.exception("Thumb creation failed for %s", src_path)
        raise web.HTTPInternalServerError()

    return web.FileResponse(target_path)

# Reload topics endpoint
async def reload_topics_handler(request):
    global quiz
    try:
        newmap = load_topics_from_dir()
        if not newmap:
            return web.json_response({"ok": False, "error": "no topics found after reload"}, status=400)
        if quiz:
            async with quiz.lock:
                quiz.topics_map = newmap
                quiz.available_topics = list(newmap.keys())
                if quiz.current_topic not in quiz.available_topics and quiz.available_topics:
                    quiz.current_topic = quiz.available_topics[0]
                quiz.questions = quiz._prepare_questions(quiz.current_topic)
                quiz.current_qidx = 0
        return web.json_response({"ok": True, "topics": list(newmap.keys())})
    except Exception as e:
        log.exception("reload topics failed")
        return web.json_response({"ok": False, "error": str(e)}, status=500)
# Health endpoint
async def ping(request):
    return web.Response(text="ok")

def create_app(topics_map: Dict[str, List[Tuple[str, str]]]):
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/' + DEFAULT_INDEX, index_handler)
    app.router.add_get(DEFAULT_WS_PATH, ws_handler)
    app.router.add_get('/{path:.*\.(js|css|png|jpg|jpeg|svg|ico)}', static_handler)
    app.router.add_get('/static/uploads/{name}', static_handler)
    app.router.add_get('/thumbs/{size}/{name}', thumb_handler)
    app.router.add_post('/api/reload-topics', reload_topics_handler)

    # <-- добавьте здесь регистрацию health endpoint
    app.router.add_get('/ping', ping)
    return app

def parse_args():
    p = argparse.ArgumentParser(description="Quiz server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--http-port", default=8000, type=int)
    return p.parse_args()

if __name__ == "__main__":
    _ensure_ratings_file()
    _ensure_topics_dir()
    loaded = load_topics_from_dir()
    if not loaded:
        log.error("No valid topic files found in %s. Please add JSON files (UTF-8) with questions; exiting.", TOPICS_DIR)
        sys.exit(1)
    log.info("Using %d topic(s) loaded from %s", len(loaded), TOPICS_DIR)

    args = parse_args()
    try:
        quiz = QuizServer(loaded)
        app = create_app(loaded)
        log.info("Starting HTTP+WS server on %s:%d ws=%s", args.host, args.http_port, DEFAULT_WS_PATH)
        web.run_app(app, host=args.host, port=args.http_port)
    except Exception:
        log.exception("Fatal error, exiting")
        sys.exit(2)
