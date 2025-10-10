#!/usr/bin/env python3
# run_server.py
#
# Quiz server with topics stored in ./topics/*.json
# - HTTP + WebSocket (aiohttp)
# - Client UI served from client.html in repo or embedded fallback
# - Supports image questions: payload.image (full) and payload.image_thumb (generated thumbnail)
# - Generates cached thumbnails with EXIF orientation fix (Pillow)
# - Endpoint POST /api/reload-topics to reload topics at runtime
# - Supports multi-topic games and a Study mode with multiple study methods
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
import uuid
from collections import OrderedDict, deque
from typing import Dict, Optional, List, Tuple, Deque

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
            tmp = RATINGS_FILE + f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ratings, f, ensure_ascii=False, indent=2)
            os.replace(tmp, RATINGS_FILE)
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
        # study-specific queue (deque of card dicts)
        self.study_queue: Deque[Dict] = deque()
        self.study_stats = {"seen": 0, "correct": 0}

class QuizServer:
    def __init__(self, topics_map: Dict[str, List[Tuple[str, str]]]):
        self.players: "OrderedDict[str, Player]" = OrderedDict()
        self.lock = asyncio.Lock()
        self.topics_map = topics_map or {}
        self.available_topics = list(self.topics_map.keys())
        self.current_topic: str = self.available_topics[0] if self.available_topics else ""
        self.questions: List[Tuple[str, str]] = self._prepare_questions([self.current_topic]) if self.current_topic else []
        self.current_qidx: int = 0
        self.current_choices: Optional[List[str]] = None
        self.current_correct: Optional[int] = None
        self.answers: Dict[str, Tuple[int, float]] = {}
        self.game_running: bool = False
        self.admin: Optional[str] = None
        self._game_task: Optional[asyncio.Task] = None
        self.current_question_start_ts: Optional[float] = None
        # mapping question index -> source topic (for per-topic stats)
        self._q_topics: Dict[int, str] = {}

        self._all_terms: List[str] = []
        for items in self.topics_map.values():
            for t, _ in items:
                if t not in self._all_terms:
                    self._all_terms.append(t)

    def _prepare_questions(self, topics: Optional[List[str]]) -> List[Tuple[str, str]]:
        pool: List[Tuple[str, str, str]] = []  # (term, definition, topic)
        selected = topics or [self.current_topic] if self.current_topic else list(self.available_topics)
        for t in selected:
            for term, definition in self.topics_map.get(t, []):
                pool.append((term, definition, t))
        if not pool:
            return []
        k = min(len(pool), NUM_QUESTIONS)
        sampled = random.sample(pool, k=k)
        # store topics mapping
        self._q_topics = {i: sampled[i][2] for i in range(len(sampled))}
        return [(s[0], s[1]) for s in sampled]

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
            # normalize and limit name length
            if not isinstance(name, str):
                name = "player"
            name = name.strip()[:48]
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
                self.questions = self._prepare_questions([self.current_topic])
                self.current_qidx = 0
            await self.broadcast_lobby()
            return

        if t == "start_game":
            async with self.lock:
                if name != self.admin:
                    return
                if self.game_running:
                    return
                fmt = msg.get("format", "game")
                topics = msg.get("topics") or [self.current_topic] if self.current_topic else list(self.available_topics)
                if fmt == "study":
                    method = msg.get("study_method", "flashcards")
                    self.game_running = True
                    self._game_task = asyncio.create_task(self._run_study(method, topics))
                else:
                    # normal game
                    self.questions = self._prepare_questions(topics)
                    self.current_qidx = 0
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

        if t == "study_feedback":
            # { type:"study_feedback", card_id:"...", result:"ok"|"wrong"|"skip", typed: "..." }
            async with self.lock:
                player = self.players.get(name)
                if not player:
                    return
                if not player.study_queue:
                    return
                card = player.study_queue[0]
                result = msg.get("result")
                player.study_stats["seen"] += 1
                if result == "ok":
                    player.study_stats["correct"] += 1
                # pop front
                player.study_queue.popleft()
                try:
                    await player.ws.send_str(json.dumps({
                        "type": "study_result",
                        "card_id": card.get("id"),
                        "status": "ok" if result == "ok" else "wrong"
                    }))
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

                    # --- Формирование вариантов только из соответствующей темы ---
                    q_topic = self._q_topics.get(self.current_qidx, self.current_topic)
                    same_topic_terms = [t for t, _ in self.topics_map.get(q_topic, []) if t != term]
                    random.shuffle(same_topic_terms)

                    wrong_choices = same_topic_terms[:3]

                    if len(wrong_choices) < 3 and same_topic_terms:
                        i = 0
                        while len(wrong_choices) < 3:
                            wrong_choices.append(same_topic_terms[i % len(same_topic_terms)])
                            i += 1

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
                    "server_ts": time.time(),
                    "question_topic": self._q_topics.get(qidx, self.current_topic)
                }
                if isinstance(definition, str) and definition.startswith("/static/"):
                    payload["image"] = definition
                    basename = os.path.basename(definition)
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

    async def _run_study(self, method: str, topics: List[str]):
        # Simple per-player flashcards-based study implementation
        async with self.lock:
            for name, p in self.players.items():
                p.study_queue.clear()
                p.study_stats = {"seen": 0, "correct": 0}
                cards = []
                for t in topics:
                    for term, definition in self.topics_map.get(t, []):
                        cards.append({"id": f"{t}|{term}|{uuid.uuid4().hex}", "term": term, "definition": definition, "topic": t})
                random.shuffle(cards)
                for c in cards:
                    p.study_queue.append(c)

        await self.broadcast({"type": "study_started", "method": method, "topics": topics})
        try:
            while True:
                async with self.lock:
                    if not self.players:
                        self.game_running = False
                        break
                    if all(len(p.study_queue) == 0 for p in self.players.values()):
                        break
                    for name, p in list(self.players.items()):
                        if not p.study_queue:
                            continue
                        card = p.study_queue[0]
                        try:
                            await p.ws.send_str(json.dumps({
                                "type": "study_card",
                                "card_id": card["id"],
                                "term": card["term"],
                                "definition": card["definition"],
                                "mode": method
                            }))
                        except Exception:
                            log.debug("Failed to send study_card to %s", name)
                await asyncio.sleep(0.2)
        finally:
            async with self.lock:
                summaries = []
                for name, p in self.players.items():
                    summaries.append({"name": name, "seen": p.study_stats.get("seen", 0), "correct": p.study_stats.get("correct", 0)})
                self.game_running = False
                self._game_task = None
            await self.broadcast({"type": "study_summary", "stats": summaries})
            await self.broadcast_lobby()

# ---------------------------
# Client HTML: embedded fallback (minimal)
# ---------------------------
def create_embedded_client_html() -> str:
    # Minimal fallback: instructs to place client.html in repo; includes tiny basic UI.
    return """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quiz client (fallback)</title></head>
<body>
<h2>Client not found on disk</h2>
<p>Place client.html next to run_server.py or in ./static/client.html to use the full UI.</p>
</body></html>
"""

# ---------------------------
# HTTP + WebSocket handlers (complete web part)
# ---------------------------
quiz: Optional[QuizServer] = None

def get_index_path() -> Optional[str]:
    # prefer STATIC_DIR/client.html then BASE_DIR/client.html
    candidates = [os.path.join(STATIC_DIR, DEFAULT_INDEX), os.path.join(BASE_DIR, DEFAULT_INDEX)]
    for p in candidates:
        if os.path.isfile(p):
            log.info("Serving client.html from %s", os.path.abspath(p))
            return p
    return None

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
    # use pathlib to avoid traversal
    candidates_dirs = [STATIC_DIR, BASE_DIR, os.path.join(STATIC_DIR, "uploads"), os.path.join(BASE_DIR, "uploads")]
    for base in candidates_dirs:
        try:
            base_p = pathlib.Path(base).resolve()
            p = (base_p / rel).resolve()
            if str(p).startswith(str(base_p)) and p.is_file():
                resp = web.FileResponse(str(p))
                resp.headers['Cache-Control'] = 'no-store'
                return resp
        except Exception:
            continue
    raise web.HTTPNotFound()

# Thumbnail handler with EXIF orientation fix
async def thumb_handler(request):
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
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass

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
            tmp = target_path + f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
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
                quiz.questions = quiz._prepare_questions([quiz.current_topic])
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
    app.router.add_get('/{path:.*\\.(js|css|png|jpg|jpeg|svg|ico)}', static_handler)
    app.router.add_get('/static/uploads/{name}', static_handler)
    app.router.add_get('/thumbs/{size}/{name}', thumb_handler)
    app.router.add_post('/api/reload-topics', reload_topics_handler)
    app.router.add_get('/ping', ping)
    return app

def parse_args():
    p = argparse.ArgumentParser(description="Quiz server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--http-port", default=8000, type=int)
    return p.parse_args()

# WebSocket handler defined after QuizServer to avoid forward reference issues
async def ws_handler(request):
    global quiz
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    name: Optional[str] = None
    try:
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
        # simple sanitation
        raw_name = str(j.get("name")).strip()[:48]
        name = await quiz.add_player(raw_name, ws)
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
