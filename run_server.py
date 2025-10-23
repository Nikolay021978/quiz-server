#!/usr/bin/env python3
# run_server.py
# Quiz server (loads topics only from topics/quiz)
# Requires: aiohttp, pillow

from __future__ import annotations

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
import unicodedata
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple

from aiohttp import web, WSMsgType
from PIL import Image, ImageOps

# ---------------------------
# Configuration
# ---------------------------
TURN_TIME_QUIZ = 40 # время на ответ для формата quiz (в секундах)
ROUND_DELAY = 0.8 # задержка между турами (в секундах)
NUM_QUESTIONS = 5 # число вопросов в раунде

THUMB_SIZE = "200x200" # размер миниатюр

BASE_DIR = os.path.dirname(__file__) or "."
STATIC_DIR = os.path.join(BASE_DIR, "static")
TOPICS_DIR = os.path.join(BASE_DIR, "topics")
QUIZ_SUBDIR = os.path.join(TOPICS_DIR, "quiz")
RATINGS_FILE = os.path.join(BASE_DIR, "ratings.json")

QUIZ_UPLOADS_DIR = os.path.join(STATIC_DIR, "quiz", "uploads")
QUIZ_THUMBS_DIR = os.path.join(STATIC_DIR, "quiz", "thumbs")

DEFAULT_WS_PATH = "/ws"
DEFAULT_INDEX = "client.html"

BASE_POINTS = 1000
MAX_SPEED_BONUS = 1000
MAX_STREAK_MULT = 3

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 10000 * 10000
MAX_THUMB_DIM = 5000

# ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TOPICS_DIR, exist_ok=True)
os.makedirs(QUIZ_SUBDIR, exist_ok=True)
os.makedirs(QUIZ_UPLOADS_DIR, exist_ok=True)
os.makedirs(QUIZ_THUMBS_DIR, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quizserver")

# ---------------------------
# Helpers: topics and images
# ---------------------------
def _ensure_topics_dir() -> None:
    try:
        os.makedirs(TOPICS_DIR, exist_ok=True)
        os.makedirs(QUIZ_SUBDIR, exist_ok=True)
    except Exception:
        log.exception("Failed to ensure topics dirs")

def _normalize_image_path(img: str) -> str:
    img = str(img).strip()
    if not img:
        return ""
    if img.startswith("/"):
        return img
    if img.startswith("http://") or img.startswith("https://"):
        return img
    return f"/static/quiz/uploads/{os.path.basename(img)}"

def _is_image_path(s: Optional[str]) -> bool:
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    lower = s.lower()
    if lower.startswith("/static/") or lower.startswith("http://") or lower.startswith("https://"):
        return True
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"):
        if lower.endswith(ext):
            return True
    return False

def _load_single_topic_file(path: str) -> Optional[Tuple[str, List[dict]]]:
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

    parsed: List[dict] = []
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')

    for it in items_raw:
        if isinstance(it, dict):
            d: dict = {}
            d["term"] = str(it.get("term") or it.get("name") or "")
            if it.get("image") and isinstance(it.get("image"), str):
                d["image"] = _normalize_image_path(it.get("image"))
            d["definition"] = str(it.get("definition") or it.get("def") or "")
            if "choices" in it and isinstance(it["choices"], list):
                choices = []
                for x in it["choices"]:
                    if isinstance(x, str):
                        s = x.strip()
                        if not s:
                            choices.append(s)
                            continue
                        lower = s.lower()
                        if lower.startswith("http://") or lower.startswith("https://") or s.startswith("/"):
                            choices.append(s)
                        elif any(lower.endswith(ext) for ext in IMAGE_EXTS):
                            choices.append(_normalize_image_path(s))
                        else:
                            choices.append(s)
                    else:
                        choices.append(str(x))
                d["choices"] = choices
            if "correct" in it:
                try:
                    d["correct"] = int(it["correct"])
                except Exception:
                    d["correct"] = None
            parsed.append(d)
        elif isinstance(it, list) and len(it) >= 2:
            parsed.append({"term": str(it[0]), "definition": str(it[1])})
        else:
            continue

    if not parsed:
        return None
    return (str(name), parsed)

def load_topics_from_dir() -> Dict[str, List[dict]]:
    """
    Load topics from topics/quiz/*.json.
    Return map: { topic_name: [items...] }
    """
    _ensure_topics_dir()
    topics_map: Dict[str, List[dict]] = {}
    pattern = os.path.join(QUIZ_SUBDIR, "*.json")
    files = sorted(glob.glob(pattern), key=os.path.basename)
    if not files:
        log.warning("No JSON files found in %s", QUIZ_SUBDIR)
    for path in files:
        res = _load_single_topic_file(path)
        if res:
            tname, items = res
            if tname in topics_map:
                log.warning("Duplicate topic name %s in topics/quiz — skipping %s", tname, path)
                continue
            topics_map[tname] = items
            img_count = sum(1 for it in items if isinstance(it.get("image"), str) and it.get("image"))
            log.info("Loaded topic '%s' (%d items, %d images) from %s", tname, len(items), img_count, path)
    return topics_map

# ---------------------------
# Ratings persistence
# ---------------------------
ratings_lock = asyncio.Lock()

def _ensure_ratings_file():
    if not os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except Exception:
            log.exception("Failed to create ratings file")
            raise

async def load_ratings() -> Dict[str, Dict[str, Dict[str, int]]]:
    async with ratings_lock:
        _ensure_ratings_file()
        try:
            try:
                size = os.path.getsize(RATINGS_FILE)
            except Exception:
                size = 0
            if size == 0:
                return {}
            if size > 5 * 1024 * 1024:
                log.warning("ratings.json too large, ignoring content")
                return {}
            with open(RATINGS_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    log.exception("ratings.json contains invalid JSON; resetting file")
                    try:
                        tmp = RATINGS_FILE + f".corrupt-{os.getpid()}-{uuid.uuid4().hex}"
                        with open(tmp, "w", encoding="utf-8") as tf:
                            json.dump({}, tf, ensure_ascii=False, indent=2)
                            tf.flush()
                            os.fsync(tf.fileno())
                        os.replace(tmp, RATINGS_FILE)
                    except Exception:
                        log.exception("failed to repair ratings.json")
                    return {}
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
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, RATINGS_FILE)
        except Exception:
            log.exception("Failed to save ratings.json")

async def update_ratings_after_game(topic: str, per_player_final: List[Dict]):
    if not isinstance(topic, str):
        return
    if not per_player_final:
        return
    ratings = await load_ratings()
    map_key = f"quiz|{topic}"
    topic_map = ratings.get(map_key, {})
    if not isinstance(topic_map, dict):
        topic_map = {}
    for r in per_player_final:
        name = r.get("name") or r.get("player") or None
        if not name or not isinstance(name, str):
            continue
        try:
            pts = int(r.get("points", r.get("total_points", 0) or 0))
        except Exception:
            pts = 0
        try:
            correct = int(r.get("correct", r.get("total_correct", 0) or 0))
        except Exception:
            correct = 0
        entry = topic_map.get(name, {"total_points": 0, "games": 0, "total_correct": 0})
        entry["total_points"] = entry.get("total_points", 0) + pts
        entry["games"] = entry.get("games", 0) + 1
        entry["total_correct"] = entry.get("total_correct", 0) + correct
        topic_map[name] = entry
    ratings[map_key] = topic_map
    await save_ratings(ratings)

async def get_ratings_for_topic(topic: str, top_n: Optional[int] = None) -> List[Dict]:
    ratings = await load_ratings()
    map_key = f"quiz|{topic}"
    topic_map = ratings.get(map_key, {}) or {}
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
# Safe websocket send helper
# ---------------------------
async def _safe_send_ws(ws, data: str, send_timeout: float = 1.0, close_timeout: float = 0.5) -> bool:
    try:
        await asyncio.wait_for(ws.send_str(data), timeout=send_timeout)
        return True
    except (asyncio.TimeoutError, RuntimeError, ConnectionError):
        log.debug("ws send timeout/conn error")
    except Exception:
        log.debug("ws send exception", exc_info=True)

    try:
        await asyncio.wait_for(ws.close(), timeout=close_timeout)
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
    return False

# ---------------------------
# Game state & logic
# ---------------------------
class Player:
    def __init__(self, name: str, ws):
        self.name = name
        self.ws = ws
        self.score = 0
        self.points = 0
        self.streak = 0
        self.wrong = 0
        self.join_ts = time.time()
        self.last_answer_ts = 0.0

    def __repr__(self):
        return f"<Player {self.name} pts={self.points} score={self.score} wrong={self.wrong}>"

class QuizServer:
    def __init__(self, topics_map: Dict[str, List[dict]]):
        self.players: "OrderedDict[str, Player]" = OrderedDict()
        self.lock = asyncio.Lock()
        self.topics = topics_map or {}
        self.available_topics = list(self.topics.keys())
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
        self._q_topics: Dict[int, str] = {}

    def _prepare_questions(self, topics: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        pool: List[Tuple[str, str, str]] = []
        selected = topics or ([self.current_topic] if self.current_topic else list(self.topics.keys()))
        for t in selected:
            items = self.topics.get(t, [])
            if not items:
                log.debug("_prepare_questions: no items for topic=%s", t)
            for item in items:
                term = item.get("term", "") or item.get("name", "")
                definition = item.get("definition", "") or item.get("def") or ""
                pool.append((term, definition, t))
        if not pool:
            log.warning("_prepare_questions: empty pool for topics=%s", selected)
            self._q_topics = {}
            return []
        k = min(len(pool), NUM_QUESTIONS)
        sampled = random.sample(pool, k=k)
        self._q_topics = {i: sampled[i][2] for i in range(len(sampled))}
        log.info("_prepare_questions: selected %d questions for topics=%s", len(sampled), selected)
        return [(s[0], s[1]) for s in sampled]

    async def send(self, player: Player, msg: dict):
        try:
            js = json.dumps(msg, ensure_ascii=False)
        except Exception:
            log.exception("Failed to encode JSON for sending to %s", player.name)
            return
        try:
            ok = await _safe_send_ws(player.ws, js)
            if not ok:
                log.debug("send: marking %s as dead", player.name)
                async with self.lock:
                    if player.name in self.players:
                        try:
                            await asyncio.wait_for(player.ws.close(), 1.0)
                        except Exception:
                            pass
                        try:
                            del self.players[player.name]
                        except KeyError:
                            pass
                try:
                    await self.broadcast_lobby()
                except Exception:
                    log.exception("broadcast_lobby failed after send removal")
        except Exception:
            log.debug("send unexpected failure to %s", player.name, exc_info=True)

    async def broadcast(self, msg: dict):
        try:
            js = json.dumps(msg, ensure_ascii=False)
        except Exception:
            log.exception("broadcast: failed to json-encode message")
            return

        async with self.lock:
            players_items = list(self.players.items())
        if not players_items:
            return

        parallel_limit = min(20, max(3, len(players_items)))
        sem = asyncio.Semaphore(parallel_limit)

        async def _send_and_report(name, pl):
            async with sem:
                try:
                    ok = await _safe_send_ws(pl.ws, js)
                    return None if ok else name
                except Exception:
                    log.debug("broadcast: send exception for %s", name, exc_info=True)
                    return name

        coros = [_send_and_report(name, pl) for name, pl in players_items]
        try:
            results = await asyncio.gather(*coros, return_exceptions=False)
        except Exception:
            results = []
            for name, pl in players_items:
                try:
                    ok = await _safe_send_ws(pl.ws, js)
                    results.append(None if ok else name)
                except Exception:
                    results.append(name)

        dead = [r for r in results if r]
        if dead:
            async with self.lock:
                for d in dead:
                    pl = self.players.pop(d, None)
                    if pl:
                        try:
                            await asyncio.wait_for(pl.ws.close(), timeout=0.5)
                        except Exception:
                            try:
                                await pl.ws.close()
                            except Exception:
                                pass
                if self.admin not in self.players:
                    self.admin = next(iter(self.players.keys()), None)
            try:
                await self.broadcast_lobby()
            except Exception:
                log.exception("broadcast_lobby failed after removing dead players")

    async def broadcast_lobby(self):
        async with self.lock:
            players = list(self.players.keys())
            admin = self.admin
            topics = list(self.topics.keys()) or []
            current_topic = self.current_topic or (topics[0] if topics else "")
            players_map = dict(self.players)

        for pname, pl in players_map.items():
            try:
                payload = {
                    "type": "lobby",
                    "players": players,
                    "admin": admin,
                    "is_admin": (pname == admin),
                    "format": "quiz",
                    "topics": topics,
                    "current_topic": current_topic,
                    "num_questions": NUM_QUESTIONS
                }
                await self.send(pl, payload)
            except Exception:
                log.exception("broadcast_lobby: failed to send lobby to %s", pname)

    async def add_player(self, name: str, ws) -> str:
        async with self.lock:
            i = 1
            if not isinstance(name, str):
                name = "player"
            safe = "".join(ch for ch in name if ch.isprintable() and ch not in "\r\n\t")
            safe = unicodedata.normalize("NFC", safe).strip()[:48] or "player"
            base = safe
            name = safe
            while name in self.players:
                name = f"{base}_{i}"
                i += 1
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
            "format": "quiz",
            "topics": list(self.topics.keys()) or [],
            "current_topic": self.current_topic or "",
            "num_questions": NUM_QUESTIONS,
            "players": list(self.players.keys())
        })
        await self.broadcast_lobby()
        return name

    async def remove_player(self, name: str):
        async with self.lock:
            if name in self.players:
                try:
                    await asyncio.wait_for(self.players[name].ws.close(), 1.0)
                except Exception:
                    pass
                del self.players[name]
                log.info("Player left: %s", name)
                if name == self.admin:
                    self.admin = next(iter(self.players.keys()), None)
                    log.info("New admin: %s", self.admin)
        await self.broadcast_lobby()

    async def handle_message(self, name: str, msg: dict):
        if not isinstance(msg, dict):
            return
        t = msg.get("type")

        if t == "choose_topic":
            topic = msg.get("topic")
            async with self.lock:
                if name != self.admin:
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":"Только админ может выбрать тему"})
                    log.info("choose_topic denied: %s is not admin (admin=%s)", name, self.admin)
                    return
                if self.game_running:
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":"Нельзя менять тему во время игры"})
                    log.info("choose_topic ignored while game running by %s", name)
                    return
                if topic not in self.topics:
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":"Неизвестная тема"})
                    log.warning("choose_topic unknown topic %r by %s", topic, name)
                    return
                self.current_topic = topic
                self.questions = self._prepare_questions([self.current_topic])
                self.current_qidx = 0
            await self.broadcast_lobby()
            return

        if t == "start_game":
            async with self.lock:
                if name != self.admin:
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":"Только админ может стартовать игру"})
                    log.info("start_game denied: %s is not admin (admin=%s)", name, self.admin)
                    return

                if self.game_running:
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":"Игра уже запущена"})
                    log.info("start_game ignored: game already running (requester=%s)", name)
                    return

                topics = msg.get("topics")
                if not isinstance(topics, list) or not topics or not all(isinstance(x, str) for x in topics):
                    log.warning("start_game aborted: topics missing or invalid from %s: %r", name, topics)
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":"Укажите тему(ы) для старта игры"})
                    return

                invalid = [t for t in topics if t not in self.topics]
                if invalid:
                    log.warning("start_game aborted: unknown topics %s requested by %s", invalid, name)
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":f"Неизвестные темы: {', '.join(invalid)}"})
                    return

                self.current_topic = topics[0]
                self.questions = self._prepare_questions(topics)
                self.current_qidx = 0

                if not self.questions:
                    log.warning("start_game aborted: no questions for topics=%s", topics)
                    pl = self.players.get(name)
                    if pl:
                        await self.send(pl, {"type":"error", "message":"Нет вопросов для выбранной темы. Проверьте файлы в topics/quiz."})
                    return

                # reset per-player stats at game start
                for p in self.players.values():
                    p.score = 0
                    p.points = 0
                    p.streak = 0
                    p.wrong = 0

                self.game_running = True
                self._last_game_context = {"topics": topics}
                self._game_task = asyncio.create_task(self._run_quiz())
                log.info("start_game accepted by %s for topics=%s", name, topics)
            return

        if t == "answer":
            async with self.lock:
                if not self.game_running:
                    return
                if name in self.answers:
                    return
                if not self.current_choices:
                    return
                choice = msg.get("choice")
                try:
                    ci = int(choice)
                except Exception:
                    return
                if ci < 0 or ci >= len(self.current_choices):
                    return
                turn_time = TURN_TIME_QUIZ
                if not self.current_question_start_ts or time.time() - self.current_question_start_ts > (turn_time + 5):
                    return
                pl = self.players.get(name)
                if not pl:
                    return
                if time.time() - pl.last_answer_ts < 0.15:
                    return
                pl.last_answer_ts = time.time()
                self.answers[name] = (ci, time.time())
            return

        if t == "get_ratings":
            req_topic = msg.get("topic")
            chosen_topic = None

            if isinstance(req_topic, str) and req_topic in self.topics:
                chosen_topic = req_topic

            if chosen_topic is None:
                if getattr(self, "current_topic", None) and self.current_topic in self.topics:
                    chosen_topic = self.current_topic

            ratings = []
            if chosen_topic:
                try:
                    ratings = await get_ratings_for_topic(chosen_topic, top_n=200)
                except Exception:
                    log.exception("Failed to load ratings for quiz|%s", chosen_topic)
                    ratings = []

            if name in self.players:
                await self.send(self.players[name], {
                    "type": "ratings",
                    "format": "quiz",
                    "topic": chosen_topic or "",
                    "ratings": ratings
                })
            return

    # ---------------------------
    # Quiz implementation
    # ---------------------------
    async def _run_quiz(self):
        try:
            async with self.lock:
                self.current_qidx = 0
                for p in self.players.values():
                    p.score = 0
                    p.points = 0
                    p.streak = 0
                    p.wrong = 0
            await self.broadcast({"type": "game_started", "format": "quiz", "topic": self.current_topic, "num_questions": NUM_QUESTIONS})

            total = len(self.questions)
            turn_time = TURN_TIME_QUIZ
            while True:
                async with self.lock:
                    if not self.players:
                        self.game_running = False
                        break
                    if self.current_qidx >= total:
                        break

                    term, definition = self.questions[self.current_qidx]
                    q_topic = self._q_topics.get(self.current_qidx, self.current_topic)

                    original_item = None
                    for raw in self.topics.get(q_topic, []):
                        raw_term = raw.get("term", "") or raw.get("name", "")
                        raw_def = raw.get("definition") or raw.get("def") or ""
                        if str(raw_term) == str(term) and (not raw_def or str(raw_def) == str(definition)):
                            original_item = raw
                            break

                    if original_item and isinstance(original_item.get("choices"), list) and len(original_item.get("choices")) > 0:
                        choices = [str(x) for x in original_item.get("choices")][:4]
                        correct_idx = None
                        if original_item.get("correct") is not None:
                            try:
                                correct_idx = int(original_item.get("correct"))
                            except Exception:
                                correct_idx = None
                        while len(choices) < 4:
                            choices.append("")
                        if len(choices) > 4:
                            choices = choices[:4]
                            if correct_idx is not None and correct_idx >= len(choices):
                                correct_idx = None
                    else:
                        same_topic_items = self.topics.get(q_topic, [])
                        same_topic_terms = [it.get("term") or it.get("name") for it in same_topic_items if (it.get("term") or it.get("name")) and (it.get("term") or it.get("name")) != term]
                        random.shuffle(same_topic_terms)
                        wrong_choices = []
                        idx = 0
                        while len(wrong_choices) < 3:
                            if idx < len(same_topic_terms):
                                candidate = same_topic_terms[idx]; idx += 1
                                if candidate and candidate not in wrong_choices:
                                    wrong_choices.append(candidate)
                            else:
                                wrong_choices.append("")
                        choices = wrong_choices[:3] + [term]
                        random.shuffle(choices)
                        try:
                            correct_idx = choices.index(term)
                        except Exception:
                            correct_idx = 3

                    self.current_choices = choices
                    self.current_correct = correct_idx
                    self.answers = {}
                    qidx = self.current_qidx

                payload = {
                    "type": "new_question",
                    "format": "quiz",
                    "question_idx": qidx,
                    "choices": self.current_choices,
                    "turn_time": turn_time,
                    "server_ts": time.time(),
                    "question_topic": q_topic
                }

                image_path = None
                if _is_image_path(term):
                    image_path = term
                elif _is_image_path(definition):
                    image_path = definition

                if image_path:
                    payload["image"] = image_path
                    payload["image_thumb"] = f"/static/quiz/thumbs/{THUMB_SIZE}/{os.path.basename(image_path)}"
                    if _is_image_path(term):
                        payload["question_text"] = definition if definition else ""
                    else:
                        payload["question_text"] = term if term else ""
                    payload["term"] = ""
                else:
                    payload["term"] = term
                    if definition:
                        payload["definition"] = definition
                    payload["question_text"] = term if term else (definition if definition else "")

                try:
                    os.makedirs(os.path.join(QUIZ_THUMBS_DIR, THUMB_SIZE), exist_ok=True)
                except Exception:
                    log.debug("could not create thumb dir for size %s", THUMB_SIZE, exc_info=True)

                self.current_question_start_ts = time.time()
                log.debug("Broadcasting question %d/%d topic=%s term=%s", qidx + 1, total, q_topic, (term[:80] if term else ""))
                await self.broadcast(payload)

                for sec in range(turn_time, 0, -1):
                    await self.broadcast({"type": "tick", "remaining": sec, "turn_time": turn_time})
                    await asyncio.sleep(1)

                results = []
                question_start = self.current_question_start_ts or (time.time() - turn_time)
                async with self.lock:
                    players_snapshot = list(self.players.items())
                    answers_snapshot = dict(self.answers)
                    correct_index_snapshot = int(self.current_correct) if self.current_correct is not None else None

                for pname, p in players_snapshot:
                    ans = answers_snapshot.get(pname)
                    if ans:
                        choice_idx, ans_ts = int(ans[0]), float(ans[1])
                        ok = (choice_idx == correct_index_snapshot)
                    else:
                        choice_idx, ans_ts = -1, None
                        ok = False

                    points_earned = 0
                    if ok:
                        p.score += 1
                        p.streak += 1
                        multiplier = min(MAX_STREAK_MULT, p.streak)
                        elapsed = max(0.0, ans_ts - question_start)
                        remaining = max(0.0, turn_time - elapsed)
                        speed_bonus = int((remaining / turn_time) * MAX_SPEED_BONUS)
                        points_earned = int((BASE_POINTS + speed_bonus) * multiplier)
                        p.points += points_earned
                    else:
                        p.streak = 0
                        p.wrong += 1

                    results.append({
                        "name": pname,
                        "choice": choice_idx,
                        "correct": ok,
                        "correct_count": p.score,
                        "incorrect": p.wrong,
                        "points_earned": points_earned,
                        "points": p.points,
                        "streak": p.streak
                    })

                async with self.lock:
                    leader_by_points = sorted(
                        [{"name": pname, "points": p.points, "correct": p.score, "incorrect": p.wrong} for pname, p in self.players.items()],
                        key=lambda x: -x["points"]
                    )

                await self.broadcast({
                    "type": "turn_result",
                    "format": "quiz",
                    "question_idx": qidx,
                    "correct_index": correct_index_snapshot,
                    "results": results,
                    "leaderboard": leader_by_points
                })

                async with self.lock:
                    self.current_qidx += 1
                    self.current_choices = None
                    self.current_correct = None
                    self.answers = {}
                    self.current_question_start_ts = None

                await asyncio.sleep(ROUND_DELAY)
        except asyncio.CancelledError:
            log.info("_run_quiz cancelled")
            raise
        except Exception:
            log.exception("Unexpected error in _run_quiz")
        finally:
            async with self.lock:
                final_lb = sorted(
                    [{"name": n, "correct": p.score, "incorrect": p.wrong, "points": p.points} for n, p in self.players.items()],
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
                ctx = getattr(self, "_last_game_context", None)
                if ctx and isinstance(ctx, dict):
                    topics = ctx.get("topics", [self.current_topic])
                    for t in topics:
                        await update_ratings_after_game(t, final_lb)
                else:
                    await update_ratings_after_game(self.current_topic, final_lb)
            except Exception:
                log.exception("Failed to update ratings after game")

            await self.broadcast({"type": "game_over", "format": "quiz", "leaderboard": final_lb})
            await self.broadcast_lobby()

# ---------------------------
# HTTP + WebSocket handlers
# ---------------------------
quiz_server: Optional[QuizServer] = None

def get_index_path() -> Optional[str]:
    candidates = [
        os.path.join(STATIC_DIR, DEFAULT_INDEX),
        os.path.join(BASE_DIR, DEFAULT_INDEX),
        os.path.join(STATIC_DIR, "game_client.html"),
        os.path.join(BASE_DIR, "game_client.html"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            log.info("Serving %s from %s", os.path.basename(p), os.path.abspath(p))
            return p
    return None

async def index_handler(request):
    idx_path = get_index_path()
    if idx_path:
        resp = web.FileResponse(idx_path)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    else:
        return web.Response(text="<h1>Client not found</h1>", content_type="text/html")

async def static_handler(request):
    rel = request.match_info.get('path', '') or request.match_info.get('name', '')
    if '..' in rel or rel.startswith('/'):
        raise web.HTTPNotFound()
    candidates_dirs = [STATIC_DIR, BASE_DIR, QUIZ_UPLOADS_DIR, os.path.join(BASE_DIR, "uploads")]
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
        if w > MAX_THUMB_DIM or h > MAX_THUMB_DIM:
            raise web.HTTPBadRequest(text="Requested thumbnail too large")
    except Exception:
        raise web.HTTPBadRequest(text="Bad size")

    src_path = os.path.join(QUIZ_UPLOADS_DIR, name)
    if not os.path.isfile(src_path):
        src_path = os.path.join(STATIC_DIR, "uploads", name)
        if not os.path.isfile(src_path):
            src_path = os.path.join(BASE_DIR, "uploads", name)
            if not os.path.isfile(src_path):
                raise web.HTTPNotFound()

    try:
        size_bytes = os.path.getsize(src_path)
    except Exception:
        raise web.HTTPNotFound()
    if size_bytes > MAX_UPLOAD_BYTES:
        raise web.HTTPRequestEntityTooLarge()

    target_dir = os.path.join(QUIZ_THUMBS_DIR, f"{size}")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, name)

    log.info("thumb request: size=%s name=%s src_path=%s", size, name, src_path)

    if os.path.isfile(target_path):
        return web.FileResponse(target_path)

    try:
        with Image.open(src_path) as im:
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass

            try:
                im.seek(0)
            except Exception:
                pass

            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255,255,255))
                try:
                    bg.paste(im, mask=im.split()[-1])
                except Exception:
                    bg.paste(im)
                im = bg
            else:
                im = im.convert("RGB")

            orig_w, orig_h = im.size
            if orig_w <= 0 or orig_h <= 0 or orig_w * orig_h > MAX_PIXELS:
                raise web.HTTPBadRequest(text="Image too large")

            if w == 0:
                ratio = h / orig_h
                w = max(1, min(MAX_THUMB_DIM, int(orig_w * ratio)))
            elif h == 0:
                ratio = w / orig_w
                h = max(1, min(MAX_THUMB_DIM, int(orig_h * ratio)))

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
    except FileNotFoundError:
        raise web.HTTPNotFound()
    except web.HTTPException:
        raise
    except Exception:
        log.exception("Thumb creation failed for %s -> %s (size=%s)", src_path, target_path, size)
        raise web.HTTPInternalServerError()

    return web.FileResponse(target_path)

async def thumb_plain_quiz_handler(request):
    name = request.match_info.get('name', '')
    if '..' in name or name.startswith('/'):
        raise web.HTTPNotFound()
    return web.HTTPFound(f"/static/quiz/thumbs/{THUMB_SIZE}/{name}")

async def reload_topics_handler(request):
    global quiz_server
    try:
        new_topics = load_topics_from_dir()
        if not new_topics:
            return web.json_response({"ok": False, "error": "no topics found after reload"}, status=400)

        if quiz_server:
            async with quiz_server.lock:
                quiz_server.topics = new_topics
                quiz_server.available_topics = list(new_topics.keys())
                if not quiz_server.current_topic and quiz_server.available_topics:
                    quiz_server.current_topic = quiz_server.available_topics[0]
                quiz_server.questions = quiz_server._prepare_questions([quiz_server.current_topic]) if quiz_server.current_topic else []
                quiz_server.current_qidx = 0

        try:
            if quiz_server:
                await quiz_server.broadcast_lobby()
        except Exception:
            log.exception("broadcast_lobby failed after reload")

        compact = { "quiz": list(new_topics.keys()) }
        return web.json_response({"ok": True, "topics": compact})
    except Exception as e:
        log.exception("reload topics failed")
        return web.json_response({"ok": False, "error": str(e)}, status=500)

async def get_ratings_api_handler(request):
    topic = request.query.get("topic", "")
    if not topic:
        return web.json_response({"ok": False, "error": "topic required"}, status=400)
    try:
        lst = await get_ratings_for_topic(topic, top_n=200)
        return web.json_response({"ok": True, "format": "quiz", "topic": topic, "ratings": lst})
    except Exception:
        log.exception("get_ratings_api_handler failed")
        return web.json_response({"ok": False, "error": "internal"}, status=500)

async def ping(request):
    return web.Response(text="ok")

async def ws_handler(request):
    global quiz_server
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    name: Optional[str] = None
    try:
        join_msg = await ws.receive()
        if join_msg.type != WSMsgType.TEXT:
            await ws.close()
            return ws
        if not isinstance(join_msg.data, str) or len(join_msg.data) > 64 * 1024:
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
        raw_name = str(j.get("name") or "").strip()[:48]
        name = await quiz_server.add_player(raw_name, ws)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    if isinstance(msg.data, str) and len(msg.data) > 256 * 1024:
                        continue
                    data = json.loads(msg.data)
                except Exception:
                    continue
                await quiz_server.handle_message(name, data)
            elif msg.type == WSMsgType.ERROR:
                log.warning("ws connection closed with exception %s", ws.exception())
                break
    except Exception:
        log.exception("ws handler exception")
    finally:
        if name:
            try:
                await quiz_server.remove_player(name)
            except Exception:
                log.exception("remove_player failed in ws_handler")
        try:
            await ws.close()
        except Exception:
            pass
    return ws

def create_app(topics_map: Dict[str, List[dict]]):
    global quiz_server
    quiz_server = QuizServer(topics_map)
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/' + DEFAULT_INDEX, index_handler)
    app.router.add_get('/game_client.html', index_handler)
    app.router.add_get(DEFAULT_WS_PATH, ws_handler)
    app.router.add_get('/static/{path:.*}', static_handler)
    app.router.add_get('/uploads/{name}', static_handler)
    app.router.add_get('/thumbs/{size}/{name}', thumb_handler)
    app.router.add_get('/static/quiz/thumbs/{size}/{name}', thumb_handler)
    app.router.add_get('/static/quiz/thumbs/{name}', thumb_plain_quiz_handler)
    app.router.add_post('/api/reload-topics', reload_topics_handler)
    app.router.add_get('/api/get-ratings', get_ratings_api_handler)
    app.router.add_get('/ping', ping)

    async def on_shutdown(app):
        global quiz_server
        if quiz_server and getattr(quiz_server, "_game_task", None):
            try:
                quiz_server._game_task.cancel()
                await asyncio.wait_for(quiz_server._game_task, timeout=3.0)
            except Exception:
                pass

    app.on_shutdown.append(on_shutdown)
    return app

def parse_args():
    p = argparse.ArgumentParser(description="Quiz server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--http-port", default=8000, type=int)
    return p.parse_args()

if __name__ == "__main__":
    try:
        _ensure_ratings_file()
    except Exception:
        log.error("Cannot create or access ratings file %s; aborting", RATINGS_FILE)
        sys.exit(1)
    _ensure_topics_dir()
    loaded = load_topics_from_dir()
    if not loaded:
        log.error("No valid topic files found in %s. Please add JSON files (UTF-8) into topics/quiz with questions; exiting.", QUIZ_SUBDIR)
        sys.exit(1)
    log.info("Loaded topics: %s", ", ".join(sorted(loaded.keys())))
    args = parse_args()
    try:
        app = create_app(loaded)
        log.info("Starting HTTP+WS server on %s:%d ws=%s", args.host, args.http_port, DEFAULT_WS_PATH)
        web.run_app(app, host=args.host, port=args.http_port)
    except Exception:
        log.exception("Fatal error, exiting")
        sys.exit(2)
