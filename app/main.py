"""舞台提示执行台 —— FastAPI + SQLite 后端。

并发模型：所有写操作（确认 / 暂停 / 恢复 / 编辑 / 重排 / 归档 / 设置 /
重置 / 以档案开新场）都在同一把锁内串行完成；其中带版本号的操作
（确认 / 暂停恢复 / 编辑 / 重排 / 归档 / 以档案开新场）在锁内完成
"检查版本号 -> 修改数据"两个步骤，因此同一版本号上并发的两个操作
只会有一个生效，另一个收到 409 及最新状态。归档的"写档案 + 清空提示"、
以档案开新场的"写入新场提示 + 推进状态"、暂停恢复的"暂停标记 + 版本 +1"
都在同一事务内提交，任一步失败整体回滚。
"""
import json
import os
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "data", "cues.db"))

MIN_CUES = 2
MAX_CUES = 12

# 串行化所有写操作，保证版本检查与落库是原子的。
lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 0,
                status  TEXT NOT NULL DEFAULT 'setup',  -- setup | live | done
                paused  INTEGER NOT NULL DEFAULT 0      -- 演出中是否暂停发出
            );
            CREATE TABLE IF NOT EXISTS cues (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                number         TEXT NOT NULL,           -- 唯一编号
                content        TEXT NOT NULL,           -- 当前（可改写的）内容
                position       INTEGER NOT NULL,        -- 待发顺序
                status         TEXT NOT NULL DEFAULT 'pending',  -- pending | issued
                issued_content TEXT,                    -- 发出那一刻的内容快照
                issued_at      TEXT,
                issued_seq     INTEGER                  -- 发出顺序
            );
            CREATE TABLE IF NOT EXISTS archives (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,              -- 场次名称
                archived_at TEXT NOT NULL,              -- 归档时间（UTC ISO）
                status      TEXT NOT NULL,              -- 归档当时状态：live | done
                version     INTEGER NOT NULL,           -- 归档时携带的页面版本
                issued      TEXT NOT NULL,              -- 已发：编号/快照/时间（按发出顺序）
                pending     TEXT NOT NULL               -- 剩余：编号/内容（按待发顺序）
            );
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta (id, version, status) VALUES (1, 0, 'setup')"
        )
        # 旧库升级：meta 表缺少 paused 列时补上，已有数据默认未暂停。
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(meta)")}
        if "paused" not in columns:
            conn.execute(
                "ALTER TABLE meta ADD COLUMN paused INTEGER NOT NULL DEFAULT 0"
            )


# ---------------------------------------------------------------- 状态组装

def build_state(conn: sqlite3.Connection) -> dict:
    meta = conn.execute(
        "SELECT version, status, paused FROM meta WHERE id = 1"
    ).fetchone()
    rows = conn.execute(
        "SELECT * FROM cues ORDER BY position ASC, id ASC"
    ).fetchall()
    pending = [r for r in rows if r["status"] == "pending"]
    issued = sorted(
        (r for r in rows if r["status"] == "issued"),
        key=lambda r: r["issued_seq"],
    )

    def pending_json(r):
        return {"id": r["id"], "number": r["number"], "content": r["content"]}

    def issued_json(r):
        return {
            "id": r["id"],
            "number": r["number"],
            "content": r["issued_content"],
            "issued_at": r["issued_at"],
        }

    return {
        "version": meta["version"],
        "status": meta["status"],
        "paused": bool(meta["paused"]),
        "current": pending_json(pending[0]) if pending else None,
        "next": pending_json(pending[1]) if len(pending) > 1 else None,
        "pending": [pending_json(r) for r in pending],
        "issued": [issued_json(r) for r in issued],
        "total": len(rows),
        "remaining": len(pending),
        "min_cues": MIN_CUES,
        "max_cues": MAX_CUES,
    }


def conflict(conn: sqlite3.Connection) -> JSONResponse:
    """409：请求已过期，附带最新状态供页面刷新。"""
    return JSONResponse(
        status_code=409,
        content={"error": "conflict", "message": "操作已过期", "state": build_state(conn)},
    )


def bad_request(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": message})


def cue_list_error(cues: list[tuple[str, str]]) -> Optional[str]:
    """校验提示清单：2～12 条、编号唯一非空、内容非空。

    入参为 (number, content) 对；通过返回 None，失败返回中文错误说明。
    """
    if not (MIN_CUES <= len(cues) <= MAX_CUES):
        return f"提示数量需在 {MIN_CUES}～{MAX_CUES} 条之间"
    numbers = [n.strip() for n, _ in cues]
    if any(not n for n in numbers):
        return "编号不能为空"
    if len(set(numbers)) != len(numbers):
        return "编号必须唯一"
    if any(not c.strip() for _, c in cues):
        return "内容不能为空"
    return None


# ---------------------------------------------------------------- 档案组装

def archive_snapshots(conn: sqlite3.Connection) -> tuple[list[dict], list[dict]]:
    """读取归档快照：已发按发出顺序、剩余按待发顺序。"""
    rows = conn.execute("SELECT * FROM cues ORDER BY position ASC, id ASC").fetchall()
    issued = [
        {
            "number": r["number"],
            "content": r["issued_content"],
            "issued_at": r["issued_at"],
        }
        for r in sorted(
            (r for r in rows if r["status"] == "issued"),
            key=lambda r: r["issued_seq"],
        )
    ]
    pending = [
        {"number": r["number"], "content": r["content"]}
        for r in rows
        if r["status"] == "pending"
    ]
    return issued, pending


def archive_summary(row: sqlite3.Row) -> dict:
    issued = json.loads(row["issued"])
    pending = json.loads(row["pending"])
    return {
        "id": row["id"],
        "name": row["name"],
        "archived_at": row["archived_at"],
        "status": row["status"],
        "issued_count": len(issued),
        "pending_count": len(pending),
    }


def archive_detail(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "archived_at": row["archived_at"],
        "status": row["status"],
        "version": row["version"],
        "issued": json.loads(row["issued"]),
        "pending": json.loads(row["pending"]),
    }


def archive_cues(row: sqlite3.Row) -> list[tuple[str, str]]:
    """档案转新场提示：先按已发顺序取编号与内容快照，再接按待发顺序的
    编号与内容；不继承发出时间或进度（全部作为新场待发提示）。"""
    issued = json.loads(row["issued"])
    pending = json.loads(row["pending"])
    return [(c["number"], c["content"]) for c in issued] + [
        (c["number"], c["content"]) for c in pending
    ]


# ---------------------------------------------------------------- 场次对照

MISSING_SIDE = {"status": "missing", "content": None, "offset_seconds": None}


def parse_time(iso_text: Optional[str]) -> Optional[datetime]:
    if not iso_text:
        return None
    return datetime.fromisoformat(iso_text.replace("Z", "+00:00"))


def archive_side_view(row: sqlite3.Row) -> dict[str, dict]:
    """档案 -> {编号: 一侧视图}：已发给内容快照与相对零点秒数，未发给内容。

    零点为该场首条已发提示的发出时间；相对秒数保留毫秒，暂停时长不扣除、
    直接计入时间差。未发条目、或已发但缺发出时间的条目秒数为 None。
    """
    view: dict[str, dict] = {}
    issued = json.loads(row["issued"])
    zero = parse_time(issued[0].get("issued_at")) if issued else None
    for cue in issued:
        at = parse_time(cue.get("issued_at"))
        offset = None
        if zero is not None and at is not None:
            offset = round((at - zero).total_seconds(), 3)
        view[cue["number"]] = {
            "status": "issued",
            "content": cue["content"],
            "offset_seconds": offset,
        }
    for cue in json.loads(row["pending"]):
        view[cue["number"]] = {
            "status": "pending",
            "content": cue["content"],
            "offset_seconds": None,
        }
    return view


def compare_rows(base_view: dict[str, dict], compare_view: dict[str, dict]) -> list[dict]:
    """按编号精确匹配取并集（编号字符串升序），逐条给出两侧状态与秒数差。

    仅两侧均已发出且内容完全相同时计算"对照减基准"的秒数差（正值表示
    对照场更晚），其余为 None 并附中文原因。
    """
    rows = []
    for number in sorted(set(base_view) | set(compare_view)):
        base = base_view.get(number)
        comp = compare_view.get(number)
        delta = None
        reason = None
        if (
            base is not None
            and comp is not None
            and base["status"] == "issued"
            and comp["status"] == "issued"
        ):
            if base["content"] != comp["content"]:
                reason = "两侧均已发出，但内容不一致"
            elif base["offset_seconds"] is None or comp["offset_seconds"] is None:
                reason = "缺少发出时间，无法计算秒数差"
            else:
                delta = round(comp["offset_seconds"] - base["offset_seconds"], 3)
        else:
            parts = []
            if base is None:
                parts.append("基准场无此编号")
            elif base["status"] != "issued":
                parts.append("基准场未发出")
            if comp is None:
                parts.append("对照场无此编号")
            elif comp["status"] != "issued":
                parts.append("对照场未发出")
            reason = "；".join(parts)
        rows.append(
            {
                "number": number,
                "base": base if base is not None else dict(MISSING_SIDE),
                "compare": comp if comp is not None else dict(MISSING_SIDE),
                "delta_seconds": delta,
                "delta_reason": reason,
            }
        )
    return rows


# ---------------------------------------------------------------- 应用

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="舞台提示执行台", lifespan=lifespan)


class CueIn(BaseModel):
    number: str
    content: str


class SetupIn(BaseModel):
    cues: list[CueIn]


class VersionedIn(BaseModel):
    version: int


class PauseIn(BaseModel):
    version: int
    paused: bool  # 目标状态：true 暂停发出，false 恢复发出


class ArchiveIn(BaseModel):
    version: int
    name: str


class EditIn(BaseModel):
    version: int
    cue_id: int
    content: str
    number: Optional[str] = None


class ReorderIn(BaseModel):
    version: int
    order: list[int]


class StartFromArchiveIn(BaseModel):
    version: int


@app.get("/api/state")
def get_state():
    with db() as conn:
        return build_state(conn)


@app.get("/api/archives")
def list_archives():
    """档案列表：按归档时间倒序、同时间按档案 ID 倒序。"""
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM archives ORDER BY archived_at DESC, id DESC"
        ).fetchall()
        return {"archives": [archive_summary(r) for r in rows]}


@app.get("/api/archives/{archive_id}")
def get_archive(archive_id: int):
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM archives WHERE id = ?", (archive_id,)
        ).fetchone()
        if row is None:
            return JSONResponse(status_code=404, content={"error": "archive_not_found"})
        return archive_detail(row)


@app.get("/api/compare")
def compare_archives(base: int, compare: int):
    """对照两场档案的提示发出节奏（只读，任意演出状态可用）。

    按编号精确匹配取两场提示的并集，按编号字符串升序返回；每条列出两侧
    内容快照、已发/未发/缺失状态与相对发出秒数。各场以首条已发提示的
    发出时间为零点，使用档案保存的发出时间，保留小数秒，暂停时长计入；
    未发、缺失或整场无已发记录时秒数为 null。仅两侧均已发出且内容完全
    相同时计算"对照减基准"的秒数差（正值表示对照场更晚），其余为 null
    并附中文原因。任一档案不存在返回 404。查询不修改档案、当前演出或
    版本号。
    """
    with db() as conn:
        sides = {}
        for key, archive_id in (("base", base), ("compare", compare)):
            row = conn.execute(
                "SELECT * FROM archives WHERE id = ?", (archive_id,)
            ).fetchone()
            if row is None:
                return JSONResponse(
                    status_code=404, content={"error": "archive_not_found"}
                )
            sides[key] = row

    views = {key: archive_side_view(row) for key, row in sides.items()}
    return {
        "base": archive_summary(sides["base"]),
        "compare": archive_summary(sides["compare"]),
        "rows": compare_rows(views["base"], views["compare"]),
    }


@app.post("/api/setup")
def setup(body: SetupIn):
    """开场前录入：2~12 条、编号唯一、内容非空。会清空旧进度。"""
    cues = [(c.number, c.content) for c in body.cues]
    error = cue_list_error(cues)
    if error:
        return bad_request(error)
    numbers = [n.strip() for n, _ in cues]
    contents = [c.strip() for _, c in cues]

    with lock, db() as conn:
        conn.execute("DELETE FROM cues")
        for pos, (number, content) in enumerate(zip(numbers, contents), start=1):
            conn.execute(
                "INSERT INTO cues (number, content, position, status) "
                "VALUES (?, ?, ?, 'pending')",
                (number, content, pos),
            )
        conn.execute(
            "UPDATE meta SET version = version + 1, status = 'live', paused = 0 "
            "WHERE id = 1"
        )
        return build_state(conn)


@app.post("/api/confirm")
def confirm(body: VersionedIn):
    """确认发出当前提示：以服务端此刻的内容为准记入已发，并推进一条。

    暂停期间即使用最新版本号也返回 409 及最新状态，须先恢复发出。
    """
    with lock, db() as conn:
        meta = conn.execute(
            "SELECT version, status, paused FROM meta WHERE id = 1"
        ).fetchone()
        if (
            meta["status"] != "live"
            or meta["paused"]
            or body.version != meta["version"]
        ):
            return conflict(conn)
        row = conn.execute(
            "SELECT * FROM cues WHERE status = 'pending' "
            "ORDER BY position ASC, id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return conflict(conn)

        new_version = meta["version"] + 1
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE cues SET status = 'issued', issued_content = content, "
            "issued_at = ?, issued_seq = ? WHERE id = ?",
            (now, new_version, row["id"]),
        )
        remaining = conn.execute(
            "SELECT COUNT(*) AS c FROM cues WHERE status = 'pending'"
        ).fetchone()["c"]
        conn.execute(
            "UPDATE meta SET version = ?, status = ? WHERE id = 1",
            (new_version, "done" if remaining == 0 else "live"),
        )
        return build_state(conn)


@app.post("/api/pause")
def pause(body: PauseIn):
    """暂停或恢复发出（仅演出中）。

    仅当状态为演出中（live）且目标 paused 与当前不同时生效；暂停标记与
    版本号 +1 在同一事务内提交，写入失败两者都不变。暂停不清空提示、不
    推进进度；暂停期间确认被拒绝（409），编辑 / 重排 / 归档仍可用。
    非演出中、目标已达成（如已暂停时再暂停）或版本过期返回 409 及最新
    状态；同版本的暂停、恢复、确认、编辑最多一个生效。
    """
    with lock, db() as conn:
        meta = conn.execute(
            "SELECT version, status, paused FROM meta WHERE id = 1"
        ).fetchone()
        if (
            meta["status"] != "live"
            or body.version != meta["version"]
            or bool(meta["paused"]) == body.paused
        ):
            return conflict(conn)
        conn.execute(
            "UPDATE meta SET paused = ?, version = version + 1 WHERE id = 1",
            (1 if body.paused else 0,),
        )
        return build_state(conn)


@app.post("/api/edit")
def edit(body: EditIn):
    """改写一条尚未发出的提示（内容，可选编号）。"""
    with lock, db() as conn:
        meta = conn.execute("SELECT version FROM meta WHERE id = 1").fetchone()
        if body.version != meta["version"]:
            return conflict(conn)
        row = conn.execute("SELECT * FROM cues WHERE id = ?", (body.cue_id,)).fetchone()
        if row is None or row["status"] != "pending":
            return conflict(conn)

        content = body.content.strip()
        if not content:
            return bad_request("内容不能为空")
        number = row["number"] if body.number is None else body.number.strip()
        if not number:
            return bad_request("编号不能为空")
        dup = conn.execute(
            "SELECT 1 FROM cues WHERE number = ? AND id != ?", (number, row["id"])
        ).fetchone()
        if dup:
            return bad_request("编号必须唯一")

        conn.execute(
            "UPDATE cues SET content = ?, number = ? WHERE id = ?",
            (content, number, row["id"]),
        )
        conn.execute("UPDATE meta SET version = version + 1 WHERE id = 1")
        return build_state(conn)


@app.post("/api/reorder")
def reorder(body: ReorderIn):
    """重排尚未发出的提示：order 必须恰好是全部待发提示的 id。"""
    with lock, db() as conn:
        meta = conn.execute("SELECT version FROM meta WHERE id = 1").fetchone()
        if body.version != meta["version"]:
            return conflict(conn)
        pending_ids = {
            r["id"]
            for r in conn.execute("SELECT id FROM cues WHERE status = 'pending'")
        }
        if len(body.order) != len(pending_ids) or set(body.order) != pending_ids:
            return conflict(conn)

        base = conn.execute(
            "SELECT COALESCE(MAX(position), 0) AS m FROM cues"
        ).fetchone()["m"]
        for offset, cue_id in enumerate(body.order, start=1):
            conn.execute(
                "UPDATE cues SET position = ? WHERE id = ?", (base + offset, cue_id)
            )
        conn.execute("UPDATE meta SET version = version + 1 WHERE id = 1")
        return build_state(conn)


@app.post("/api/reset")
def reset():
    """清空进度，回到开场前设置。"""
    with lock, db() as conn:
        conn.execute("DELETE FROM cues")
        conn.execute(
            "UPDATE meta SET version = version + 1, status = 'setup', paused = 0 "
            "WHERE id = 1"
        )
        return build_state(conn)


@app.post("/api/archive")
def archive(body: ArchiveIn):
    """归档本场并收场。

    仅演出中（live）或已完成（done）可归档；携带页面所见版本号。
    快照（已发按发出顺序、剩余按待发顺序）写入 archives 与清空 cues、
    回到 setup、版本号 +1 在同一事务内提交：写档案失败则整体回滚，
    当前提示不会被清空。同版本的重复归档因版本不再匹配而拿到 409，
    不会多建档案。
    """
    name = body.name.strip()
    if not name:
        return bad_request("场次名称不能为空")

    with lock, db() as conn:
        meta = conn.execute("SELECT version, status FROM meta WHERE id = 1").fetchone()
        if meta["status"] not in ("live", "done") or body.version != meta["version"]:
            return conflict(conn)

        issued, pending = archive_snapshots(conn)
        now = datetime.now(timezone.utc).isoformat()
        new_version = meta["version"] + 1
        conn.execute(
            "INSERT INTO archives (name, archived_at, status, version, issued, pending) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                now,
                meta["status"],
                meta["version"],
                json.dumps(issued, ensure_ascii=False),
                json.dumps(pending, ensure_ascii=False),
            ),
        )
        conn.execute("DELETE FROM cues")
        conn.execute(
            "UPDATE meta SET version = ?, status = 'setup', paused = 0 WHERE id = 1",
            (new_version,),
        )
        return build_state(conn)


@app.post("/api/archives/{archive_id}/start")
def start_from_archive(archive_id: int, body: StartFromArchiveIn):
    """以档案开新场。

    按档案已发顺序取编号与内容快照，再接待发顺序的编号与内容，全部作为
    新场待发提示；不继承发出时间或进度，也不修改原档案。仅开场前
    （setup）允许生效，携带页面所见版本号。"写入新场提示 + 推进到演出中
    + 版本号 +1"在同一事务内提交：任一步失败整体回滚，当前场次与档案都
    不变。同版本的重复/并发提交只有一次生效，其余收到 409 及最新状态。

    档案不存在返回 404；拼出的提示不满足 2～12 条 / 编号唯一 / 内容非空
    返回 400；非开场前或版本过期返回 409。
    """
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM archives WHERE id = ?", (archive_id,)
        ).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"error": "archive_not_found"})

    cues = archive_cues(row)
    error = cue_list_error(cues)
    if error:
        return bad_request(error)
    numbers = [n.strip() for n, _ in cues]
    contents = [c.strip() for _, c in cues]

    with lock, db() as conn:
        meta = conn.execute("SELECT version, status FROM meta WHERE id = 1").fetchone()
        if meta["status"] != "setup" or body.version != meta["version"]:
            return conflict(conn)

        conn.execute("DELETE FROM cues")
        for pos, (number, content) in enumerate(zip(numbers, contents), start=1):
            conn.execute(
                "INSERT INTO cues (number, content, position, status) "
                "VALUES (?, ?, ?, 'pending')",
                (number, content, pos),
            )
        conn.execute(
            "UPDATE meta SET version = version + 1, status = 'live', paused = 0 "
            "WHERE id = 1"
        )
        return build_state(conn)


app.mount(
    "/",
    StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True),
    name="static",
)
