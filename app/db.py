"""Persistence layer: datasets, their current/original data, profiles and history.

Works with Postgres (hosted: Neon / Render / Supabase) and SQLite (local) through
SQLAlchemy.  DataFrames are stored as compressed pickles so every dtype survives
a round trip.  A small in-process cache avoids unpickling on every request.
"""
from __future__ import annotations

import json
import pickle
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import Column, DateTime, Integer, LargeBinary, String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings
from .utils import jsonable


class Base(DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"
    id = Column(String(16), primary_key=True)
    session_token = Column(String(64), index=True, nullable=False)
    file_name = Column(String(255), nullable=False)
    meta_json = Column(Text, default="{}")
    original_blob = Column(LargeBinary, nullable=False)
    current_blob = Column(LargeBinary, nullable=False)
    profile_json = Column(Text)             # profile of the current data
    profile_original_json = Column(Text)    # profile of the uploaded data
    history_json = Column(Text, default="[]")
    last_plan_json = Column(Text)
    rows = Column(Integer, default=0)
    columns = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def _make_engine():
    url = settings.database_url.strip().strip('"').strip("'")
    # Accept every spelling Neon / Render / Supabase hand out and force the psycopg2 driver we ship.
    for prefix in ("postgres://", "postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://", "postgresql+asyncpg://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg2://" + url[len(prefix):]
            break
    if url.startswith("postgresql+psycopg2://"):
        url = url.replace("&channel_binding=require", "").replace("?channel_binding=require&", "?").replace("?channel_binding=require", "")
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=3, max_overflow=5, pool_recycle=300)
    return create_engine(url, **kwargs)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------- blobs & cache

def df_to_blob(df: pd.DataFrame) -> bytes:
    return zlib.compress(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL), 6)


def blob_to_df(blob: bytes) -> pd.DataFrame:
    return pickle.loads(zlib.decompress(blob))


class _Cache:
    """Tiny LRU for unpickled frames: {dataset_id: {"original": df, "current": df}}."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.items: OrderedDict[str, dict[str, pd.DataFrame]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
            return None

    def put(self, key: str, value: dict[str, pd.DataFrame]):
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)

    def drop(self, key: str):
        with self.lock:
            self.items.pop(key, None)


cache = _Cache()


# ---------------------------------------------------------------------- CRUD helpers

def _loads(text: str | None, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def create_dataset(session_token: str, df: pd.DataFrame, meta: dict[str, Any]) -> str:
    dataset_id = uuid.uuid4().hex[:12]
    blob = df_to_blob(df)
    with SessionLocal() as s:
        # enforce per-session quota: drop the oldest datasets beyond the limit
        existing = s.execute(select(DatasetRow.id).where(DatasetRow.session_token == session_token)
                             .order_by(DatasetRow.created_at.desc())).scalars().all()
        for old in existing[max(settings.max_datasets_per_session - 1, 0):]:
            s.execute(delete(DatasetRow).where(DatasetRow.id == old))
            cache.drop(old)
        s.add(DatasetRow(id=dataset_id, session_token=session_token, file_name=meta.get("file_name", "dataset"),
                         meta_json=json.dumps(jsonable(meta)), original_blob=blob, current_blob=blob,
                         rows=int(df.shape[0]), columns=int(df.shape[1]), size_bytes=len(blob) * 2))
        s.commit()
    cache.put(dataset_id, {"original": df, "current": df.copy()})
    return dataset_id


def load_dataset(dataset_id: str, session_token: str) -> dict[str, Any] | None:
    """Return a dict like the old in-memory store (original, current, meta, profile, history ...)."""
    with SessionLocal() as s:
        row = s.get(DatasetRow, dataset_id)
        if row is None or row.session_token != session_token:
            return None
        frames = cache.get(dataset_id)
        if frames is None:
            frames = {"original": blob_to_df(row.original_blob), "current": blob_to_df(row.current_blob)}
            cache.put(dataset_id, frames)
        return {
            "id": row.id, "file_name": row.file_name, "meta": _loads(row.meta_json, {}),
            "original": frames["original"], "current": frames["current"],
            "profile": _loads(row.profile_json, None), "profile_original": _loads(row.profile_original_json, None),
            "history": _loads(row.history_json, []), "last_plan": _loads(row.last_plan_json, None),
            "created": row.created_at.isoformat(timespec="seconds") if row.created_at else None,
            "updated": row.updated_at.isoformat(timespec="seconds") if row.updated_at else None,
            "path": None,
        }


def save_dataset(ds: dict[str, Any], *, current: bool = False, original: bool = False,
                 profile: bool = False, history: bool = False, meta: bool = False, plan: bool = False):
    """Persist the parts of `ds` that changed."""
    with SessionLocal() as s:
        row = s.get(DatasetRow, ds["id"])
        if row is None:
            return
        if original:
            row.original_blob = df_to_blob(ds["original"])
        if current:
            row.current_blob = df_to_blob(ds["current"])
            row.rows, row.columns = int(ds["current"].shape[0]), int(ds["current"].shape[1])
        if profile:
            row.profile_json = json.dumps(jsonable(ds.get("profile")))
            row.profile_original_json = json.dumps(jsonable(ds.get("profile_original")))
        if history:
            row.history_json = json.dumps(jsonable(ds.get("history", [])))
        if meta:
            row.meta_json = json.dumps(jsonable(ds.get("meta", {})))
            row.file_name = ds.get("meta", {}).get("file_name", row.file_name)
        if plan:
            row.last_plan_json = json.dumps(jsonable(ds.get("last_plan")))
        row.size_bytes = len(row.original_blob or b"") + len(row.current_blob or b"")
        row.updated_at = datetime.utcnow()
        s.commit()
    cache.put(ds["id"], {"original": ds["original"], "current": ds["current"]})


def list_datasets(session_token: str) -> list[dict[str, Any]]:
    with SessionLocal() as s:
        rows = s.execute(select(DatasetRow.id, DatasetRow.file_name, DatasetRow.rows, DatasetRow.columns,
                                DatasetRow.created_at, DatasetRow.updated_at, DatasetRow.history_json,
                                DatasetRow.profile_json)
                         .where(DatasetRow.session_token == session_token)
                         .order_by(DatasetRow.updated_at.desc())).all()
    out = []
    for r in rows:
        prof = _loads(r.profile_json, None)
        out.append({"dataset_id": r.id, "file_name": r.file_name, "rows": r.rows, "columns": r.columns,
                    "created": r.created_at.isoformat(timespec="seconds") if r.created_at else None,
                    "updated": r.updated_at.isoformat(timespec="seconds") if r.updated_at else None,
                    "passes": len(_loads(r.history_json, [])),
                    "score": prof["score"]["score"] if prof else None})
    return out


def delete_dataset(dataset_id: str, session_token: str) -> bool:
    with SessionLocal() as s:
        row = s.get(DatasetRow, dataset_id)
        if row is None or row.session_token != session_token:
            return False
        s.delete(row)
        s.commit()
    cache.drop(dataset_id)
    return True


def cleanup_old(days: int | None = None) -> int:
    days = settings.dataset_ttl_days if days is None else days
    if days <= 0:
        return 0
    cutoff = datetime.utcnow() - timedelta(days=days)
    with SessionLocal() as s:
        ids = s.execute(select(DatasetRow.id).where(DatasetRow.updated_at < cutoff)).scalars().all()
        if ids:
            s.execute(delete(DatasetRow).where(DatasetRow.id.in_(ids)))
            s.commit()
    for i in ids:
        cache.drop(i)
    return len(ids)


def stats() -> dict[str, Any]:
    with SessionLocal() as s:
        rows = s.execute(select(DatasetRow.size_bytes)).scalars().all()
    return {"datasets": len(rows), "storage_mb": round(sum(r or 0 for r in rows) / 1e6, 2)}
