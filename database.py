import sqlite3
import json
import hashlib
import os
from typing import Optional, Tuple, Dict, Any

import tempfile

DEFAULT_DB = os.path.join(tempfile.gettempdir(), "incidents.db")
DB_PATH = os.getenv("DB_PATH", DEFAULT_DB)

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            request_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            receipt_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            receipt_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
        """)
        conn.commit()

def hash_body(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()

def get_run_state(run_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.execute("SELECT state_json FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row["state_json"])
    return None

def check_incidents_replay(run_id: str, raw_bytes: bytes) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Returns (status, stored_state)
    status can be:
    - 'NEW': No previous run found for this run_id.
    - 'REPLAY': Same run_id and identical request hash. Return stored state.
    - 'CONFLICT': Same run_id but different request hash. Return 409 Conflict.
    """
    req_hash = hash_body(raw_bytes)
    with get_db() as conn:
        cur = conn.execute("SELECT request_hash, state_json FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return ("NEW", None)
        if row["request_hash"] == req_hash:
            return ("REPLAY", json.loads(row["state_json"]))
        else:
            return ("CONFLICT", None)

def save_run_state(run_id: str, raw_bytes: bytes, status: str, state_dict: Dict[str, Any]):
    req_hash = hash_body(raw_bytes)
    state_json = json.dumps(state_dict)
    with get_db() as conn:
        conn.execute("""
        INSERT INTO runs (run_id, request_hash, status, state_json, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(run_id) DO UPDATE SET
            status = excluded.status,
            state_json = excluded.state_json,
            updated_at = CURRENT_TIMESTAMP
        """, (run_id, req_hash, status, state_json))
        conn.commit()

def check_receipt_replay(receipt_id: str, run_id: str, raw_bytes: bytes) -> str:
    """
    Returns:
    - 'NEW': Receipt has not been processed yet.
    - 'REPLAY': Receipt processed with identical hash.
    - 'CONFLICT': Receipt ID already exists with different payload hash.
    """
    rec_hash = hash_body(raw_bytes)
    with get_db() as conn:
        cur = conn.execute("SELECT receipt_hash FROM receipts WHERE receipt_id = ?", (receipt_id,))
        row = cur.fetchone()
        if not row:
            return "NEW"
        if row["receipt_hash"] == rec_hash:
            return "REPLAY"
        return "CONFLICT"

def record_receipt(receipt_id: str, run_id: str, raw_bytes: bytes):
    rec_hash = hash_body(raw_bytes)
    with get_db() as conn:
        conn.execute("""
        INSERT OR IGNORE INTO receipts (receipt_id, run_id, receipt_hash)
        VALUES (?, ?, ?)
        """, (receipt_id, run_id, rec_hash))
        conn.commit()
