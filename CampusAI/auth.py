"""
auth.py — Role-based login for CampusAI
---------------------------------------
Two roles:
  - admin: upload/index PDFs and use the chatbot
  - user: chatbot only (no document ingestion)
Passwords are stored hashed in data/users.json (created on first run).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# Demo logins (also used to seed data/users.json on first run)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
USER_USERNAME = os.getenv("USER_USERNAME", "student").strip().lower()
USER_PASSWORD = os.getenv("USER_PASSWORD", "student123")

USERS_FILE = _PROJECT_ROOT / "data" / "users.json"
PBKDF2_ROUNDS = 120_000


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ROUNDS,
    ).hex()


def _load_users() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    try:
        payload = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    users = payload.get("users", [])
    return users if isinstance(users, list) else []


def _save_users(users: list[dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(
        json.dumps({"users": users}, indent=2),
        encoding="utf-8",
    )


def _make_user(username: str, password: str, role: str) -> dict:
    salt = secrets.token_hex(16)
    return {
        "username": username.strip().lower(),
        "role": role,
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }


def ensure_default_users() -> None:
    """
    Create demo admin + user accounts if the store is empty.
    Existing users.json is never overwritten.
    """
    if _load_users():
        return
    users = [
        _make_user(ADMIN_USERNAME, ADMIN_PASSWORD, "admin"),
        _make_user(USER_USERNAME, USER_PASSWORD, "user"),
    ]
    _save_users(users)


def authenticate(username: str, password: str) -> Optional[dict]:
    """Return {username, role} on success, otherwise None."""
    name = (username or "").strip().lower()
    if not name or not password:
        return None
    for user in _load_users():
        if user.get("username") != name:
            continue
        salt = user.get("salt", "")
        expected = user.get("password_hash", "")
        actual = _hash_password(password, salt)
        if secrets.compare_digest(actual, expected):
            role = user.get("role", "user")
            if role not in {"admin", "user"}:
                role = "user"
            return {"username": name, "role": role}
    return None
