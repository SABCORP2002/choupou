from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from config import settings

DB_PATH = Path(settings.db_path)


def list_users() -> list:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        c = conn.cursor()
        c.execute("SELECT id, email, role FROM users ORDER BY id ASC")
        return c.fetchall()
    finally:
        conn.close()


def make_admin(email: str) -> bool:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        c = conn.cursor()
        c.execute("SELECT id, role FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        if not user:
            print(f"[ERREUR] Aucun utilisateur avec email={email}")
            return False

        if user[1] == "admin":
            print(f"[OK] {email} est deja admin.")
            return True

        c.execute("UPDATE users SET role = ? WHERE email = ?", ("admin", email))
        conn.commit()
        print(f"[OK] {email} est maintenant admin.")
        return True
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Promouvoir un utilisateur en admin.")
    parser.add_argument("--email", help="Email de l'utilisateur a promouvoir.")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"[ERREUR] Base SQLite introuvable: {DB_PATH}")
        return 1

    users = list_users()
    if not users:
        print("[INFO] Aucun utilisateur en base.")
        return 0

    print("=== Utilisateurs ===")
    for user in users:
        print(f"- id={user[0]} email={user[1]} role={user[2]}")

    email = args.email
    if not email:
        email = input("Email a promouvoir admin: ").strip()
    if not email:
        print("[ERREUR] Email vide.")
        return 2
    return 0 if make_admin(email) else 3


if __name__ == "__main__":
    raise SystemExit(main())
