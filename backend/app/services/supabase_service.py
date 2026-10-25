import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _config():
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SECRET_KEY")
    )

    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(or SUPABASE_SECRET_KEY) in backend/.env"
        )

    return url, key


def db_request(method, table, *, params=None, payload=None):
    url, key = _config()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        response = requests.request(
            method,
            f"{url}/rest/v1/{table}",
            headers=headers,
            params=params,
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase connection failed: {exc}") from exc

    if not response.ok:
        detail = response.text[:1000]
        if response.status_code == 409:
            raise ValueError(f"Record already exists: {detail}")
        raise RuntimeError(
            f"Supabase returned HTTP {response.status_code}: {detail}"
        )

    if response.status_code == 204 or not response.content:
        return []

    return response.json()
