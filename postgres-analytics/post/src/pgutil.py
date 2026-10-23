"""Shared helpers: start the local PostgreSQL cluster and connect to it.

The project runs against a local PostgreSQL 16 install (Debian/Ubuntu
package). The database and login role are created once by setup_db.py;
these are throwaway local-development credentials, not a real secret.
"""

import subprocess
import time

import psycopg

DB_NAME = "customer_analytics"
DB_USER = "osamah"
DB_PASSWORD = "osamah"  # local dev only
DSN = f"host=localhost port=5432 dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
ADMIN_DSN = "host=localhost port=5432 dbname=postgres user=postgres"


def ensure_server() -> None:
    """Start the PostgreSQL cluster if it is not already running."""
    subprocess.run(
        ["pg_ctlcluster", "16", "main", "start"],
        capture_output=True, text=True,  # 'already running' is fine
    )
    # pg_isready needs no credentials, so this works before the app role
    # and database have been bootstrapped.
    for _ in range(30):
        result = subprocess.run(
            ["pg_isready", "-h", "localhost", "-p", "5432"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("PostgreSQL did not come up on localhost:5432")


def connect() -> psycopg.Connection:
    ensure_server()
    return psycopg.connect(DSN)
