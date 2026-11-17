
import os
import re
import sqlite3

from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from .config import DB_PATH

# Load local .env file (Cloud Run supplies environment variables directly).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")


SQLITE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS farmers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    farm_name TEXT NOT NULL,
    location_name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    country TEXT,
    state TEXT,
    city TEXT,
    crop TEXT DEFAULT 'rice',
    field_id TEXT DEFAULT 'FIELD-001',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    farmer_id INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (farmer_id)
        REFERENCES farmers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    farmer_id INTEGER,
    field_id TEXT,
    crop TEXT,
    growth_stage TEXT,
    temperature REAL,
    humidity REAL,
    rainfall REAL,
    soil_moisture REAL,
    disease TEXT,
    confidence REAL,
    uncertain INTEGER,
    severity REAL,
    severity_class TEXT,
    dpi REAL,
    risk_score REAL,
    risk_level TEXT,
    recommendation TEXT,
    FOREIGN KEY (farmer_id)
        REFERENCES farmers(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS environment_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id TEXT,
    timestamp TEXT,
    temperature REAL,
    humidity REAL,
    rainfall REAL,
    soil_moisture REAL,
    disease_score REAL
);
"""


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS farmers (
    id BIGSERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    farm_name TEXT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    country TEXT,
    state TEXT,
    city TEXT,
    crop TEXT DEFAULT 'rice',
    field_id TEXT DEFAULT 'FIELD-001',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    farmer_id BIGINT NOT NULL REFERENCES farmers(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    farmer_id BIGINT REFERENCES farmers(id) ON DELETE SET NULL,
    field_id TEXT,
    crop TEXT,
    growth_stage TEXT,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    rainfall DOUBLE PRECISION,
    soil_moisture DOUBLE PRECISION,
    disease TEXT,
    confidence DOUBLE PRECISION,
    uncertain INTEGER,
    severity DOUBLE PRECISION,
    severity_class TEXT,
    dpi DOUBLE PRECISION,
    risk_score DOUBLE PRECISION,
    risk_level TEXT,
    recommendation TEXT
);

CREATE TABLE IF NOT EXISTS environment_readings (
    id BIGSERIAL PRIMARY KEY,
    field_id TEXT,
    timestamp TIMESTAMP,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    rainfall DOUBLE PRECISION,
    soil_moisture DOUBLE PRECISION,
    disease_score DOUBLE PRECISION
);
"""


class PostgresConnection:
    """Small compatibility layer for existing SQLite-style queries."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query, params=()):
        query = query.replace("?", "%s")
        return self.connection.execute(query, params)


@contextmanager
def get_db():

    if DATABASE_URL:

        import psycopg
        from psycopg.rows import dict_row

        conn = psycopg.connect(
            DATABASE_URL,
            row_factory=dict_row
        )

        db = PostgresConnection(conn)

        try:
            with conn.cursor() as cursor:
                cursor.execute(POSTGRES_SCHEMA)

            yield db
            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    else:

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SQLITE_SCHEMA)

        try:
            yield conn
            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()
