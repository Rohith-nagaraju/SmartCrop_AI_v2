import sqlite3

from contextlib import contextmanager

from .config import DB_PATH


SCHEMA = """
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
        REFERENCES farmers(id)
        ON DELETE CASCADE
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
        REFERENCES farmers(id)
        ON DELETE SET NULL
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


@contextmanager
def get_db():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(SCHEMA)

    try:
        yield conn

        conn.commit()

    finally:
        conn.close()