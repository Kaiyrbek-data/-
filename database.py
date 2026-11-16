"""SQLite helpers. The database path does not depend on the working directory."""
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent / "data" / "project.sqlite3"


def connect():
    """Return a connection. Close it after use."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=30)


def save_dataframe(df, table_name, if_exists="fail"):
    """Save a pandas DataFrame; use append/replace explicitly when needed."""
    connection = connect()
    try:
        df.to_sql(table_name, connection, if_exists=if_exists, index=False)
    finally:
        connection.close()


def read_dataframe(query, params=None):
    """Read SQL into a pandas DataFrame. Use ? placeholders for values."""
    import pandas as pd
    connection = connect()
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


def list_tables():
    connection = connect()
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        connection.close()


if __name__ == "__main__":
    connection = connect()
    try:
        version = connection.execute("SELECT sqlite_version()").fetchone()[0]
    finally:
        connection.close()
    print(f"SQLite {version}")
    print(f"Database: {DB_PATH}")
    print(f"Tables: {list_tables()}")
