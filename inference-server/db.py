"""Session factory so main.py/alpr.py don't hand-roll transactions.

Persistence is best-effort throughout this server: a DB outage must never
break inference, so callers should catch exceptions around `SessionLocal()`
usage themselves (see `persist.py`) rather than let them propagate.
"""
from sqlalchemy.orm import sessionmaker

from orm import engine_from_env

engine = engine_from_env()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
