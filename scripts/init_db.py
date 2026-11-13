"""Create all PostgreSQL tables."""

from dotenv import load_dotenv

load_dotenv()

from ragapp.storage.db import init_db

if __name__ == "__main__":
    init_db()
    print("Database tables created.")
