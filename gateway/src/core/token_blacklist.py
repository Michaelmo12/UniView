"""
SQLite-based JWT Token Blacklist
*maybe later change to redis or other db for scalability*
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading


class TokenBlacklist:
    """SQLite-based token blacklist singleton"""

    # singleton instance
    _instance: Optional["TokenBlacklist"] = None
    # stopping race conditions in multithreaded environment
    _lock = threading.Lock()

    # if 2 threads try to create instance simultaneously must have double check locking and creates only one instance without initializing
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # returns instance of the class
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    # initializes the database and creates table if not exists
    def __init__(self):
        if self._initialized:
            return

        # Create database in gateway root directory maybe later add to settings instead
        self.db_path = Path(__file__).parent.parent.parent / "token_blacklist.db"
        self._create_table()
        self._initialized = True

    # Creates the blacklist table if it doesn't exist
    def _create_table(self):
        """Create the blacklist table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS blacklisted_tokens (
                token TEXT PRIMARY KEY,
                blacklisted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        """
        )

        # Create index for faster expiry cleanup
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expires_at
            ON blacklisted_tokens(expires_at)
        """
        )

        conn.commit()
        conn.close()

    # Add a token to the blacklist
    def add_token(self, token: str, expires_at: datetime) -> bool:
        """
        Add a token to the blacklist

        Args:
            token: JWT token to blacklist
            expires_at: When the token expires

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT OR REPLACE INTO blacklisted_tokens (token, expires_at) VALUES (?, ?)",
                (token, expires_at.isoformat()),
            )

            conn.commit()
            conn.close()

            # Clean up expired tokens while we're here
            self._cleanup_expired()

            return True
        except Exception as e:
            print(f"Error adding token to blacklist: {e}")
            return False

    # Check if a token is blacklisted
    def is_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted

        Args:
            token: JWT token to check

        Returns:
            True if blacklisted, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT 1 FROM blacklisted_tokens
                WHERE token = ? AND expires_at > datetime('now')
                """,
                (token,),
            )

            result = cursor.fetchone()
            conn.close()

            return result is not None
        except Exception as e:
            print(f"Error checking token blacklist: {e}")
            # Fail open: allow request if database error
            return False

    # Remove expired tokens from the blacklist
    def _cleanup_expired(self):
        """Remove expired tokens from the blacklist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM blacklisted_tokens WHERE expires_at <= datetime('now')"
            )

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} expired tokens from blacklist")

        except Exception as e:
            print(f"Error cleaning up expired tokens: {e}")

    # Get statistics about the blacklist
    def get_stats(self) -> dict:
        """
        Get blacklist statistics

        Returns:
            Dictionary with blacklist stats
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM blacklisted_tokens WHERE expires_at > datetime('now')"
            )
            active_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM blacklisted_tokens")
            total_count = cursor.fetchone()[0]

            conn.close()

            return {
                "status": "healthy",
                "active_tokens": active_count,
                "total_tokens": total_count,
                "expired_tokens": total_count - active_count,
                "database_path": str(self.db_path),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Singleton instance
_blacklist = TokenBlacklist()


def add_token_to_blacklist(token: str, expires_at: datetime) -> bool:
    return _blacklist.add_token(token, expires_at)


def is_token_blacklisted(token: str) -> bool:
    return _blacklist.is_blacklisted(token)


def get_blacklist_stats() -> dict:
    return _blacklist.get_stats()
