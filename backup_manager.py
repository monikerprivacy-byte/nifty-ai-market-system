"""Backup Manager — automatic daily zip of duckdb files."""

import os, shutil, logging, zipfile, time
from datetime import datetime, timedelta
from pathlib import Path
from config_manager import get_config

logger = logging.getLogger("backup_manager")

class BackupManager:
    def __init__(self):
        cfg = get_config()
        self.enabled = cfg.get("backup.enabled", True)
        self.interval_hours = cfg.get("backup.interval_hours", 24)
        self.max_backups = cfg.get("backup.max_backups", 7)
        self.compress = cfg.get("backup.compress", True)

        self.data_dir = Path(cfg.get("app.data_dir", "/Volumes/Untitled/market_data"))
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        self._last_backup = None
        self.db_paths = [
            Path(cfg.get("databases.market", "")),
            Path(cfg.get("databases.memory", "")),
        ]

    def need_backup(self):
        """Check if backup is needed based on interval."""
        if not self.enabled:
            return False
        if self._last_backup is None:
            return True
        return time.time() - self._last_backup >= self.interval_hours * 3600

    def run_backup(self):
        """Run backup now. Returns dict with status."""
        if not self.enabled:
            return {"status": "skipped", "reason": "Backup disabled in config"}

        existing = [p for p in self.db_paths if p and p.exists()]
        if not existing:
            return {"status": "skipped", "reason": "No database files found"}

        try:
            # Flush DBs (vacuum/checkpoint)
            import duckdb
            for p in existing:
                try:
                    con = duckdb.connect(str(p))
                    con.execute("CHECKPOINT")
                    con.close()
                except:
                    pass

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"market_data_backup_{timestamp}.zip"
            backup_path = self.backup_dir / backup_name

            with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED if self.compress else zipfile.ZIP_STORED) as zf:
                for p in existing:
                    zf.write(str(p), p.name)
                    logger.info(f"  Backed up {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")

            self._last_backup = time.time()
            self._cleanup_old()

            size_mb = backup_path.stat().st_size / 1024 / 1024
            logger.info(f"Backup saved: {backup_name} ({size_mb:.1f} MB)")
            return {
                "status": "success",
                "file": backup_name,
                "size_mb": round(size_mb, 1),
                "databases": [p.name for p in existing],
                "timestamp": timestamp,
            }
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _cleanup_old(self):
        """Remove old backups beyond max_backups."""
        backups = sorted(self.backup_dir.glob("market_data_backup_*.zip"))
        while len(backups) > self.max_backups:
            oldest = backups.pop(0)
            try:
                oldest.unlink()
                logger.info(f"  Removed old backup: {oldest.name}")
            except Exception as e:
                logger.warning(f"  Could not remove {oldest.name}: {e}")

    def list_backups(self):
        """List available backups."""
        backups = sorted(self.backup_dir.glob("market_data_backup_*.zip"))
        return [
            {
                "name": b.name,
                "size_mb": round(b.stat().st_size / 1024 / 1024, 1),
                "modified": datetime.fromtimestamp(b.stat().st_mtime).isoformat(),
            }
            for b in backups
        ]

    def restore_backup(self, backup_name):
        """Restore from a specific backup file."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            return {"status": "failed", "error": f"Backup not found: {backup_name}"}

        try:
            with zipfile.ZipFile(backup_path, "r") as zf:
                zf.extractall(self.data_dir)
            logger.info(f"Restored from {backup_name}")
            return {"status": "success", "restored_from": backup_name}
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {"status": "failed", "error": str(e)}

# Singleton
_instance = None

def get_backup_manager():
    global _instance
    if _instance is None:
        _instance = BackupManager()
    return _instance
