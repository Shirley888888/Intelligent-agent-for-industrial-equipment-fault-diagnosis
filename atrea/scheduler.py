"""Resilient daily Asia/Shanghai scheduler for the ATREA research callback."""
from __future__ import annotations
import argparse, importlib, json, os, time, traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")


def next_run(now=None, hour=8, minute=0):
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    target = now.astimezone(TZ).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now.astimezone(TZ):
        target += timedelta(days=1)
    return target


def _write_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_due_once(callback, *, state_path="agent_workspace/daily_scheduler.json",
                 now=None, hour=8, minute=0, retry_seconds=900):
    """Run at most once successfully per China calendar day; failures remain retryable."""
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(TZ)
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    today = now.date().isoformat()
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now < due:
        return {"status": "not_due", "next_run": due.isoformat()}
    if state.get("last_success_date") == today:
        return {"status": "already_succeeded", "last_success_date": today}
    last_attempt = state.get("last_attempt_at")
    if last_attempt:
        try:
            elapsed = (now - datetime.fromisoformat(last_attempt).astimezone(TZ)).total_seconds()
            if elapsed < retry_seconds:
                return {"status": "retry_backoff", "retry_in_seconds": int(retry_seconds-elapsed)}
        except Exception:
            pass
    # Prevent two scheduler processes from launching duplicate GPU training jobs.
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            stale = (time.time() - lock_path.stat().st_mtime) > 24 * 3600
            if stale:
                lock_path.unlink(missing_ok=True)
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                return {"status": "already_running", "lock_path": str(lock_path)}
        except (FileNotFoundError, FileExistsError):
            return {"status": "already_running", "lock_path": str(lock_path)}
    os.write(lock_fd, f"pid={os.getpid()} started={now.isoformat()}".encode("utf-8"))
    os.close(lock_fd)
    try:
        state.update({"last_attempt_at": now.isoformat(), "last_attempt_status": "running"})
        _write_state(path, state)
        try:
            result = callback(now=now)
            state.update({"last_success_date": today, "last_run_at": now.isoformat(),
                          "last_attempt_status": "success",
                          "last_result": result if isinstance(result, (dict, str, int, float, bool, list)) else str(result),
                          "last_error": None})
            _write_state(path, state)
            return {"status": "success", "result": result}
        except Exception as exc:
            state.update({"last_attempt_status": "failed",
                          "last_error": f"{type(exc).__name__}: {exc}",
                          "last_traceback": traceback.format_exc(limit=12)})
            _write_state(path, state)
            return {"status": "failed", "error": state["last_error"]}
    finally:
        lock_path.unlink(missing_ok=True)


def run_forever(callback, *, state_path="agent_workspace/daily_scheduler.json",
                hour=8, minute=0, poll_seconds=20, retry_seconds=900):
    """Poll continuously; callback failures are logged and retried without killing scheduler."""
    while True:
        run_due_once(callback, state_path=state_path, hour=hour, minute=minute,
                     retry_seconds=retry_seconds)
        time.sleep(poll_seconds)


def load_callback(spec):
    module_name, func_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), func_name)


def main():
    p = argparse.ArgumentParser(description="Daily 08:00 China-time ATREA scheduler")
    p.add_argument("--callback", required=True, help="Python callback as package.module:function; callback(now=aware_datetime)")
    p.add_argument("--state", default="agent_workspace/daily_scheduler.json")
    p.add_argument("--hour", type=int, default=8); p.add_argument("--minute", type=int, default=0)
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--retry-seconds", type=int, default=900)
    a = p.parse_args()
    if not 0 <= a.hour <= 23 or not 0 <= a.minute <= 59:
        p.error("--hour must be 0..23 and --minute 0..59")
    if a.poll_seconds < 1 or a.retry_seconds < 1:
        p.error("poll/retry intervals must be positive")
    run_forever(load_callback(a.callback), state_path=a.state, hour=a.hour,
                minute=a.minute, poll_seconds=a.poll_seconds, retry_seconds=a.retry_seconds)


if __name__ == "__main__":
    main()
