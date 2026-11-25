"""Logging helpers shared by the tools in this folder.

Every script that uses this writes a run log next to its output (`<output>.log`): timestamps, every subprocess
call with its duration and full context for every failure (command, exit code, stderr/stdout tails, tracebacks).
Records that end in failure carry the same context, so a results file alone explains why something failed.

    log = runlog.setup("/path/to/results.jsonl", "run_set")
    r = runlog.run(["cmd", "arg"], timeout=60, log=log, what="compile foo")
    if not r["ok"]: record["failure"] = runlog.failure(r)
"""
import logging
import os
import subprocess
import sys
import time
import traceback
from collections import Counter

TAIL = 4000


def setup(output_path, name, verbose=False):
    """A logger writing DEBUG and up to `<output_path>.log` and INFO and up to stderr."""
    log_path = output_path + ".log" if not output_path.endswith(".log") else output_path
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(threadName)s %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)
    log.log_path = log_path
    log.info(f"{name}: log at {log_path}")
    log.debug("argv: %s", " ".join(sys.argv))
    return log


def tail(text, n=TAIL):
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return text if len(text) <= n else "..." + text[-n:]


def run(cmd, timeout, log, what, **kwargs):
    """Run a command; never raises for failures. Returns
    {ok, rc, seconds, timed_out, stdout, stderr, cmd, what, error}."""
    t0 = time.time()
    rec = {"what": what, "cmd": [str(c) for c in cmd], "timed_out": False, "rc": None, "error": None}
    log.debug("start %s: %s", what, " ".join(rec["cmd"]))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
        rec.update(rc=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        rec.update(timed_out=True, stdout=tail(e.stdout), stderr=tail(e.stderr))
    except OSError as e:
        rec.update(error=f"{type(e).__name__}: {e}", stdout="", stderr="")
    rec["seconds"] = round(time.time() - t0, 3)
    rec["ok"] = not rec["timed_out"] and rec["error"] is None and rec["rc"] == 0
    if rec["ok"]:
        log.debug("done  %s in %.2fs", what, rec["seconds"])
    else:
        why = ("timed out after %ss" % timeout) if rec["timed_out"] else (rec["error"] or f"exit code {rec['rc']}")
        log.warning("FAILED %s (%s, %.2fs)\n  cmd: %s\n  stderr: %s\n  stdout: %s", what, why, rec["seconds"],
                    " ".join(rec["cmd"]), tail(rec["stderr"], 1500).strip() or "(empty)",
                    tail(rec["stdout"], 500).strip() or "(empty)")
    return rec


def failure(r):
    """The part of a run() record worth keeping in a results file."""
    return {"what": r["what"], "cmd": " ".join(r["cmd"]), "rc": r["rc"], "timed_out": r["timed_out"],
            "seconds": r["seconds"], "error": r["error"], "stderr": tail(r.get("stderr"), 1500),
            "stdout": tail(r.get("stdout"), 500)}


def exception(e, log=None, what=""):
    """A record for a caught exception, logged with its traceback."""
    tb = traceback.format_exc()
    if log:
        log.error("EXCEPTION %s: %s: %s\n%s", what, type(e).__name__, e, tb)
    return {"what": what, "exception": f"{type(e).__name__}: {e}", "traceback": tail(tb, 3000)}


def summarise_failures(records, key, log, limit=10):
    """Group failed records by `key(record)` and log each group with one example."""
    groups, example = Counter(), {}
    for r in records:
        k = key(r)
        if k is None:
            continue
        groups[k] += 1
        example.setdefault(k, r.get("name", ""))
    if not groups:
        return
    log.info("failures by reason (details in %s):", log.log_path)
    for k, n in groups.most_common(limit):
        log.info("  %5d  %s   e.g. %s", n, k, example[k])
