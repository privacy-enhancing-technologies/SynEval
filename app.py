#!/usr/bin/env python3
"""
SynEval Web Interface — Flask backend for the interactive evaluation frontend.

Usage:
    pip install flask
    python app.py
    Open http://localhost:5050
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("syneval_web")

# ---------------------------------------------------------------------------
# Task store
# ---------------------------------------------------------------------------
_tasks: dict = {}
_tasks_lock = threading.Lock()


def _new_task() -> str:
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            "id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "Initializing...",
            "dimension_status": {},
            "events": [],
            "results": None,
            "error": None,
            "start_time": None,
        }
    return task_id


def _push(task_id: str, progress: int, message: str,
          dimension: str = None, dim_status: str = None):
    with _tasks_lock:
        task = _tasks.get(task_id)
        if not task:
            return
        task["progress"] = progress
        task["message"] = message
        elapsed = (time.time() - task["start_time"]) if task["start_time"] else 0
        if dimension and dim_status:
            task["dimension_status"][dimension] = dim_status

        remaining = None
        if progress > 5 and elapsed > 0:
            rate = progress / elapsed
            remaining = max(0, int((100 - progress) / rate))

        task["events"].append({
            "progress": progress,
            "message": message,
            "dimension_status": dict(task["dimension_status"]),
            "elapsed": round(elapsed, 1),
            "remaining": remaining,
        })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Receive original + synthetic CSV and optional metadata JSON."""
    try:
        session_id = str(uuid.uuid4())
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True)

        original_path = synthetic_path = metadata_path = None

        if "original" in request.files:
            f = request.files["original"]
            original_path = session_dir / "original.csv"
            f.save(str(original_path))

        if "synthetic" in request.files:
            f = request.files["synthetic"]
            synthetic_path = session_dir / "synthetic.csv"
            f.save(str(synthetic_path))

        if "metadata" in request.files and request.files["metadata"].filename:
            f = request.files["metadata"]
            metadata_path = session_dir / "metadata.json"
            f.save(str(metadata_path))

        if not original_path or not synthetic_path:
            return jsonify({"success": False, "error": "Both original and synthetic datasets are required"}), 400

        orig_df = pd.read_csv(str(original_path))
        syn_df = pd.read_csv(str(synthetic_path))

        # Auto-detect column types
        text_cols, tab_cols = [], []
        try:
            from evaluation import auto_detect_columns
            detected = auto_detect_columns(orig_df)
            if isinstance(detected, dict):
                text_cols = detected.get("text", [])
                tab_cols  = detected.get("tabular", [])
            else:
                text_cols, tab_cols = detected
        except Exception as exc:
            logger.warning("auto_detect_columns failed: %s", exc)
            for col in orig_df.columns:
                if pd.api.types.is_numeric_dtype(orig_df[col]):
                    tab_cols.append(col)
                else:
                    avg_len = orig_df[col].dropna().astype(str).str.len().mean()
                    card = orig_df[col].nunique() / max(len(orig_df), 1)
                    if avg_len > 40 and card > 0.5:
                        text_cols.append(col)
                    else:
                        tab_cols.append(col)

        # Auto-generate metadata if not supplied
        metadata_generated = False
        metadata = {}
        if not metadata_path:
            metadata = _auto_metadata(orig_df)
            metadata_path = session_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))
            metadata_generated = True
        else:
            metadata = json.loads(metadata_path.read_text())

        return jsonify({
            "success": True,
            "session_id": session_id,
            "columns": list(orig_df.columns),
            "orig_shape": list(orig_df.shape),
            "syn_shape": list(syn_df.shape),
            "text_columns": text_cols,
            "tabular_columns": tab_cols,
            "metadata": metadata,
            "metadata_generated": metadata_generated,
        })

    except Exception:
        logger.error("Upload error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "File processing failed — please check the file format"}), 500


def _auto_metadata(df: pd.DataFrame) -> dict:
    cols = {}
    for col in df.columns:
        if col == "_id":
            cols[col] = {"sdtype": "id"}
            continue
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            cols[col] = {"sdtype": "boolean"}
        elif pd.api.types.is_numeric_dtype(dtype):
            cols[col] = {"sdtype": "numerical"}
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            cols[col] = {"sdtype": "datetime"}
        else:
            sample = df[col].dropna()
            if len(sample) == 0:
                cols[col] = {"sdtype": "categorical"}
                continue
            avg_len = sample.astype(str).str.len().mean()
            card = sample.nunique() / len(sample)
            if avg_len > 40 and card > 0.5:
                cols[col] = {"sdtype": "text"}
            else:
                cols[col] = {"sdtype": "categorical"}
    return {"columns": cols}


@app.route("/api/evaluate", methods=["POST"])
def start_evaluation():
    """Start evaluation in a background thread; return task_id immediately."""
    try:
        data = request.json
        session_id = data.get("session_id")
        dimensions = data.get("dimensions", ["fidelity", "utility", "diversity", "privacy"])
        metrics_config = data.get("metrics_config", {})
        utility_config = data.get("utility_config", {})

        session_dir = UPLOAD_DIR / session_id
        original_path = session_dir / "original.csv"
        synthetic_path = session_dir / "synthetic.csv"
        metadata_path = session_dir / "metadata.json"

        if not original_path.exists() or not synthetic_path.exists():
            return jsonify({"success": False, "error": "Uploaded files not found — please upload again"}), 400

        task_id = _new_task()
        threading.Thread(
            target=_run_evaluation,
            args=(task_id, str(original_path), str(synthetic_path),
                  str(metadata_path), dimensions, metrics_config, utility_config),
            daemon=True,
        ).start()

        return jsonify({"success": True, "task_id": task_id})

    except Exception:
        logger.error("Evaluate start error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Failed to start evaluation task"}), 500


def _make_serializable(obj):
    """Recursively make a result dict JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v) else v  # NaN → None
    return obj


def _run_evaluation(task_id, original_path, synthetic_path, metadata_path,
                    dimensions, metrics_config, utility_config):
    with _tasks_lock:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["start_time"] = time.time()

    try:
        _push(task_id, 2, "Loading datasets...")
        orig_df = pd.read_csv(original_path)
        syn_df = pd.read_csv(synthetic_path)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Strip _id if absent from metadata
        meta_cols = metadata.get("columns", {})
        for df in (orig_df, syn_df):
            if "_id" in df.columns and "_id" not in meta_cols:
                df.drop(columns=["_id"], inplace=True)

        _push(task_id, 5, "Initializing evaluators...")
        from run import SynEval
        evaluator = SynEval(syn_df, orig_df, metadata)

        results = {}
        n = len(dimensions)
        progress_per_dim = 88 // max(n, 1)

        for i, dim in enumerate(dimensions):
            p_start = 7 + i * progress_per_dim
            p_end = 7 + (i + 1) * progress_per_dim

            _push(task_id, p_start, f"Running {_dim_en(dim)} evaluation...",
                  dimension=dim, dim_status="running")

            selected = metrics_config.get(dim) or None

            try:
                if dim == "fidelity":
                    _push(task_id, p_start + 3, "Analysing data structure and distributions...")
                    results["fidelity"] = evaluator.evaluate_fidelity(selected_metrics=selected)

                elif dim == "utility":
                    in_cols = utility_config.get("input_columns", [])
                    out_cols = utility_config.get("output_columns", [])
                    if not in_cols or not out_cols:
                        raise ValueError("Utility evaluation requires input feature columns and a target column")
                    _push(task_id, p_start + 3, "Training machine learning models...")
                    results["utility"] = evaluator.evaluate_utility(
                        input_columns=in_cols,
                        output_columns=out_cols,
                        selected_metrics=selected,
                    )

                elif dim == "diversity":
                    _push(task_id, p_start + 3, "Computing entropy and coverage...")
                    results["diversity"] = evaluator.evaluate_diversity(selected_metrics=selected)

                elif dim == "privacy":
                    _push(task_id, p_start + 3, "Analysing privacy risks...")
                    results["privacy"] = evaluator.evaluate_privacy(selected_metrics=selected)

                _push(task_id, p_end, f"{_dim_en(dim)} evaluation complete",
                      dimension=dim, dim_status="done")

            except Exception as exc:
                logger.error("Dimension %s failed:\n%s", dim, traceback.format_exc())
                results[dim] = {"error": str(exc)}
                _push(task_id, p_end, f"{_dim_en(dim)} evaluation failed: {str(exc)[:80]}",
                      dimension=dim, dim_status="failed")

        _push(task_id, 97, "Compiling report...")
        clean = _make_serializable(results)

        with _tasks_lock:
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["results"] = clean

        _push(task_id, 100, "Evaluation complete!")

    except Exception as exc:
        logger.error("Task %s crashed:\n%s", task_id, traceback.format_exc())
        with _tasks_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(exc)
        _push(task_id, -1, f"Evaluation failed: {str(exc)}")


def _dim_en(dim: str) -> str:
    return {"fidelity": "Fidelity", "utility": "Utility",
            "diversity": "Diversity", "privacy": "Privacy"}.get(dim, dim.title())


@app.route("/api/progress/<task_id>")
def progress_stream(task_id):
    """Server-Sent Events stream — one event per update."""
    def generate():
        sent = 0
        deadline = time.time() + 3600  # max 1 h
        while time.time() < deadline:
            with _tasks_lock:
                task = _tasks.get(task_id)
                if not task:
                    yield f"data: {json.dumps({'error': 'task not found'})}\n\n"
                    return
                new_events = task["events"][sent:]
                status = task["status"]
                results = task.get("results")
                error = task.get("error")

            for ev in new_events:
                yield f"data: {json.dumps(ev)}\n\n"
                sent += 1

            if status == "completed":
                yield f"data: {json.dumps({'done': True, 'results': results})}\n\n"
                return
            if status == "failed":
                yield f"data: {json.dumps({'done': True, 'error': error})}\n\n"
                return

            time.sleep(0.4)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/results/<task_id>")
def get_results(task_id):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({
        "status": task["status"],
        "results": task.get("results"),
        "error": task.get("error"),
    })


if __name__ == "__main__":
    print("\n  SynEval Web Interface")
    print("  ─────────────────────")
    print("  http://localhost:5050\n")
    app.run(debug=False, host="0.0.0.0", port=5050, threaded=True)
