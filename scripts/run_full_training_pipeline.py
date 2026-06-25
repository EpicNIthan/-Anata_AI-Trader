from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request


def _run_step(command: list[str], *, name: str) -> dict:
    print(f"\n=== {name} ===")
    print(" ".join(str(item) for item in command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} did not return JSON output.") from exc


def _json_api(url: str, token: str, path: str, *, body: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = request.Request(
        url.rstrip("/") + path,
        data=data,
        method="POST" if body is not None else "GET",
        headers={"x-admin-token": token, "Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw data, prepare dataset, train best model, upload candidate, optionally activate/start paper trader.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--since-date", default=None)
    parser.add_argument("--until-date", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/pipeline_runs"))
    parser.add_argument("--activate-if-pass", action="store_true")
    parser.add_argument("--start-paper-trader-if-pass", action="store_true")
    parser.add_argument("--target", default="target_trade_quality_score")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--news-converter", choices=["smart", "finbert", "cryptobert", "rule-based"], default="smart")
    parser.add_argument("--cleanup-railway-after-download", action="store_true", help="Delete Railway raw export files/finished_data after local ZIP verification.")
    parser.add_argument("--delete-railway-db-rows", action="store_true", help="With cleanup, delete matching raw DB rows from Railway after verified PC download.")
    parser.add_argument("--delete-all-finished-data", action="store_true", help="With cleanup, delete every finished_data folder on Railway.")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"run_id": run_id, "steps": {}}
    raw_zip = run_dir / "raw_data_export.zip"

    download_command = [
        sys.executable,
        "scripts/download_raw_data.py",
        "--url",
        args.url,
        "--token",
        args.token,
        "--output",
        str(raw_zip),
        "--timeout",
        str(args.timeout),
    ]
    if args.date:
        download_command += ["--date", args.date]
    if args.since_date:
        download_command += ["--since-date", args.since_date]
    if args.until_date:
        download_command += ["--until-date", args.until_date]
    if args.use_all_data:
        download_command += ["--use-all-data"]
    if args.cleanup_railway_after_download:
        download_command += ["--cleanup-after-download"]
    if args.delete_railway_db_rows:
        download_command += ["--delete-railway-db-rows"]
    if args.delete_all_finished_data:
        download_command += ["--delete-all-finished-data"]
    manifest["steps"]["download_raw_data"] = _run_step(download_command, name="Download raw data")

    prepared = _run_step(
        [
            sys.executable,
            "scripts/prepare_training_data.py",
            "--input",
            str(raw_zip),
            "--output-dir",
            str(run_dir / "processed"),
            "--news-converter",
            args.news_converter,
        ],
        name="Prepare training data",
    )
    manifest["steps"]["prepare_training_data"] = prepared
    dataset_path = prepared["dataset"]

    trained = _run_step(
        [
            sys.executable,
            "scripts/train_best_model.py",
            "--dataset",
            dataset_path,
            "--target",
            args.target,
            "--out-dir",
            str(run_dir / "models"),
        ],
        name="Train best model",
    )
    manifest["steps"]["train_best_model"] = trained
    if trained.get("status") != "passed":
        manifest["status"] = "stopped_model_failed_checks"
        manifest_path = run_dir / "pipeline_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path), "message": trained.get("message")}, indent=2))
        return

    uploaded = _run_step(
        [
            sys.executable,
            "scripts/upload_model.py",
            "--url",
            args.url,
            "--token",
            args.token,
            "--package",
            trained["package"],
        ],
        name="Upload candidate model",
    )
    manifest["steps"]["upload_model"] = uploaded

    activation = None
    if args.activate_if_pass:
        model_id = trained.get("model_id")
        activation = _json_api(args.url, args.token, "/api/models/activate", body={"model_id": model_id})
        print(json.dumps({"activation": activation}, indent=2))
        manifest["steps"]["activate_model"] = activation

    auto_trader = None
    if args.start_paper_trader_if_pass:
        if activation is None or activation.get("status") != "active":
            print("Skipping auto trader start because model was not activated.")
        else:
            try:
                collectors = _json_api(args.url, args.token, "/api/collectors/status")
                manifest["steps"]["collector_status"] = collectors
            except Exception as exc:
                manifest["steps"]["collector_status"] = {"warning": str(exc)}
            auto_trader = _json_api(args.url, args.token, "/api/auto-trader/start", body={})
            print(json.dumps({"auto_trader": auto_trader}, indent=2))
            manifest["steps"]["auto_trader_start"] = auto_trader

    manifest["status"] = "ok"
    manifest["model_id"] = trained.get("model_id")
    manifest["activated"] = bool(activation and activation.get("status") == "active")
    manifest["auto_trader_started"] = bool(auto_trader and auto_trader.get("running"))
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "model_id": manifest["model_id"]}, indent=2))


if __name__ == "__main__":
    main()
