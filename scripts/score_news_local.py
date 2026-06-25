from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _open_text(path: Path, mode: str):
    return gzip.open(path, mode, newline="", encoding="utf-8") if path.suffix == ".gz" else open(path, mode, newline="", encoding="utf-8")


def _normalize_label(label: str) -> str:
    lowered = label.lower()
    if "positive" in lowered or lowered in {"label_2", "2"}:
        return "positive"
    if "negative" in lowered or lowered in {"label_0", "0"}:
        return "negative"
    return "neutral"


def _load_pipeline(model_name: str) -> Any | None:
    try:
        from transformers import pipeline
    except Exception as exc:
        print(f"WARNING: transformers is not installed or failed to import: {type(exc).__name__}: {exc}")
        print("Install locally with: python -m pip install -r requirements-hf.txt")
        return None
    return pipeline("sentiment-analysis", model=model_name)


def _local_hf_score(model: Any, text: str) -> tuple[str, float, float, dict[str, Any]]:
    result = model(text[:1800], truncation=True)
    first = result[0] if isinstance(result, list) and result else result
    if not isinstance(first, dict):
        return "neutral", 0.0, 0.0, {"raw_result": result}
    label = _normalize_label(str(first.get("label") or "neutral"))
    confidence = float(first.get("score") or 0.0)
    return label, LABEL_TO_SCORE[label] * confidence, confidence, {"raw_result": first}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw Railway news locally and write uploadable sentiment JSONL.")
    parser.add_argument("--input", type=Path, default=Path("datasets/latest_raw_news.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("datasets/latest_news_sentiment.jsonl.gz"))
    parser.add_argument("--model", default="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    parser.add_argument("--fallback-only", action="store_true", help="Use the app rule-based analyzer instead of local Hugging Face.")
    args = parser.parse_args()

    model = None if args.fallback_only else _load_pipeline(args.model)
    if model is None:
        from app.ai.news_sentiment import analyze_news
    else:
        from app.ai.news_sentiment import _keyword_analysis

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_in = 0
    rows_out = 0
    with _open_text(args.input, "rt") as source, _open_text(args.output, "wt") as target:
        reader = csv.DictReader(source)
        for row in reader:
            rows_in += 1
            text = f"{row.get('title') or ''}\n{row.get('raw_text') or ''}".strip()
            if not text:
                continue
            if model is None:
                result = analyze_news(text)
                payload = result.raw_payload
                label = result.label
                sentiment_score = result.sentiment_score
                confidence = result.confidence
                risk_score = result.risk_score
                topics = result.topics
                affected_symbols = result.affected_symbols
                model_name = result.model_name
            else:
                keyword_sentiment, risk_score, topics, affected_symbols = _keyword_analysis(text)
                label, sentiment_score, confidence, payload = _local_hf_score(model, text)
                payload["keyword_sentiment_overlay"] = keyword_sentiment
                model_name = args.model
            target.write(
                json.dumps(
                    {
                        "article_id": row.get("article_id"),
                        "url": row.get("url"),
                        "sentiment_score": sentiment_score,
                        "risk_score": risk_score,
                        "topics": topics,
                        "affected_symbols": affected_symbols,
                        "label": label,
                        "confidence": confidence,
                        "model_name": model_name,
                        "raw_payload": payload,
                    },
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )
            rows_out += 1
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows_in": rows_in, "rows_out": rows_out, "model": args.model if model else "rule-based-local-fallback"}, indent=2))


if __name__ == "__main__":
    main()
