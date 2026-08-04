"""Create structured teacher labels from raw-news archives on the laptop.

The deterministic rule teacher is always available.  ``--teacher-mode hf`` adds a
locally loaded Hugging Face sentiment teacher and merges its numeric output into the
same strictly validated event schema.  Neither mode is imported by Railway runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from scripts.news_student_utils import document_from_row, ensure_writable_output, iter_jsonl
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from news_student_utils import document_from_row, ensure_writable_output, iter_jsonl

from app.intelligence.providers import LocalRuleProvider
from app.intelligence.schemas import EventDirection, ProviderResponse, StructuredNewsEvent


class HeavyLocalTeacher:
    """Laptop-only hybrid teacher: heavy sentiment plus deterministic structure."""

    name = "local_hf_teacher"

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        classifier: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.revision = revision
        if classifier is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face teacher dependencies are missing; install requirements-hf.txt"
                ) from exc
            load_kwargs: dict[str, Any] = {
                "revision": revision,
                "local_files_only": local_files_only,
                "trust_remote_code": False,
            }
            tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
            trained_model = AutoModelForSequenceClassification.from_pretrained(model, **load_kwargs)
            classifier = pipeline(
                "text-classification",
                model=trained_model,
                tokenizer=tokenizer,
                device=-1,
            )
        self.classifier = classifier
        self.rules = LocalRuleProvider()

    @staticmethod
    def _sentiment(result: Mapping[str, Any]) -> tuple[float, float, str]:
        label = str(result.get("label") or "neutral").strip().lower()
        score = max(0.0, min(float(result.get("score") or 0.0), 1.0))
        if any(token in label for token in ("positive", "bullish", "label_2")):
            return score, score, EventDirection.BULLISH.value
        if any(token in label for token in ("negative", "bearish", "label_0")):
            return -score, score, EventDirection.BEARISH.value
        return 0.0, score, EventDirection.NEUTRAL.value

    async def enrich(self, document: Any, *, prompt_version: str) -> Any:
        baseline = await self.rules.enrich(document, prompt_version="offline-rule-structure-v1")
        raw = self.classifier(document.text[:12_000], truncation=True)
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], Mapping):
            raise ValueError("local heavy teacher returned an invalid classification")
        sentiment, confidence, direction = self._sentiment(raw[0])
        payload = baseline.event.model_dump()
        payload.update(
            {
                "sentiment": sentiment,
                "direction": direction,
                "confidence": confidence,
                "provider": self.name,
                "model": self.model,
                "prompt_version": prompt_version,
                "metadata": {
                    **dict(payload.get("metadata") or {}),
                    "teacher_mode": "huggingface_text_classification",
                    "teacher_revision": self.revision,
                    "heavy_label": str(raw[0].get("label") or ""),
                    "heavy_score": confidence,
                    "rule_structure_model": self.rules.model,
                },
            }
        )
        event = StructuredNewsEvent.from_mapping(
            payload,
            provider=self.name,
            model=self.model,
            prompt_version=prompt_version,
            source_reference=document.source_reference,
            source_text=document.text,
        )
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            event=event,
            usage=baseline.usage,
            raw_response=None,
        )


async def _extract(
    input_path: Path,
    *,
    limit: int | None,
    provider: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider = provider or LocalRuleProvider()
    labeled: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for line_number, row in iter_jsonl(input_path):
        if limit is not None and len(labeled) >= limit:
            break
        try:
            document = document_from_row(row)
            response = await provider.enrich(document, prompt_version="offline-rule-teacher-v1")
            labeled.append(
                {
                    "document": document.model_dump(),
                    "teacher_event": response.event.model_dump(),
                    "teacher_provider": provider.name,
                    "teacher_model": provider.model,
                }
            )
        except Exception as exc:
            rejected.append({"line_number": line_number, "error": type(exc).__name__, "message": str(exc)})
    return labeled, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate safe offline structured teacher labels from archived raw news.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Raw news JSONL/JSONL.GZ, a verified daily ZIP, or a directory containing those files.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Teacher-label JSONL to create.")
    parser.add_argument("--rejects", type=Path, help="Optional JSONL report for invalid input rows.")
    parser.add_argument("--limit", type=int, help="Maximum source rows to process.")
    parser.add_argument(
        "--teacher-mode",
        choices=("rule", "hf"),
        default="rule",
        help="Use deterministic rules or a laptop-only Hugging Face sentiment teacher.",
    )
    parser.add_argument(
        "--teacher-model",
        default="ProsusAI/finbert",
        help="Hugging Face model ID/path used only with --teacher-mode hf.",
    )
    parser.add_argument("--teacher-revision", help="Optional immutable model revision/commit.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse model downloads and load the heavy teacher from the local cache/path only.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing output files.")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input does not exist: {args.input}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    ensure_writable_output(args.output, overwrite=args.overwrite)
    if args.rejects:
        ensure_writable_output(args.rejects, overwrite=args.overwrite)
    provider = (
        HeavyLocalTeacher(
            args.teacher_model,
            revision=args.teacher_revision,
            local_files_only=args.local_files_only,
        )
        if args.teacher_mode == "hf"
        else LocalRuleProvider()
    )
    labeled, rejected = asyncio.run(_extract(args.input, limit=args.limit, provider=provider))
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.rejects:
        with args.rejects.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "teacher": provider.name,
                "teacher_model": provider.model,
                "teacher_revision": args.teacher_revision,
                "input": str(args.input),
                "output": str(args.output),
                "labeled_rows": len(labeled),
                "rejected_rows": len(rejected),
                "next_step": "Validate this file before building a student dataset.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
