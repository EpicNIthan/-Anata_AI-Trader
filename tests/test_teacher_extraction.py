from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.intelligence.schemas import NewsDocument
from scripts.run_teacher_extraction import HeavyLocalTeacher


def test_heavy_local_teacher_merges_numeric_sentiment_into_structured_schema() -> None:
    calls: list[tuple[str, bool]] = []

    def classifier(text: str, *, truncation: bool) -> list[dict[str, object]]:
        calls.append((text, truncation))
        return [{"label": "negative", "score": 0.91}]

    teacher = HeavyLocalTeacher(
        "local-finbert-test",
        revision="immutable-test-revision",
        classifier=classifier,
    )
    document = NewsDocument(
        title="Exchange reports security breach",
        content="A crypto exchange reported a hack affecting Bitcoin reserves.",
        source="fixture-wire",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        available_to_model_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    response = asyncio.run(teacher.enrich(document, prompt_version="teacher-test-v1"))

    assert calls and calls[0][1] is True
    assert response.provider == "local_hf_teacher"
    assert response.model == "local-finbert-test"
    assert response.event.sentiment == -0.91
    assert response.event.confidence == 0.91
    assert response.event.direction.value == "bearish"
    assert response.event.event_type.value == "security_incident"
    assert response.event.metadata["teacher_revision"] == "immutable-test-revision"


def test_news_document_cleans_html_before_hashing_and_records_missingness() -> None:
    observed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    html_document = NewsDocument(
        title=" <b>Bitcoin &amp; markets</b> ",
        content="<p>Price rises.</p><script>steal()</script>",
        source="fixture-wire",
        received_at=observed,
        available_to_model_at=observed,
    )
    plain_document = NewsDocument(
        title="Bitcoin & markets",
        content="Price rises.",
        source="fixture-wire",
        received_at=observed,
        available_to_model_at=observed,
    )

    assert html_document.title == "Bitcoin & markets"
    assert html_document.content == "Price rises."
    assert html_document.content_hash == plain_document.content_hash
    assert html_document.metadata["text_cleanup_applied"] is True
    assert "published_at" in html_document.metadata["missing_fields"]
    assert "relevant_assets" in html_document.metadata["missing_fields"]
