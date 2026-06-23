from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RssFeedItem:
    source: str
    title: str
    url: str
    published_at: datetime
    raw_text: str
    raw_payload: dict[str, Any]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_child(element: ElementTree.Element, names: set[str]) -> ElementTree.Element | None:
    for child in list(element):
        if _local_name(child.tag) in names:
            return child
    return None


def _first_text(element: ElementTree.Element, names: set[str]) -> str | None:
    child = _first_child(element, names)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _link_text(element: ElementTree.Element) -> str | None:
    link = _first_child(element, {"link"})
    if link is None:
        return None
    href = link.attrib.get("href")
    if href:
        return href.strip()
    return link.text.strip() if link.text else None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        logger.debug("Could not parse RSS timestamp: %s", value)
        return None


def _domain(value: str) -> str:
    parsed = urlparse(value)
    return parsed.netloc or parsed.path or "rss"


def parse_rss_feed(xml_text: str, feed_url: str) -> list[RssFeedItem]:
    root = ElementTree.fromstring(xml_text)
    channel = next((child for child in root.iter() if _local_name(child.tag) == "channel"), None)
    feed_source = _first_text(channel, {"title"}) if channel is not None else _first_text(root, {"title"})
    source = feed_source or _domain(feed_url)
    entries = [entry for entry in root.iter() if _local_name(entry.tag) == "item"]
    if not entries:
        entries = [entry for entry in root.iter() if _local_name(entry.tag) == "entry"]

    items: list[RssFeedItem] = []
    for entry in entries:
        title = _first_text(entry, {"title"}) or ""
        url = _link_text(entry) or _first_text(entry, {"guid", "id"}) or ""
        if not title or not url:
            continue
        published_raw = _first_text(entry, {"pubdate", "published", "updated", "date"})
        published_at = _parse_datetime(published_raw) or datetime.now(timezone.utc)
        summary = _first_text(entry, {"description", "summary", "content", "encoded"}) or ""
        item_source = _first_text(entry, {"source"}) or source
        raw_text = " ".join(part for part in [title, summary, item_source] if part)
        items.append(
            RssFeedItem(
                source=item_source[:128],
                title=title,
                url=url,
                published_at=published_at,
                raw_text=raw_text,
                raw_payload={
                    "provider": "rss",
                    "feed_url": feed_url,
                    "source": item_source,
                    "title": title,
                    "url": url,
                    "published": published_raw,
                    "summary": summary,
                },
            )
        )
    return items


class RssNewsCollector:
    def __init__(self) -> None:
        self.last_errors: dict[str, str] = {}

    async def fetch(self, client: httpx.AsyncClient, feeds: list[str]) -> list[RssFeedItem]:
        self.last_errors = {}
        articles: list[RssFeedItem] = []
        for feed_url in feeds:
            try:
                if feed_url.startswith("file://"):
                    xml_text = Path(feed_url.removeprefix("file://")).read_text(encoding="utf-8")
                else:
                    response = await client.get(feed_url, follow_redirects=True)
                    response.raise_for_status()
                    xml_text = response.text
                articles.extend(parse_rss_feed(xml_text, feed_url))
            except Exception as exc:
                logger.exception("RSS feed failed: %s", feed_url)
                self.last_errors[feed_url] = str(exc).strip() or type(exc).__name__
        return articles
