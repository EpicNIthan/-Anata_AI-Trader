from __future__ import annotations

from typing import Any

HIGH_QUALITY_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "BCHUSDT",
    "XLMUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "SUIUSDT",
    "ATOMUSDT",
    "FILUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "AAVEUSDT",
    "ICPUSDT",
    "SEIUSDT",
    "RENDERUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
]

SYMBOL_IDENTITY_COLUMNS = [f"symbol_is_{symbol}" for symbol in HIGH_QUALITY_SYMBOLS]
SYMBOL_GROUP_COLUMNS = [
    "symbol_group_major",
    "symbol_group_layer1",
    "symbol_group_layer2",
    "symbol_group_defi",
    "symbol_group_oracle_infra",
    "symbol_group_legacy",
    "symbol_group_meme",
    "symbol_group_ai_infra",
]

SYMBOL_FEATURE_COLUMNS = [*SYMBOL_IDENTITY_COLUMNS, *SYMBOL_GROUP_COLUMNS]

_GROUPS = {
    "major": {"BTCUSDT", "ETHUSDT", "BNBUSDT"},
    "layer1": {
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "TRXUSDT",
        "XLMUSDT",
        "NEARUSDT",
        "APTUSDT",
        "SUIUSDT",
        "ATOMUSDT",
        "ICPUSDT",
        "SEIUSDT",
    },
    "layer2": {"ARBUSDT", "OPUSDT"},
    "defi": {"UNIUSDT", "AAVEUSDT", "INJUSDT"},
    "oracle_infra": {"LINKUSDT", "FILUSDT", "RENDERUSDT"},
    "legacy": {"LTCUSDT", "BCHUSDT", "ETCUSDT"},
    "meme": {"DOGEUSDT", "SHIBUSDT", "PEPEUSDT"},
    "ai_infra": {"RENDERUSDT", "ICPUSDT", "FETUSDT"},
}


def normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def symbol_identity_values(symbol: Any) -> dict[str, float]:
    normalized = normalize_symbol(symbol)
    values = {column: 0.0 for column in SYMBOL_FEATURE_COLUMNS}
    identity_column = f"symbol_is_{normalized}"
    if identity_column in values:
        values[identity_column] = 1.0
    for group, symbols in _GROUPS.items():
        values[f"symbol_group_{group}"] = 1.0 if normalized in symbols else 0.0
    return values
