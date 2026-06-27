from __future__ import annotations

from collect_coingecko_history import main
import collect_coingecko_history as collector

collector.DEFAULT_SYMBOL_COINS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "AVAXUSDT": "avalanche-2",
    "LINKUSDT": "chainlink",
    "LTCUSDT": "litecoin",
    "DOTUSDT": "polkadot",
    "TRXUSDT": "tron",
    "BCHUSDT": "bitcoin-cash",
    "XLMUSDT": "stellar",
    "NEARUSDT": "near",
    "APTUSDT": "aptos",
    "ARBUSDT": "arbitrum",
    "OPUSDT": "optimism",
    "INJUSDT": "injective-protocol",
    "SUIUSDT": "sui",
    "ATOMUSDT": "cosmos",
    "FILUSDT": "filecoin",
    "UNIUSDT": "uniswap",
    "ETCUSDT": "ethereum-classic",
    "AAVEUSDT": "aave",
    "ICPUSDT": "internet-computer",
    "SEIUSDT": "sei-network",
    "RENDERUSDT": "render-token",
    "SHIBUSDT": "shiba-inu",
    "PEPEUSDT": "pepe",
}

collector.DEFAULT_GDELT_QUERY = (
    "(bitcoin OR btc OR ethereum OR eth OR solana OR bnb OR xrp OR ripple OR cardano OR ada OR "
    "dogecoin OR doge OR avalanche OR chainlink OR litecoin OR polkadot OR tron OR stellar OR "
    "near OR aptos OR arbitrum OR optimism OR injective OR sui OR cosmos OR filecoin OR uniswap OR "
    "aave OR internet computer OR sei OR render OR shiba OR pepe OR crypto OR cryptocurrency OR "
    "stablecoin OR binance OR coinbase OR etf)"
)


if __name__ == "__main__":
    main()
