"""
Script to fetch the list of stablecoins
"""

from environ.fetch.coingecko import CoinGecko

cg = CoinGecko()
stablecoins = [_["id"] for _ in cg.market("stablecoins")]
