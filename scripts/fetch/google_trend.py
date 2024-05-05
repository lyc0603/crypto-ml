"""
Script to fetch google trend data
"""

from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["blockchain"]
pytrends.build_payload(kw_list, cat=0, timeframe="2018-01-01 2020-01-01", geo='', gprop='')
pytrends.interest_over_time()

