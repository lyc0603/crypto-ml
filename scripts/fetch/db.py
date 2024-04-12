"""
Script to initialize the mongodb
"""

import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
