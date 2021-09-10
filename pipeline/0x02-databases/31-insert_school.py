#!/usr/bin/env python3
""" inserts a new document in a collection based on kwargs """


def insert_school(mongo_collection, **kwargs):
    """ Inserts a new document in a collection based on kwargs """
    id = mongo_collection.insert_one(kwargs).inserted_id
    return id
