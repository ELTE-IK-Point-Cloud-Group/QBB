"""
Created by Márton Ambrus-Dobai;
"""

import json

class Endpoints:
    def __init__(self, descriptor_folder):
        self.path = descriptor_folder+'/all_endpoints.json'
        self.loaded = False
        self.data = {}

    def try_load(self) -> bool:
        try:
            with open(self.path,'r') as f:
                self.data = json.load(f)
                self.loaded = True
                return True
        except:
            return False

    def is_loaded(self) -> bool:
        return self.loaded

    def has_key(self, key) -> bool:
        return str(key) in self.data.keys()

    def get(self, key) -> list:
        return self.data[str(key)]['endpoints'][self.data[str(key)]['level']-1]

    def get(self, key, index = None) -> list:
        if index:
            return self.data[str(key)]['endpoints'][index]
        return self.data[str(key)]['endpoints'][self.data[str(key)]['level']-1]

    def get_level(self, key):
        return self.data[str(key)]['level']

    def add(self, key, level, endpoints):
        self.data[str(key)] = {
            'endpoints' : endpoints,
            'level' : level
        }
        with open(self.path, 'w', encoding ='utf8') as f:
            json.dump(self.data, f, indent=2)