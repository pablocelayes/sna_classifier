#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import json
from itertools import chain

def json_dump_unicode(data, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def json_load_unicode(file_path):
    with codecs.open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)    


def concatenate(lists):
    return list(chain(*lists))