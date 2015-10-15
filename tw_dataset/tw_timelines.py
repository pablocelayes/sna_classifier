#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dbmodels import db_connect, sessionmaker

DB_ENGINE = db_connect()
DB_SESSION = sessionmaker(DB_ENGINE)

def fetch_timelines(user_ids):


if __name__ == '__main__':
    main()