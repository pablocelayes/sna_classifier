from tw_dataset.dbmodels import *

DATE_LOWER_LIMIT = datetime(year=2015, month=7, day=24)

DATE_UPPER_LIMIT = datetime(year=2015, month=8, day=24)

SQLITE_CONNECTION = 'sqlite:///twitter_sample2.db'

def open_session():
    engine = create_engine(SQLITE_CONNECTION)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session

if __name__ == '__main__':
    initialize_db()
    
    import networkx as nx
    graph = nx.read_gpickle('subgraph.gpickle')
    user_ids = graph.nodes()

    s = open_session()

    visited_user_ids = [u.id for u in s.query(User).all() if u and len(u.timeline)]
    print "%d visited users" % len(visited_user_ids)

    unvisited_users = [s.query(User).get(uid) for uid in user_ids if uid not in visited_user_ids]
    unvisited_users = [u for u in unvisited_users if u]    
    print "%d unvisited users pending" % len(unvisited_users)
    
    for user in unvisited_users:
        user.fetch_timeline(s)
        user.fetch_favorites(s)

    s.close()
    
