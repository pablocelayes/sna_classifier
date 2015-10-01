import networkx as nx
from dbmodels import initialize_db, User, open_session

if __name__ == '__main__':
    initialize_db()
    
    graph = nx.read_gpickle('subgraph.gpickle')
    user_ids = graph.nodes()
    users = [User(id=int(uid)) for uid in user_ids]
    
    session = open_session()
    session.add_all(users)
    session.close()

    for user in users:
        user.fetch_timeline(session)
        user.fetch_favorites(session)