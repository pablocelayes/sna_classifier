import graph_tool.all as gt
import json
from dbmodels import *

g = gt.load_graph('graph_myfollowed.graphml')

s=open_session()

rtcounts = {u.id: len([x for x in u.retweets if x.lang == 'es' and x.author_id != u.id]) for u in s.query(User).all()}

def get_twitter_id(g, v):
    return g.vertex_properties['_graphml_vertex_id'][v]

neighborhood_sizes = {}

for v in g.vertices():
    neighbors = set(v.out_neighbours())
    for n in v.out_neighbours():
        neighbors.update(n.out_neighbours())
    neighborhood_sizes[int(get_twitter_id(g,v))] = len(neighbors)

active_with_neighbors = [x for x in rtcounts if rtcounts[x] > 100 and neighborhood_sizes[x] > 100 if x != USER_DATA['id']]

with open('active_with_neighbours.json','w') as f:
    json.dump(active_with_neighbors, f)
