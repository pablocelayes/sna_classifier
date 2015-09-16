from utils import json_load_unicode

def load_timeline_json(user_id):
    timeline_file = "timelines/%s.json" % user_id
    timeline = json_load_unicode(timeline_file)

    return timeline

def count_shared_tweets(user_id_1, user_id_2):
    pass