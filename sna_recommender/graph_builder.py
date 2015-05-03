#!/usr/local/bin python
# -*- coding: utf-8 -*-
__author__ = 'Pablo Celayes'
import networkx as nx


"""
    Implementation of different methods
    for building a social graph of users
    out of their article feedback history
"""

class SocialGraphBuilder(object):
    """docstring for SocialGraphBuilder"""
    def __init__(self):
        pass        


class CommonReadsSGBuilder(SocialGraphBuilder):
    """ Connects two users if their number of articles
        read in common is above a certain threshold.

        By default, we use the average number of
        common articles as threshold
    """
    def __init__(self, dataset, threshold=None):
        super(CommonReadsSGBuilder, self).__init__()
        """
            Parameters
            ----------
            dataset: dict
                A dictionary with keys = user_ids
                                values = lists of read urls

            threshold: int
                minimum number of common read urls to create an edge
                if None, the average number of common reads will be used
        """
        user_ids = dataset.keys()
        cr_graph = nx.Graph()
        for i, user_i in enumerate(user_ids):
            for user_j in user_ids[i + 1:]:
                read_i = dataset[user_i]
                read_j = dataset[user_j]
                common = [url for url in read_i if url in read_j]
                weight = len(common)
                if threshold:
                    weight = 1 if weight >= threshold else 0
                if weight:
                    cr_graph.add_edge(user_i, user_j, weight=weight)

        if not threshold:
            # TODO: calculate average common reads
            # and apply threhsold
            raise NotImplementedError

        self.graph = cr_graph



class MutualModelFittingSGBuilder(SocialGraphBuilder):
    """ Connects two users if the model of each
        of them fits well to the reading history
        of the other
    """
    def __init__(self, arg):
        super(CommonReadsSGBuilder, self).__init__()
        self.arg = arg

if __name__ == '__main__':
    from datasets import TOY_DATASET, load_facebook_dataset
    socialgraph = CommonReadsSGBuilder(dataset=load_facebook_dataset(), threshold=1)
    import matplotlib.pyplot as plt
    nx.draw(socialgraph.graph)
    plt.show()