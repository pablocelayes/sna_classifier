# -*- coding: utf-8 -*-

"""
Copyright (C) Cogfor <info@cogfor.com>
Copyright (C) 2014 Alejandro Naser Pastoriza <alejandro@cogfor.com>
Document Titles Abstraction
"""

import logging
import codecs

LOGGER = logging.getLogger("esa.document_titles")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


class DocumentTitles(object):
    """
    Loads a list of document titles form a text file.
    Each line is considered to be a title.
    """
    def __init__(self):
        self.document_titles = []

    @classmethod
    def load(cls, file_path):
        """
        Import data from file on disk
        """
        LOGGER.info("Loading concept titles from %s", file_path)
        result = DocumentTitles()
        result.file_path = file_path
        with open(file_path, "r") as input_file:
            for line in input_file:
                doc_title = line.strip("\n").decode("UTF-8")
                result.append(doc_title)
        LOGGER.info("Loaded %d concept titles.", len(result.document_titles))
        return result

    def save(self, file_path=None):
        if file_path is None:
            file_path = self.file_path

        with codecs.open(file_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(self.document_titles))

    def append(self, title):
        """
        Append a title to the list of document titles
        """
        self.document_titles.append(title)

    def __iter__(self):
        for title in self.document_titles:
            yield title

    def __getitem__(self, key):
        return self.document_titles[key]

    def __len__(self):
        return len(self.document_titles)
