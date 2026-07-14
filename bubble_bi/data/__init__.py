"""Everything to do with getting and describing the data.

    download(settings)               -> the raw prices, one table
    add_features(prices, settings)   -> the same table, re-described 22 ways
    find_leaks(prices, settings)     -> proof that nothing looks into the future
"""

from bubble_bi.data.prices import download
from bubble_bi.data.features import FAMILIES, add_features, by_family, names
from bubble_bi.data.leakage import find_leaks
from bubble_bi.data.sentences import Sentences, make_sentences
from bubble_bi.data.tensors import Batches, make_tensors

__all__ = ["download", "add_features", "by_family", "names", "FAMILIES", "find_leaks",
           "make_tensors", "Batches",
           "make_sentences", "Sentences"]
