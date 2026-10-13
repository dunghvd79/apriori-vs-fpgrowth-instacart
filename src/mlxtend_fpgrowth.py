"""mlxtend-backed FP-Growth with the project's common estimator interface."""

from typing import Iterable


class MlxtendFPGrowth:
    def __init__(self, min_support: float = 0.01):
        if not 0 < min_support <= 1:
            raise ValueError("min_support must be in the interval (0, 1].")
        self.min_support = float(min_support)
        self.num_transactions = 0
        self.frequent_itemsets = {}

    def fit(self, transactions: Iterable[Iterable[str]]):
        try:
            import pandas as pd
            from mlxtend.frequent_patterns import fpgrowth
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError as exc:
            raise ImportError("Install dependencies with `pip install -r requirements.txt` to use MlxtendFPGrowth.") from exc

        baskets = [sorted(set(tx)) for tx in transactions]
        self.num_transactions = len(baskets)
        self.frequent_itemsets = {}
        if not baskets:
            return self

        encoder = TransactionEncoder()
        encoded = encoder.fit(baskets).transform(baskets, sparse=True)
        one_hot = pd.DataFrame.sparse.from_spmatrix(encoded, columns=encoder.columns_)
        result = fpgrowth(one_hot, min_support=self.min_support, use_colnames=True)
        self.frequent_itemsets = {
            frozenset(row.itemsets): float(row.support)
            for row in result.itertuples(index=False)
        }
        return self

    def get_frequent_itemsets(self):
        rows = [
            {'itemsets': tuple(sorted(itemset)), 'support': support, 'length': len(itemset)}
            for itemset, support in self.frequent_itemsets.items()
        ]
        return sorted(rows, key=lambda row: (-row['support'], row['itemsets']))
