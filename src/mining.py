"""
Module: mining.py
Mục đích: Lớp mỏng bọc thư viện mlxtend để khai phá tập phổ biến bằng Apriori / FP-Growth.

Hai thuật toán dùng CÙNG đầu vào (ma trận one-hot thưa) và CÙNG tham số,
nên thời gian và bộ nhớ đo được so sánh công bằng với nhau.

Điểm quan trọng - cắt tỉa món hàng hiếm:
    mlxtend cần ma trận one-hot (giao dịch x món). Với 100k đơn x 35.400 món, ma trận
    đặc sẽ chiếm ~3,5 GB. Một món có support < min_support không thể nằm trong bất kỳ
    tập phổ biến nào, nên có thể loại bỏ TRƯỚC khi tạo ma trận mà không đổi kết quả.
"""

import sys
from pathlib import Path
from typing import List

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ALGORITHMS = ('apriori', 'fpgrowth')


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_transactions(parquet_path) -> List[List[str]]:
    """Đọc file parquet do preprocess.py tạo ra, trả về danh sách giỏ hàng."""
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=['items'])
    return [list(items) for items in df['items']]


def encode_transactions(transactions: List[List[str]], min_support: float):
    """
    Mã hóa giỏ hàng thành DataFrame one-hot THƯA (sparse), sau khi cắt tỉa món hiếm.

    Tham số:
        transactions: danh sách giỏ hàng.
        min_support: ngưỡng dùng để cắt tỉa (món có support thấp hơn bị loại).

    Trả về:
        DataFrame bool thưa, cột là tên món, dòng là giao dịch.
    """
    import pandas as pd
    from collections import Counter
    from mlxtend.preprocessing import TransactionEncoder

    n = len(transactions)
    counts = Counter(item for tx in transactions for item in tx)
    keep = {item for item, c in counts.items() if c / n >= min_support}
    pruned = [[item for item in tx if item in keep] for tx in transactions]

    te = TransactionEncoder()
    matrix = te.fit(pruned).transform(pruned, sparse=True)
    return pd.DataFrame.sparse.from_spmatrix(matrix, columns=te.columns_)


def mine_frequent_itemsets(algorithm: str, onehot_df, min_support: float, max_len=None):
    """
    Chạy một thuật toán của mlxtend.

    Trả về:
        DataFrame gồm cột 'support' (tỷ lệ) và 'itemsets' (frozenset).
    """
    from mlxtend.frequent_patterns import apriori, fpgrowth

    if algorithm == 'apriori':
        return apriori(onehot_df, min_support=min_support, use_colnames=True, max_len=max_len)
    if algorithm == 'fpgrowth':
        return fpgrowth(onehot_df, min_support=min_support, use_colnames=True, max_len=max_len)
    raise ValueError(f"Thuật toán không hợp lệ: {algorithm!r}. Chọn một trong {ALGORITHMS}.")


def mine_rules(frequent_itemsets, min_confidence: float = 0.2, min_lift: float = 1.5):
    """
    Sinh luật kết hợp từ tập phổ biến rồi lọc theo Confidence và Lift.

    Trả về:
        DataFrame luật, xếp giảm dần theo lift rồi confidence.
    """
    from mlxtend.frequent_patterns import association_rules

    if len(frequent_itemsets) == 0:
        return frequent_itemsets.iloc[0:0]

    rules = association_rules(
        frequent_itemsets, num_itemsets=len(frequent_itemsets),
        metric='confidence', min_threshold=min_confidence,
    )
    rules = rules[rules['lift'] >= min_lift]
    return rules.sort_values(['lift', 'confidence'], ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    # Thử nhanh trên mẫu 10k: hai thuật toán phải cho cùng tập phổ biến
    path = get_project_root() / 'data' / 'processed' / 'transactions_product_10k.parquet'
    if not path.exists():
        print(f"Chưa có {path.name}. Hãy chạy src/preprocess.py trước.")
        sys.exit(1)

    txs = load_transactions(path)
    supp = 0.01
    onehot = encode_transactions(txs, supp)
    print(f"{len(txs):,} giao dịch, còn {onehot.shape[1]} món sau khi cắt tỉa (min_support={supp}).")

    results = {}
    for algo in ALGORITHMS:
        fi = mine_frequent_itemsets(algo, onehot, supp)
        results[algo] = {(frozenset(s), round(v, 10)) for s, v in zip(fi['itemsets'], fi['support'])}
        print(f"  {algo:<9}: {len(fi)} tập phổ biến")
    print("Hai thuật toán cho kết quả giống nhau:", results['apriori'] == results['fpgrowth'])
