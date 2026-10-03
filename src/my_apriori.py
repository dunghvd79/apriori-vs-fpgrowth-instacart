"""
Module: my_apriori.py
Mục đích: Tự cài đặt thuật toán Apriori từ đầu (from scratch) để khai phá tập mục phổ biến (Frequent Itemsets).
Nguyên lý: 
- Tiên đề Apriori (Apriori Property): Mọi tập con của một tập phổ biến đều phải là tập phổ biến.
- Sinh ứng viên C_k từ L_{k-1} (Join step & Prune step) và lọc theo min_support qua từng lượt quét dữ liệu.
"""

import sys
import time
from typing import List, Set, Dict, Tuple
from collections import defaultdict

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class MyApriori:
    def __init__(self, min_support: float = 0.01):
        """
        Khởi tạo thuật toán Apriori.

        Tham số:
            min_support: Ngưỡng hỗ trợ tối thiểu (0.0 < min_support <= 1.0).
        """
        self.min_support = min_support
        self.itemsets_by_length = {}  # {length_k: {frozenset: support_count}}
        self.frequent_itemsets = {}   # {frozenset: support_val}
        self.num_transactions = 0

    def fit(self, transactions: List[List[str]]):
        """
        Huấn luyện thuật toán Apriori trên danh sách giao dịch.

        Tham số:
            transactions: Danh sách các giao dịch, mỗi giao dịch là một list các item (tên sản phẩm hoặc ID).
        
        Trả về:
            DataFrame hoặc Dict chứa các tập phổ biến và chỉ số support tương ứng.
        """
        self.num_transactions = len(transactions)
        if self.num_transactions == 0:
            return self

        min_count = self.min_support * self.num_transactions

        # Bước 1: Tìm L1 (Tập phổ biến kích thước 1)
        item_counts = defaultdict(int)
        for tx in transactions:
            for item in tx:
                item_counts[frozenset([item])] += 1

        l1 = {itemset: count for itemset, count in item_counts.items() if count >= min_count}
        self.itemsets_by_length[1] = l1
        k = 2
        current_l = l1

        # Bước lặp sinh C_k và tìm L_k (k >= 2)
        while current_l:
            candidates = self._generate_candidates(list(current_l.keys()), k)
            if not candidates:
                break

            # Đếm support cho các ứng viên
            candidate_counts = defaultdict(int)
            for tx in transactions:
                tx_set = set(tx)
                for cand in candidates:
                    if cand.issubset(tx_set):
                        candidate_counts[cand] += 1

            # Lọc theo min_support
            current_l = {cand: count for cand, count in candidate_counts.items() if count >= min_count}
            if current_l:
                self.itemsets_by_length[k] = current_l
                k += 1
            else:
                break

        # Tổng hợp toàn bộ tập phổ biến
        self.frequent_itemsets = {}
        for length, itemsets in self.itemsets_by_length.items():
            for itemset, count in itemsets.items():
                self.frequent_itemsets[itemset] = count / self.num_transactions

        return self

    def _generate_candidates(self, prev_itemsets: List[frozenset], k: int) -> Set[frozenset]:
        """
        Sinh tập ứng viên C_k từ L_{k-1} sử dụng phép kết nối (Join) và tỉa (Prune).
        """
        candidates = set()
        n = len(prev_itemsets)
        prev_list = [sorted(list(x)) for x in prev_itemsets]

        # Join step
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = prev_list[i]
                itemset2 = prev_list[j]
                # Hai tập hợp kích thước k-1 ghép được khi k-2 phần tử đầu giống nhau
                if itemset1[:k - 2] == itemset2[:k - 2]:
                    cand = frozenset(itemset1).union(frozenset(itemset2))
                    if len(cand) == k:
                        # Prune step: Kiểm tra mọi tập con kích thước k-1 có nằm trong prev_itemsets không
                        if self._all_subsets_frequent(cand, prev_itemsets, k - 1):
                            candidates.add(cand)
        return candidates

    def _all_subsets_frequent(self, candidate: frozenset, prev_itemsets: List[frozenset], k_minus_1: int) -> bool:
        """Kiểm tra điều kiện tiên đề Apriori (Prune step)."""
        prev_set = set(prev_itemsets)
        cand_list = list(candidate)
        for i in range(len(cand_list)):
            subset = frozenset(cand_list[:i] + cand_list[i + 1:])
            if subset not in prev_set:
                return False
        return True

    def get_frequent_itemsets(self):
        """Trả về danh sách các tập phổ biến dưới dạng danh sách từ điển."""
        results = []
        for itemset, support in self.frequent_itemsets.items():
            results.append({
                'itemsets': tuple(sorted(list(itemset))),
                'support': support,
                'length': len(itemset)
            })
        return sorted(results, key=lambda x: x['support'], reverse=True)


if __name__ == '__main__':
    # Kiểm thử nhanh với dữ liệu mẫu nhỏ
    sample_data = [
        ['Milk', 'Bread', 'Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread'],
        ['Milk', 'Eggs'],
        ['Bread', 'Butter', 'Eggs']
    ]

    ap = MyApriori(min_support=0.4)
    ap.fit(sample_data)
    items = ap.get_frequent_itemsets()
    print("Test Apriori với dữ liệu đồ chơi:")
    for item in items:
        print(f" - {item['itemsets']}: support = {item['support']:.2f}")
