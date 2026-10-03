"""
Module: my_fpgrowth.py
Mục đích: Tự cài đặt thuật toán FP-Growth (Frequent Pattern Growth) từ đầu.
Nguyên lý:
- Nén toàn bộ tập giao dịch vào cấu trúc cây FP-Tree (Frequent Pattern Tree).
- Dùng phương pháp phân chia để trị (Divide and Conquer): khai phá các cơ sở mẫu có điều kiện (conditional pattern base) 
  và xây dựng các cây FP-Tree có điều kiện đệ quy mà KHÔNG cần sinh ứng viên như Apriori.
"""

import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class FPNode:
    """Đại diện cho một nút trong cây FP-Tree."""
    def __init__(self, item: str, count: int = 1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}  # {item: FPNode}
        self.node_link = None  # Con trỏ liên kết đến nút cùng loại tiếp theo (cho Header Table)

    def increment(self, count: int = 1):
        self.count += count


class MyFPGrowth:
    def __init__(self, min_support: float = 0.01):
        """
        Khởi tạo thuật toán FP-Growth.

        Tham số:
            min_support: Ngưỡng hỗ trợ tối thiểu (0.0 < min_support <= 1.0).
        """
        self.min_support = min_support
        self.frequent_itemsets = {}  # {frozenset: support_val}
        self.num_transactions = 0

    def fit(self, transactions: List[List[str]]):
        """
        Huấn luyện thuật toán FP-Growth trên tập danh sách giao dịch.
        """
        self.num_transactions = len(transactions)
        if self.num_transactions == 0:
            return self

        min_count = self.min_support * self.num_transactions

        # Bước 1: Quét lần 1 - Đếm tần suất các item kích thước 1
        item_counts = defaultdict(int)
        for tx in transactions:
            for item in tx:
                item_counts[item] += 1

        # Lọc các item thỏa mãn min_support
        header_table = {item: [count, None] for item, count in item_counts.items() if count >= min_count}
        if not header_table:
            return self

        # Bước 2: Quét lần 2 - Xây dựng cây FP-Tree gốc
        root = FPNode('root', count=1, parent=None)

        for tx in transactions:
            # Lọc các item phổ biến trong giao dịch và sắp xếp giảm dần theo tần suất
            filtered_tx = [item for item in tx if item in header_table]
            filtered_tx.sort(key=lambda item: (header_table[item][0], item), reverse=True)

            if filtered_tx:
                self._insert_tree(filtered_tx, root, header_table)

        # Bước 3: Khai phá cây FP-Tree đệ quy
        self.frequent_itemsets = {}
        self._mine_tree(header_table, min_count, set(), self.frequent_itemsets)

        # Chuyển đổi support_count thành support tỷ lệ
        for itemset, count in list(self.frequent_itemsets.items()):
            self.frequent_itemsets[itemset] = count / self.num_transactions

        return self

    def _insert_tree(self, items: List[str], current_node: FPNode, header_table: Dict):
        """Chèn một chuỗi items đã sắp xếp vào cây FP-Tree."""
        first = items[0]
        if first in current_node.children:
            current_node.children[first].increment()
        else:
            new_node = FPNode(first, count=1, parent=current_node)
            current_node.children[first] = new_node

            # Cập nhật node_link trong header table
            if header_table[first][1] is None:
                header_table[first][1] = new_node
            else:
                current_link = header_table[first][1]
                while current_link.node_link is not None:
                    current_link = current_link.node_link
                current_link.node_link = new_node

        # Gọi đệ quy cho các phần tử còn lại
        if len(items) > 1:
            self._insert_tree(items[1:], current_node.children[first], header_table)

    def _mine_tree(self, header_table: Dict, min_count: float, prefix: set, frequent_itemsets: Dict):
        """Khai phá các mẫu phổ biến từ Header Table và Conditional FP-Tree."""
        # Sắp xếp các item theo tần suất tăng dần (khai phá từ dưới lên)
        sorted_items = sorted(header_table.keys(), key=lambda x: header_table[x][0])

        for item in sorted_items:
            new_prefix = prefix.copy()
            new_prefix.add(item)
            support_count = header_table[item][0]
            frequent_itemsets[frozenset(new_prefix)] = support_count

            # Tìm conditional pattern base cho item
            cond_patterns = self._find_conditional_patterns(header_table[item][1])

            # Xây dựng conditional FP-Tree
            cond_header = {}
            for pattern, count in cond_patterns.items():
                for pat_item in pattern:
                    cond_header[pat_item] = cond_header.get(pat_item, 0) + count

            # Lọc theo min_count
            cond_header = {k: [v, None] for k, v in cond_header.items() if v >= min_count}

            # Dựng conditional tree nếu header không rỗng
            if cond_header:
                cond_root = FPNode('root', count=1, parent=None)
                for pattern, count in cond_patterns.items():
                    filtered_pattern = [p for p in pattern if p in cond_header]
                    filtered_pattern.sort(key=lambda p: (cond_header[p][0], p), reverse=True)
                    if filtered_pattern:
                        self._insert_cond_tree(filtered_pattern, count, cond_root, cond_header)

                # Đệ quy khai phá trên conditional tree
                self._mine_tree(cond_header, min_count, new_prefix, frequent_itemsets)

    def _insert_cond_tree(self, items: List[str], count: int, current_node: FPNode, header_table: Dict):
        """Chèn pattern vào conditional FP-Tree có kèm trọng số count."""
        first = items[0]
        if first in current_node.children:
            current_node.children[first].increment(count)
        else:
            new_node = FPNode(first, count=count, parent=current_node)
            current_node.children[first] = new_node

            if header_table[first][1] is None:
                header_table[first][1] = new_node
            else:
                current_link = header_table[first][1]
                while current_link.node_link is not None:
                    current_link = current_link.node_link
                current_link.node_link = new_node

        if len(items) > 1:
            self._insert_cond_tree(items[1:], count, current_node.children[first], header_table)

    def _find_conditional_patterns(self, node: Optional[FPNode]) -> Dict[Tuple[str, ...], int]:
        """Truy ngược từ lá lên gốc để tìm đường dẫn tiền tố (prefix paths)."""
        cond_patterns = {}
        while node is not None:
            prefix_path = []
            parent = node.parent
            while parent is not None and parent.item != 'root':
                prefix_path.append(parent.item)
                parent = parent.parent

            if prefix_path:
                cond_patterns[tuple(prefix_path)] = node.count
            node = node.node_link
        return cond_patterns

    def get_frequent_itemsets(self):
        """Trả về danh sách các tập phổ biến."""
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

    fpg = MyFPGrowth(min_support=0.4)
    fpg.fit(sample_data)
    items = fpg.get_frequent_itemsets()
    print("Test FP-Growth với dữ liệu đồ chơi:")
    for item in items:
        print(f" - {item['itemsets']}: support = {item['support']:.2f}")
