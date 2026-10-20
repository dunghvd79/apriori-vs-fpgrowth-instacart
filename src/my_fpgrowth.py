

from collections import defaultdict
from itertools import combinations
import time
from typing import Dict, FrozenSet, List, Set, Tuple
import pandas as pd


class FPNode:
    """Nút trong cây tiền tố FP-Tree, dùng __slots__ để tối ưu hóa bộ nhớ RAM."""
    __slots__ = ('item', 'count', 'parent', 'children', 'node_link')

    def __init__(self, item: int, count: int = 1, parent=None):
        self.item = item          # ID sản phẩm (số nguyên)
        self.count = count        # Tần suất xuất hiện
        self.parent = parent      # Con trỏ trỏ về nút cha
        self.children = {}        # Danh sách nút con: {item_id: child_node}
        self.node_link = None     # Con trỏ liên kết các nút cùng item_id trong cây


class OptimizedFPGrowth:
    """
    Thuật toán FP-Growth tự cài đặt, tối ưu hóa tốc độ và bộ nhớ.
    """

    def __init__(self, min_support: float = 0.01):
        if not 0 < min_support <= 1.0:
            raise ValueError("min_support phải nằm trong khoảng (0, 1].")
        self.min_support = float(min_support)
        self.item2id: Dict[str, int] = {}
        self.id2item: Dict[int, str] = {}
        self.frequent_itemsets: List[Dict] = []
        self.n_transactions = 0

    def fit(self, transactions: List[List[str]]):
        """
        Huấn luyện và khai phá tập phổ biến từ danh sách giỏ hàng.
        """
        self.n_transactions = len(transactions)
        if self.n_transactions == 0:
            return self

        min_count = max(1, int(self.min_support * self.n_transactions))

        # ======================================================================
        # BƯỚC 1: Quét CSDL lần 1 — Đếm tần suất và lọc các món đạt min_support
        # ======================================================================
        item_counts = defaultdict(int)
        for tx in transactions:
            for item in tx:
                item_counts[item] += 1

        # Lọc các món thỏa mãn min_support và sắp xếp theo tần suất giảm dần
        freq_items = sorted(
            [item for item, c in item_counts.items() if c >= min_count],
            key=lambda x: (-item_counts[x], x)
        )

        # Số hóa tên sản phẩm sang ID số nguyên
        self.item2id = {item: idx for idx, item in enumerate(freq_items)}
        self.id2item = {idx: item for idx, item in enumerate(freq_items)}
        rank = self.item2id

        # ======================================================================
        # BƯỚC 2: Quét CSDL lần 2 — Xây dựng cây tiền tố FP-Tree & Header Table
        # ======================================================================
        root = FPNode(-1, 0, None)
        # Header Table: {item_id: [tổng_tần_suất, nút_đầu_tiên]}
        header_table = {idx: [item_counts[item], None] for idx, item in self.id2item.items()}

        for tx in transactions:
            # Lọc chỉ giữ lại các món phổ biến và sắp xếp theo thứ tự tần suất giảm dần
            filtered = [rank[item] for item in tx if item in rank]
            if not filtered:
                continue
            filtered.sort()

            # Chèn giao dịch vào cây FP-Tree
            curr = root
            for item_id in filtered:
                if item_id in curr.children:
                    curr.children[item_id].count += 1
                else:
                    new_node = FPNode(item_id, 1, curr)
                    curr.children[item_id] = new_node
                    # Nối vào danh sách liên kết Header Table
                    new_node.node_link = header_table[item_id][1]
                    header_table[item_id][1] = new_node
                curr = curr.children[item_id]

        # ======================================================================
        # BƯỚC 3: Khai phá đệ quy cây FP-Tree (Divide-and-Conquer)
        # ======================================================================
        self.frequent_itemsets = []
        self._mine_tree(header_table, min_count, [], self.n_transactions)
        return self

    def _mine_tree(self, header_table: dict, min_count: int, prefix: list, n_tx: int):
        """Khai phá đệ quy các cây điều kiện (Conditional FP-Tree) từ dưới lên."""
        # Duyệt từ món có tần suất thấp nhất đến cao nhất trong Header Table
        for item_id in sorted(header_table.keys(), reverse=True):
            support_count = header_table[item_id][0]
            new_prefix = prefix + [item_id]

            self.frequent_itemsets.append({
                'itemsets': frozenset(self.id2item[i] for i in new_prefix),
                'support': support_count / n_tx,
                'count': support_count
            })

            # Tìm Conditional Pattern Base (các nhánh tiền đề dẫn tới item_id)
            cond_patterns = []
            node = header_table[item_id][1]
            while node:
                path = []
                p = node.parent
                while p and p.item != -1:
                    path.append(p.item)
                    p = p.parent
                if path:
                    cond_patterns.append((path, node.count))
                node = node.node_link

            # Đếm tần suất các mục trong Conditional Pattern Base
            cond_counts = defaultdict(int)
            for path, count in cond_patterns:
                for p_item in path:
                    cond_counts[p_item] += count

            # Lọc các mục thỏa mãn min_count trong cây điều kiện
            cond_header = {p_item: [cond_counts[p_item], None] for p_item, c in cond_counts.items() if c >= min_count}
            if not cond_header:
                continue

            # Xây dựng Conditional FP-Tree
            cond_root = FPNode(-1, 0, None)
            for path, count in cond_patterns:
                filtered_path = [p_item for p_item in path if p_item in cond_header]
                if not filtered_path:
                    continue
                filtered_path.sort()

                curr = cond_root
                for p_item in filtered_path:
                    if p_item in curr.children:
                        curr.children[p_item].count += count
                    else:
                        new_node = FPNode(p_item, count, curr)
                        curr.children[p_item] = new_node
                        new_node.node_link = cond_header[p_item][1]
                        cond_header[p_item][1] = new_node
                    curr = curr.children[p_item]

            # Đệ quy khai phá tiếp cây điều kiện
            self._mine_tree(cond_header, min_count, new_prefix, n_tx)

    def get_frequent_itemsets(self) -> pd.DataFrame:
        """Trả về DataFrame các tập phổ biến, sắp xếp giảm dần theo support."""
        if not self.frequent_itemsets:
            return pd.DataFrame(columns=['itemsets', 'support', 'count', 'length'])
        df = pd.DataFrame(self.frequent_itemsets)
        df['length'] = df['itemsets'].apply(len)
        return df.sort_values(by=['support', 'length'], ascending=[False, False]).reset_index(drop=True)


def generate_rules(frequent_itemsets_df: pd.DataFrame, min_confidence: float = 0.2, min_lift: float = 1.2) -> pd.DataFrame:
    """
    Trích xuất luật kết hợp (Association Rules: A -> B) từ bảng tập phổ biến.

    Chỉ số:
        Support(A -> B) = Support(A ∪ B)
        Confidence(A -> B) = Support(A ∪ B) / Support(A)
        Lift(A -> B) = Confidence(A -> B) / Support(B)
    """
    if frequent_itemsets_df.empty:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

    # Tạo bảng tra cứu Support theo O(1)
    supp_map = {row.itemsets: float(row.support) for row in frequent_itemsets_df.itertuples(index=False)}
    rules = []

    # Chỉ xét các tập phổ biến có từ 2 món trở lên
    candidates = frequent_itemsets_df[frequent_itemsets_df['itemsets'].apply(len) >= 2]

    for row in candidates.itertuples(index=False):
        itemset = row.itemsets
        supp_itemset = float(row.support)
        items_list = list(itemset)
        k = len(items_list)

        # Thử tất cả các tập con tiền đề (Antecedents) có độ dài từ 1 đến k-1
        for r in range(1, k):
            for ante_tuple in combinations(items_list, r):
                ante = frozenset(ante_tuple)
                conseq = itemset - ante

                ante_supp = supp_map.get(ante, 0.0)
                conseq_supp = supp_map.get(conseq, 0.0)
                if ante_supp == 0 or conseq_supp == 0:
                    continue

                conf = supp_itemset / ante_supp
                if conf < min_confidence:
                    continue

                lift = conf / conseq_supp
                if lift < min_lift:
                    continue

                rules.append({
                    'antecedents': ante,
                    'consequents': conseq,
                    'support': supp_itemset,
                    'confidence': conf,
                    'lift': lift
                })

    if not rules:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

    df_rules = pd.DataFrame(rules)
    return df_rules.sort_values(by=['lift', 'confidence'], ascending=[False, False]).reset_index(drop=True)


def recommend_products(cart_items: list, rules_df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
    """
    Dự đoán và gợi ý các sản phẩm mua kèm dựa trên giỏ hàng hiện tại của khách hàng.

    Thuật toán:
    1. Lọc các luật có tiền đề là tập con của giỏ hàng (Antecedents ⊆ Cart).
    2. Lấy các sản phẩm hệ quả chưa có trong giỏ hàng (Consequents \\ Cart).
    3. Xếp hạng sản phẩm theo Score = Lift * Confidence.
    4. Trả về danh sách gợi ý kèm giải thích nghiệp vụ tự nhiên.
    """
    if not cart_items or rules_df.empty:
        return []

    cart_set = set(cart_items)
    candidate_map = {}

    for row in rules_df.itertuples(index=False):
        ante = row.antecedents
        conseq = row.consequents

        # Nếu tiền đề của luật nằm trọn trong giỏ hàng
        if ante.issubset(cart_set):
            unbought = conseq - cart_set
            for item in unbought:
                score = row.lift * row.confidence
                if item not in candidate_map or score > candidate_map[item]['score']:
                    candidate_map[item] = {
                        'item': item,
                        'confidence': row.confidence,
                        'lift': row.lift,
                        'support': row.support,
                        'score': score,
                        'antecedent': ', '.join(list(ante)),
                        'explanation': (
                            f"Vì bạn đã chọn [{', '.join(list(ante))}], xác suất mua thêm [{item}] "
                            f"đạt {row.confidence:.1%}, cao gấp {row.lift:.2f} lần so với mua ngẫu nhiên."
                        )
                    }

    # Sắp xếp theo điểm Score giảm dần
    ranked = sorted(candidate_map.values(), key=lambda x: x['score'], reverse=True)
    return ranked[:top_n]


if __name__ == '__main__':
    # Test thử nghiệm nhanh
    sample_baskets = [
        ['Banana', 'Organic Milk', 'Bread'],
        ['Banana', 'Organic Avocado', 'Apple'],
        ['Organic Milk', 'Bread', 'Eggs'],
        ['Banana', 'Organic Avocado', 'Organic Milk'],
        ['Banana', 'Organic Milk', 'Bread', 'Organic Avocado'],
    ]
    model = OptimizedFPGrowth(min_support=0.3).fit(sample_baskets)
    fi_df = model.get_frequent_itemsets()
    print("Tập phổ biến tìm được:")
    print(fi_df)

    rules = generate_rules(fi_df, min_confidence=0.4, min_lift=1.0)
    print("\nLuật kết hợp sinh ra:")
    print(rules)

    recs = recommend_products(['Banana'], rules)
    print("\nGợi ý khi giỏ hàng có ['Banana']:")
    for r in recs:
        print(f"- {r['item']} (Confidence: {r['confidence']:.1%}, Lift: {r['lift']:.2f})")
        print(f"  {r['explanation']}")
