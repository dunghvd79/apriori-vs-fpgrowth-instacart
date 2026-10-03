"""
Module: association_rules.py
Mục đích: Sinh và đánh giá các luật kết hợp (Association Rules) từ danh sách các tập phổ biến (Frequent Itemsets).
Các chỉ số đánh giá:
- Support(A -> B) = Support(A ∪ B)
- Confidence(A -> B) = Support(A ∪ B) / Support(A)
- Lift(A -> B) = Confidence(A -> B) / Support(B)
- Leverage(A -> B) = Support(A ∪ B) - (Support(A) * Support(B))
- Conviction(A -> B) = (1 - Support(B)) / (1 - Confidence(A -> B))
"""

import sys
from itertools import combinations
from typing import List, Dict, Tuple, Any

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def generate_association_rules(
    frequent_itemsets: List[Dict[str, Any]],
    min_confidence: float = 0.2,
    min_lift: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Sinh các luật kết hợp A -> B từ tập phổ biến.

    Tham số:
        frequent_itemsets: Danh sách dict [{'itemsets': tuple, 'support': float, 'length': int}, ...]
        min_confidence: Ngưỡng tin cậy tối thiểu (0.0 đến 1.0)
        min_lift: Ngưỡng Lift tối thiểu (thường >= 1.0 để lọc các luật có tương quan dương)

    Trả về:
        Danh sách các luật kết hợp được sắp xếp giảm dần theo Lift hoặc Confidence.
    """
    # Tạo map tra cứu support nhanh theo frozenset
    support_map = {frozenset(entry['itemsets']): entry['support'] for entry in frequent_itemsets}
    rules = []

    # Chỉ xét các tập có từ 2 item trở lên
    candidate_itemsets = [frozenset(entry['itemsets']) for entry in frequent_itemsets if entry['length'] >= 2]

    for itemset in candidate_itemsets:
        itemset_support = support_map[itemset]
        items_list = list(itemset)
        k = len(items_list)

        # Xét mọi tập con tiền đề (Antecedent) kích thước từ 1 đến k-1
        for r in range(1, k):
            for antecedent_tuple in combinations(items_list, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                antecedent_support = support_map.get(antecedent, 0.0)
                consequent_support = support_map.get(consequent, 0.0)

                if antecedent_support <= 0 or consequent_support <= 0:
                    continue

                confidence = itemset_support / antecedent_support
                if confidence < min_confidence:
                    continue

                lift = confidence / consequent_support
                if lift < min_lift:
                    continue

                # Các chỉ số bổ trợ
                leverage = itemset_support - (antecedent_support * consequent_support)
                if confidence < 1.0:
                    conviction = (1.0 - consequent_support) / (1.0 - confidence)
                else:
                    conviction = float('inf')

                rules.append({
                    'antecedents': tuple(sorted(list(antecedent))),
                    'consequents': tuple(sorted(list(consequent))),
                    'antecedent_support': antecedent_support,
                    'consequent_support': consequent_support,
                    'support': itemset_support,
                    'confidence': confidence,
                    'lift': lift,
                    'leverage': leverage,
                    'conviction': conviction
                })

    # Sắp xếp giảm dần theo Lift, sau đó đến Confidence
    rules.sort(key=lambda r: (r['lift'], r['confidence']), reverse=True)
    return rules


if __name__ == '__main__':
    from my_fpgrowth import MyFPGrowth

    sample_data = [
        ['Milk', 'Bread', 'Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread'],
        ['Milk', 'Eggs'],
        ['Bread', 'Butter', 'Eggs']
    ]

    fpg = MyFPGrowth(min_support=0.3).fit(sample_data)
    itemsets = fpg.get_frequent_itemsets()
    rules = generate_association_rules(itemsets, min_confidence=0.5, min_lift=1.0)

    print("Test sinh luật kết hợp:")
    for r in rules:
        ant = " + ".join(r['antecedents'])
        con = " + ".join(r['consequents'])
        print(f"[{ant}] => [{con}] | Supp: {r['support']:.2f} | Conf: {r['confidence']:.2f} | Lift: {r['lift']:.2f}")
