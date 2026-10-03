"""
Module: benchmark.py
Mục đích: Đo đạc và so sánh hiệu năng (Execution Time & Peak Memory) giữa Apriori và FP-Growth.
Xuất kết quả ra file bảng (CSV) trong thư mục results/ để phục vụ báo cáo và Dashboard.
"""

import sys
import time
import tracemalloc
import argparse
from pathlib import Path
from typing import List, Dict, Any

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Đảm bảo import được các module trong src
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from my_apriori import MyApriori
from my_fpgrowth import MyFPGrowth


def benchmark_single_algorithm(algo_class, transactions: List[List[str]], min_support: float) -> Dict[str, Any]:
    """
    Đo thời gian chạy (giây) và bộ nhớ RAM đỉnh (MB) của một thuật toán.
    """
    tracemalloc.start()
    start_time = time.perf_counter()

    model = algo_class(min_support=min_support)
    model.fit(transactions)
    itemsets = model.get_frequent_itemsets()

    elapsed_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'min_support': min_support,
        'execution_time_sec': round(elapsed_time, 4),
        'peak_memory_mb': round(peak_mem / (1024 * 1024), 2),
        'num_itemsets': len(itemsets)
    }


def run_benchmark_suite(
    transactions: List[List[str]],
    support_thresholds: List[float] = [0.05, 0.03, 0.02, 0.01],
    output_csv: Path = None
):
    """
    Chạy bộ kiểm thử so sánh trên nhiều ngưỡng min_support khác nhau.
    """
    results = []

    print("\n" + "=" * 70)
    print(f"BẮT ĐẦU BENCHMARK: APRIORI VS FP-GROWTH ({len(transactions):,} GIAO DỊCH)")
    print("=" * 70)
    print(f"{'min_supp':<10} | {'Thuật toán':<12} | {'Thời gian (s)':<15} | {'RAM đỉnh (MB)':<15} | {'Số tập PB':<10}")
    print("-" * 70)

    for supp in support_thresholds:
        # Chạy Apriori
        res_ap = benchmark_single_algorithm(MyApriori, transactions, supp)
        print(f"{supp:<10.3f} | {'Apriori':<12} | {res_ap['execution_time_sec']:<15.4f} | {res_ap['peak_memory_mb']:<15.2f} | {res_ap['num_itemsets']:<10}")
        results.append({
            'algorithm': 'Apriori',
            **res_ap
        })

        # Chạy FP-Growth
        res_fp = benchmark_single_algorithm(MyFPGrowth, transactions, supp)
        print(f"{supp:<10.3f} | {'FP-Growth':<12} | {res_fp['execution_time_sec']:<15.4f} | {res_fp['peak_memory_mb']:<15.2f} | {res_fp['num_itemsets']:<10}")
        results.append({
            'algorithm': 'FP-Growth',
            **res_fp
        })
        print("-" * 70)

    # Lưu ra CSV nếu có yêu cầu
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd
            df_res = pd.DataFrame(results)
            df_res.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n[Thành công] Đã lưu kết quả benchmark vào: {output_csv}")
        except ImportError:
            # Ghi file csv thuần nếu chưa có pandas
            import csv
            with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['algorithm', 'min_support', 'execution_time_sec', 'peak_memory_mb', 'num_itemsets'])
                writer.writeheader()
                writer.writerows(results)
            print(f"\n[Thành công] Đã lưu kết quả benchmark vào: {output_csv}")

    return results


if __name__ == '__main__':
    # Kiểm thử nhanh với tập dữ liệu đồ chơi
    toy_transactions = [
        ['Milk', 'Bread', 'Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread'],
        ['Milk', 'Eggs'],
        ['Bread', 'Butter', 'Eggs'],
        ['Milk', 'Bread', 'Butter', 'Coke'],
        ['Bread', 'Milk', 'Eggs'],
    ] * 100  # Nhân bản lên 700 giao dịch để đo lường

    root = current_dir.parent
    out_file = root / 'results' / 'benchmark_demo.csv'
    run_benchmark_suite(toy_transactions, support_thresholds=[0.3, 0.2, 0.1], output_csv=out_file)
