"""
Module: benchmark.py
Mục đích: So sánh hiệu năng Apriori vs FP-Growth (thư viện mlxtend) theo thời gian và RAM.

Thiết kế đo lường:
- Mỗi lần chạy nằm trong MỘT TIẾN TRÌNH RIÊNG (spawn), vì vậy:
    + RAM đo được không bị nhiễm bởi lần chạy trước;
    + có thể dừng cứng khi vượt timeout.
- Chỉ bấm giờ phần khai phá. Bước đọc dữ liệu và mã hóa one-hot KHÔNG nằm trong thời gian đo,
  vì hai thuật toán dùng chung đầu vào đó.
- RAM = mức tăng so với trước khi khai phá (RSS đỉnh - RSS nền), lấy mẫu mỗi 20 ms.
  Không dùng tracemalloc vì nó làm chậm code Python và bỏ sót bộ nhớ của numpy/pandas.
- Nếu một thuật toán timeout ở ngưỡng min_support nào đó, bỏ qua các ngưỡng thấp hơn
  của cùng quy mô (chắc chắn còn chậm hơn), ghi trạng thái 'skipped'.

Cách dùng:
    python src/benchmark.py                      # chạy đầy đủ theo kế hoạch
    python src/benchmark.py --quick              # chạy nhanh để kiểm tra pipeline
    python src/benchmark.py --sizes 10k --supports 0.02 0.01 --repeats 1
"""

import sys
import time
import argparse
import threading
import multiprocessing as mp
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mining import ALGORITHMS, get_project_root, load_transactions, encode_transactions, mine_frequent_itemsets

DEFAULT_SIZES = ['10k', '50k', '100k']
DEFAULT_SUPPORTS = [0.05, 0.03, 0.02, 0.01, 0.005]
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_MIN = 30
FIELDS = ['algorithm', 'size', 'n_transactions', 'min_support', 'run',
          'time_sec', 'peak_mem_mb', 'n_itemsets', 'n_items_after_prune', 'status']


class _PeakRSS:
    """Lấy mẫu RSS của tiến trình hiện tại trong nền để biết mức đỉnh."""

    def __init__(self, interval: float = 0.02):
        import psutil
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self._interval)

    def __enter__(self):
        self.baseline = self._proc.memory_info().rss
        self.peak = self.baseline
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)


def _worker(queue, algorithm, parquet_path, min_support):
    """Chạy trong tiến trình con: nạp dữ liệu, mã hóa, rồi đo riêng phần khai phá."""
    try:
        transactions = load_transactions(parquet_path)
        onehot = encode_transactions(transactions, min_support)

        with _PeakRSS() as mem:
            start = time.perf_counter()
            result = mine_frequent_itemsets(algorithm, onehot, min_support)
            elapsed = time.perf_counter() - start

        queue.put({
            'time_sec': round(elapsed, 4),
            'peak_mem_mb': round((mem.peak - mem.baseline) / (1024 * 1024), 2),
            'n_itemsets': len(result),
            'n_items_after_prune': onehot.shape[1],
            'status': 'ok',
        })
    except MemoryError:
        queue.put({'status': 'oom'})
    except Exception as e:  # noqa: BLE001
        queue.put({'status': f'error: {type(e).__name__}: {e}'})


def run_once(algorithm, parquet_path, min_support, timeout_sec):
    """Chạy một lần trong tiến trình riêng, trả về dict kết quả."""
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, algorithm, str(parquet_path), min_support))
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {'status': 'timeout'}
    if not queue.empty():
        return queue.get()
    return {'status': f'crashed (exit code {proc.exitcode})'}


def run_benchmark(sizes, supports, repeats, timeout_min, out_csv: Path):
    import csv
    root = get_project_root()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    supports = sorted(supports, reverse=True)  # cao -> thấp, để cắt sớm khi timeout

    total = len(sizes) * len(supports) * len(ALGORITHMS) * repeats
    print(f"\nBENCHMARK: {len(sizes)} quy mô x {len(supports)} ngưỡng x {len(ALGORITHMS)} thuật toán "
          f"x {repeats} lần = tối đa {total} lần chạy (timeout {timeout_min} phút/lần)\n")
    header = f"{'size':<6}{'min_supp':<10}{'thuật toán':<11}{'lần':<5}{'thời gian (s)':<15}{'RAM +MB':<10}{'#itemset':<10}{'trạng thái'}"
    print(header)
    print('-' * len(header))

    rows = []
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for size in sizes:
            path = root / 'data' / 'processed' / f'transactions_product_{size}.parquet'
            if not path.exists():
                print(f"[BỎ QUA] Thiếu file {path.name}. Chạy src/preprocess.py trước.")
                continue
            n_tx = len(load_transactions(path))
            blocked = set()  # thuật toán đã timeout ở quy mô này

            for supp in supports:
                for algo in ALGORITHMS:
                    for run in range(1, repeats + 1):
                        base = {'algorithm': algo, 'size': size, 'n_transactions': n_tx,
                                'min_support': supp, 'run': run}
                        if algo in blocked:
                            res = {'status': 'skipped'}
                        else:
                            res = run_once(algo, path, supp, timeout_min * 60)
                            if res['status'] in ('timeout', 'oom'):
                                blocked.add(algo)
                        row = {k: '' for k in FIELDS} | base | res
                        writer.writerow(row)
                        f.flush()
                        rows.append(row)
                        print(f"{size:<6}{supp:<10.3f}{algo:<11}{run:<5}{str(row['time_sec']):<15}"
                              f"{str(row['peak_mem_mb']):<10}{str(row['n_itemsets']):<10}{row['status']}")
                        if res['status'] in ('timeout', 'oom', 'skipped'):
                            break  # không lặp lại nếu đã hỏng

    print(f"\n✓ Đã lưu kết quả: {out_csv}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="So sánh Apriori vs FP-Growth (mlxtend)")
    parser.add_argument('--sizes', nargs='+', default=DEFAULT_SIZES, help="Quy mô: 10k 50k 100k")
    parser.add_argument('--supports', nargs='+', type=float, default=DEFAULT_SUPPORTS, help="Các ngưỡng min_support")
    parser.add_argument('--repeats', type=int, default=DEFAULT_REPEATS, help="Số lần lặp mỗi cấu hình")
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT_MIN, help="Timeout (phút) mỗi lần chạy")
    parser.add_argument('--out', type=str, default=None, help="File CSV đầu ra")
    parser.add_argument('--quick', action='store_true', help="Chạy nhanh: 10k, 2 ngưỡng, 1 lần")
    args = parser.parse_args()

    if args.quick:
        args.sizes, args.supports, args.repeats = ['10k'], [0.02, 0.01], 1

    out = Path(args.out) if args.out else get_project_root() / 'results' / (
        'benchmark_quick.csv' if args.quick else 'benchmark_results.csv')
    run_benchmark(args.sizes, args.supports, args.repeats, args.timeout, out)


if __name__ == '__main__':
    mp.freeze_support()
    main()
