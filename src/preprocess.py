"""
Module: preprocess.py
Mục đích: Đọc và tiền xử lý dữ liệu Instacart Market Basket Analysis từ file raw CSV.
Chuyển đổi các đơn hàng thành danh sách giao dịch (Transactions) và lưu dưới dạng .parquet.

Đầu ra (mặc định):
    data/processed/transactions_product_10k.parquet
    data/processed/transactions_product_50k.parquet
    data/processed/transactions_product_100k.parquet
    data/processed/transactions_aisle_100k.parquet

Các mẫu được lấy LỒNG NHAU (10k ⊂ 50k ⊂ 100k): xáo trộn danh sách đơn hàng một lần
với seed cố định rồi lấy N đơn đầu tiên. Nhờ vậy khi so sánh theo quy mô, khác biệt
chỉ đến từ số lượng đơn chứ không phải do mỗi mẫu là một tập đơn hàng khác nhau.
"""

import sys
import argparse
from pathlib import Path

# Đảm bảo hiển thị tiếng Việt trên Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def get_project_root() -> Path:
    """Lấy đường dẫn thư mục gốc của project."""
    return Path(__file__).resolve().parent.parent


def check_raw_files(raw_dir: Path) -> dict:
    """
    Kiểm tra sự tồn tại của các file dữ liệu Instacart trong thư mục data/raw.
    """
    expected_files = {
        'products': 'products.csv',
        'orders': 'orders.csv',
        'order_products_prior': 'order_products__prior.csv',
        'order_products_train': 'order_products__train.csv',
        'aisles': 'aisles.csv',
        'departments': 'departments.csv',
    }
    status = {}
    for key, filename in expected_files.items():
        file_path = raw_dir / filename
        status[key] = {
            'path': file_path,
            'exists': file_path.is_file(),
            'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2) if file_path.is_file() else 0
        }
    return status


def size_tag(n: int) -> str:
    """10000 -> '10k'."""
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def load_raw(raw_dir: Path, source: str = 'prior'):
    """
    Đọc các bảng cần thiết với kiểu dữ liệu tối ưu RAM.

    Trả về:
        (df_op, df_products) trong đó df_products đã có cột 'aisle' (tên kệ hàng).
    """
    import pandas as pd
    import numpy as np

    order_products_file = raw_dir / f'order_products__{source}.csv'
    for f in [raw_dir / 'products.csv', raw_dir / 'aisles.csv', order_products_file]:
        if not f.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {f}")

    print(f"\n[1/4] Đọc bảng sản phẩm và kệ hàng...")
    df_products = pd.read_csv(
        raw_dir / 'products.csv',
        usecols=['product_id', 'product_name', 'aisle_id'],
        dtype={'product_id': np.int32, 'product_name': 'string', 'aisle_id': np.int16}
    )
    df_aisles = pd.read_csv(raw_dir / 'aisles.csv', dtype={'aisle_id': np.int16, 'aisle': 'string'})
    df_products = df_products.merge(df_aisles, on='aisle_id', how='left')
    print(f"      -> {len(df_products):,} sản phẩm, {len(df_aisles):,} kệ hàng.")

    print(f"\n[2/4] Đọc chi tiết đơn hàng ({order_products_file.name})...")
    df_op = pd.read_csv(
        order_products_file,
        usecols=['order_id', 'product_id'],
        dtype={'order_id': np.int32, 'product_id': np.int32}
    )
    print(f"      -> {len(df_op):,} dòng (đơn hàng - sản phẩm).")
    return df_op, df_products


def nested_sample_order_ids(df_op, sizes, min_basket_size: int, seed: int):
    """
    Lọc các đơn đủ kích thước tối thiểu, xáo trộn MỘT lần, rồi cắt N đơn đầu cho từng mức.

    Lọc trước khi lấy mẫu để mỗi mẫu có đúng N đơn hàng.

    Trả về:
        dict {size: np.ndarray các order_id}
    """
    import numpy as np

    basket_sizes = df_op.groupby('order_id').size()
    eligible = basket_sizes[basket_sizes >= min_basket_size].index.to_numpy()
    print(f"      -> {len(basket_sizes):,} đơn hàng, trong đó {len(eligible):,} đơn có >= {min_basket_size} món.")

    max_size = max(sizes)
    if max_size > len(eligible):
        raise ValueError(f"Mẫu lớn nhất ({max_size:,}) vượt quá số đơn hợp lệ ({len(eligible):,}).")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible)
    return {n: shuffled[:n] for n in sorted(sizes)}


def build_transactions(df_op, df_products, order_ids, level: str = 'product', min_basket_size: int = 1):
    """
    Gom các dòng thuộc order_ids thành giao dịch.

    Tham số:
        level: 'product' (tên sản phẩm) hoặc 'aisle' (tên kệ hàng, loại trùng trong mỗi đơn).
        min_basket_size: áp lại quy tắc giỏ tối thiểu SAU khi gom. Cần cho mức kệ hàng,
            vì một đơn nhiều sản phẩm có thể chỉ còn 1 kệ sau khi bỏ trùng.

    Trả về:
        DataFrame gồm các cột: ['order_id', 'items', 'item_count']
    """
    col = 'product_name' if level == 'product' else 'aisle'
    subset = df_op[df_op['order_id'].isin(order_ids)]
    merged = subset.merge(df_products[['product_id', col]], on='product_id', how='inner')

    if level == 'aisle':
        # Bỏ kệ rỗng (NA) và nhãn 'missing' ("chưa phân loại", không phải kệ hàng thật)
        merged = merged[merged[col].notna() & (merged[col] != 'missing')]
        merged = merged.drop_duplicates(subset=['order_id', col])

    transactions = (
        merged.groupby('order_id')[col]
        .apply(lambda s: sorted(s.astype(str).tolist()))
        .reset_index(name='items')
    )
    transactions['item_count'] = transactions['items'].apply(len)
    transactions = transactions[transactions['item_count'] >= min_basket_size].reset_index(drop=True)
    return transactions


def print_summary_statistics(df_transactions, title: str = ""):
    """In các thông số thống kê cơ bản của tập giao dịch."""
    counts = df_transactions['item_count']
    all_items = [item for sublist in df_transactions['items'] for item in sublist]
    print(f"    {title}")
    print(f"      Số giao dịch:            {len(df_transactions):,}")
    print(f"      Số item phân biệt:       {len(set(all_items)):,}")
    print(f"      Kích thước giỏ TB:       {counts.mean():.2f} (min {counts.min()}, max {counts.max()})")


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu Instacart thành danh sách giỏ hàng (.parquet)")
    parser.add_argument('--raw_dir', type=str, default=None, help="Đường dẫn thư mục data/raw")
    parser.add_argument('--out_dir', type=str, default=None, help="Thư mục đầu ra (mặc định data/processed)")
    parser.add_argument('--source', type=str, default='prior', choices=['prior', 'train'], help="Nguồn: prior hoặc train")
    parser.add_argument('--sizes', type=int, nargs='+', default=[10000, 50000, 100000],
                        help="Các mức lấy mẫu (lồng nhau) cho mức sản phẩm")
    parser.add_argument('--aisle_size', type=int, default=100000,
                        help="Mức mẫu dùng cho phân tích kệ hàng (0 = bỏ qua)")
    parser.add_argument('--min_basket_size', type=int, default=2, help="Kích thước giỏ hàng tối thiểu")
    parser.add_argument('--seed', type=int, default=42, help="Seed để tái lập kết quả lấy mẫu")
    parser.add_argument('--check_only', action='store_true', help="Chỉ kiểm tra sự tồn tại của file raw và thoát")
    args = parser.parse_args()
    if any(n <= 0 for n in args.sizes) or args.aisle_size < 0 or args.min_basket_size < 1:
        parser.error("--sizes phải > 0, --aisle_size >= 0, --min_basket_size >= 1")

    root = get_project_root()
    raw_dir = Path(args.raw_dir) if args.raw_dir else root / 'data' / 'raw'
    out_dir = Path(args.out_dir) if args.out_dir else root / 'data' / 'processed'

    file_status = check_raw_files(raw_dir)
    print("\nTrạng thái thư mục data/raw:")
    for info in file_status.values():
        status_str = f"CO ({info['size_mb']} MB)" if info['exists'] else "CHUA CO"
        print(f" - {info['path'].name:<30}: {status_str}")
    if args.check_only:
        return

    try:
        import pandas  # noqa: F401
    except ImportError:
        print("[LỖI] Chưa cài 'pandas'. Chạy: pip install -r requirements.txt")
        sys.exit(1)

    sizes = sorted(set(args.sizes))
    all_sizes = sizes + ([args.aisle_size] if args.aisle_size > 0 else [])

    try:
        df_op, df_products = load_raw(raw_dir, args.source)
    except FileNotFoundError as e:
        print(f"\n[LỖI] {e}")
        sys.exit(1)

    print(f"\n[3/4] Lấy mẫu lồng nhau (seed={args.seed})...")
    samples = nested_sample_order_ids(df_op, all_sizes, args.min_basket_size, args.seed)

    print(f"\n[4/4] Tạo giao dịch và lưu file...")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [('product', n) for n in sizes]
    if args.aisle_size > 0:
        jobs.append(('aisle', args.aisle_size))

    for level, n in jobs:
        df_tx = build_transactions(df_op, df_products, samples[n], level=level,
                                   min_basket_size=args.min_basket_size)
        
        # 1. Lưu file .parquet (Tối ưu tốc độ & bộ nhớ cho thuật toán Mining)
        out_parquet_path = out_dir / f"transactions_{level}_{size_tag(n)}.parquet"
        df_tx.to_parquet(out_parquet_path, engine='pyarrow', index=False)
        parquet_size_mb = out_parquet_path.stat().st_size / (1024 * 1024)
        
        # 2. Lưu file .csv (Mở xem trực tiếp trong Excel / Notepad)
        out_csv_path = out_dir / f"transactions_{level}_{size_tag(n)}.csv"
        df_csv = df_tx.copy()
        # Chuyển list thành chuỗi phân cách bởi dấu ' | ' để không trùng với dấu phẩy trong tên sản phẩm
        df_csv['items'] = df_csv['items'].apply(' | '.join)
        df_csv.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
        csv_size_mb = out_csv_path.stat().st_size / (1024 * 1024)
        
        print_summary_statistics(
            df_tx, 
            title=f"{out_parquet_path.name} ({parquet_size_mb:.2f} MB) & {out_csv_path.name} ({csv_size_mb:.2f} MB)"
        )

    print("\n✓ Hoàn tất tiền xử lý (Đã xuất song song cả .parquet và .csv).")


if __name__ == '__main__':
    main()
