"""
Module: preprocess.py
Mục đích: Đọc và tiền xử lý dữ liệu Instacart Market Basket Analysis từ file raw CSV.
Chuyển đổi các đơn hàng thành danh sách giao dịch (Transactions) và lưu dưới dạng .parquet.
"""

import os
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


def load_and_preprocess_transactions(
    raw_dir: Path,
    source: str = 'prior',
    sample_orders: int = 50000,
    min_basket_size: int = 2,
    random_state: int = 42
):
    """
    Đọc dữ liệu, lọc mẫu đơn hàng, ánh xạ tên sản phẩm và gom nhóm thành giao dịch.

    Tham số:
        raw_dir: Thư mục chứa các file raw CSV.
        source: 'prior' (tập prior lớn) hoặc 'train' (tập train vừa).
        sample_orders: Số lượng order cần lấy mẫu (<= 0 để lấy toàn bộ).
        min_basket_size: Kích thước giỏ hàng tối thiểu (mặc định = 2).
        random_state: Seed để tái lập kết quả sampling.

    Trả về:
        DataFrame gồm các cột: ['order_id', 'items', 'item_count']
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("[LỖI] Chưa cài đặt thư viện 'pandas' hoặc 'numpy'.")
        print("Vui lòng chạy: pip install -r requirements.txt")
        sys.exit(1)

    products_file = raw_dir / 'products.csv'
    order_products_file = raw_dir / f'order_products__{source}.csv'

    # Kiểm tra file cần thiết
    if not products_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {products_file}")
    if not order_products_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {order_products_file}")

    print(f"\n[1/4] Đang đọc bảng thông tin sản phẩm ({products_file.name})...")
    df_products = pd.read_csv(
        products_file,
        usecols=['product_id', 'product_name'],
        dtype={'product_id': np.int32, 'product_name': 'string'}
    )
    print(f"      -> Tổng cộng {len(df_products):,} sản phẩm.")

    print(f"\n[2/4] Đang nạp danh sách chi tiết đơn hàng ({order_products_file.name})...")
    # Đọc với dtype tối ưu dung lượng RAM
    df_op = pd.read_csv(
        order_products_file,
        usecols=['order_id', 'product_id'],
        dtype={'order_id': np.int32, 'product_id': np.int32}
    )
    print(f"      -> Tổng số bản ghi (order-item): {len(df_op):,}")

    # Lấy mẫu số lượng orders nếu được yêu cầu
    distinct_orders = df_op['order_id'].drop_duplicates()
    total_orders = len(distinct_orders)
    print(f"      -> Tổng số đơn hàng duy nhất: {total_orders:,}")

    if 0 < sample_orders < total_orders:
        print(f"      -> Đang lấy mẫu ngẫu nhiên {sample_orders:,} đơn hàng (random_state={random_state})...")
        sampled_order_ids = distinct_orders.sample(n=sample_orders, random_state=random_state)
        df_op = df_op[df_op['order_id'].isin(sampled_order_ids)]
        print(f"      -> Số bản ghi sau khi lấy mẫu: {len(df_op):,}")

    print("\n[3/4] Ánh xạ tên sản phẩm (Merge product_name)...")
    df_merged = df_op.merge(df_products, on='product_id', how='inner')

    print("\n[4/4] Gom nhóm theo từng đơn hàng (Giao dịch giỏ hàng)...")
    # Gom nhóm theo order_id và gom các product_name thành list
    transactions = (
        df_merged.groupby('order_id')['product_name']
        .apply(list)
        .reset_index(name='items')
    )

    # Đếm số lượng sản phẩm trong giỏ hàng
    transactions['item_count'] = transactions['items'].apply(len)

    # Lọc các đơn hàng có ít nhất min_basket_size sản phẩm
    if min_basket_size > 1:
        before_filter = len(transactions)
        transactions = transactions[transactions['item_count'] >= min_basket_size].reset_index(drop=True)
        print(f"      -> Đã lọc giỏ hàng >= {min_basket_size} món: từ {before_filter:,} còn {len(transactions):,} đơn.")

    return transactions


def print_summary_statistics(df_transactions):
    """In các thông số thống kê cơ bản của tập giao dịch."""
    print("\n" + "=" * 50)
    print("THỐNG KÊ TẬP GIAO DỊCH (TRANSACTION SUMMARY)")
    print("=" * 50)
    total_tx = len(df_transactions)
    counts = df_transactions['item_count']
    all_items = [item for sublist in df_transactions['items'] for item in sublist]
    distinct_items = len(set(all_items))

    print(f"Số lượng giao dịch (Orders):      {total_tx:,}")
    print(f"Tổng số lượt sản phẩm mua:        {len(all_items):,}")
    print(f"Số sản phẩm phân biệt:            {distinct_items:,}")
    print(f"Kích thước giỏ hàng trung bình:   {counts.mean():.2f} món")
    print(f"Kích thước giỏ hàng nhỏ nhất:     {counts.min()} món")
    print(f"Kích thước giỏ hàng lớn nhất:     {counts.max()} món")
    print(f"Độ lệch chuẩn kích thước giỏ:     {counts.std():.2f}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu Instacart thành danh sách giỏ hàng (.parquet)")
    parser.add_argument('--raw_dir', type=str, default=None, help="Đường dẫn thư mục data/raw")
    parser.add_argument('--output', type=str, default=None, help="Đường dẫn file .parquet đầu ra")
    parser.add_argument('--source', type=str, default='prior', choices=['prior', 'train'], help="Nguồn: prior hoặc train")
    parser.add_argument('--sample', type=int, default=50000, help="Số lượng order lấy mẫu (0 = lấy tất cả)")
    parser.add_argument('--min_basket_size', type=int, default=2, help="Kích thước giỏ hàng tối thiểu")
    parser.add_argument('--check_only', action='store_true', help="Chỉ kiểm tra sự tồn tại của file raw và thoát")

    args = parser.parse_args()
    root = get_project_root()

    raw_dir = Path(args.raw_dir) if args.raw_dir else root / 'data' / 'raw'
    
    # Kiểm tra file raw
    file_status = check_raw_files(raw_dir)
    print("\nTrạng thái thư mục data/raw:")
    for key, info in file_status.items():
        status_str = f"CO ({info['size_mb']} MB)" if info['exists'] else "CHUA CO"
        print(f" - {info['path'].name:<30}: {status_str}")

    if args.check_only:
        return

    # Xác định đường dẫn file đầu ra
    if args.output:
        out_path = Path(args.output)
    else:
        sample_tag = f"{args.sample // 1000}k" if args.sample > 0 else "all"
        out_path = root / 'data' / 'processed' / f'transactions_{args.source}_{sample_tag}.parquet'

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_tx = load_and_preprocess_transactions(
            raw_dir=raw_dir,
            source=args.source,
            sample_orders=args.sample,
            min_basket_size=args.min_basket_size
        )
        print_summary_statistics(df_tx)

        print(f"Đang lưu kết quả ra file Parquet: {out_path} ...")
        df_tx.to_parquet(out_path, engine='pyarrow', index=False)
        file_size_mb = round(out_path.stat().st_size / (1024 * 1024), 2)
        print(f"✓ Hoàn tất! Kích thước file: {file_size_mb} MB")

    except FileNotFoundError as e:
        print(f"\n[LỖI] {e}")
        print("Vui lòng tải bộ dữ liệu Instacart từ Kaggle và đặt vào thư mục data/raw/ trước khi chạy.")
        sys.exit(1)


if __name__ == '__main__':
    main()
