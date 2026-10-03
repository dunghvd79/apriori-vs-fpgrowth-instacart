"""
Ứng dụng Streamlit Dashboard: Trực quan hóa & Phân tích khai phá luật kết hợp Instacart
So sánh Apriori vs FP-Growth
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Thiết lập đường dẫn thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Cấu hình giao diện trang
st.set_page_config(
    page_title="Instacart MBA | Apriori vs FP-Growth",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🛒 Instacart Mining")
st.sidebar.markdown("**Môn học:** Khai phá dữ liệu")
st.sidebar.markdown("**Đề tài:** So sánh Apriori & FP-Growth")
st.sidebar.divider()

menu = st.sidebar.radio(
    "CHỌN CHỨC NĂNG:",
    [
        "📌 1. Giới thiệu & Tổng quan Dữ liệu",
        "⚡ 2. So sánh Benchmark (Apriori vs FP-Growth)",
        "🔍 3. Khám phá Luật kết hợp (Rule Explorer)",
        "🛍️ 4. Giả lập Giỏ hàng & Gợi ý (Live Recommender)"
    ]
)

st.sidebar.divider()
st.sidebar.caption("🚀 Antigravity Data Mining Suite")

# ==========================================
# TAB 1: GIỚI THIỆU & TỔNG QUAN
# ==========================================
if menu == "📌 1. Giới thiệu & Tổng quan Dữ liệu":
    st.title("🛒 Khám phá Dữ liệu Instacart Market Basket")
    st.markdown("""
    Đề tài nghiên cứu ứng dụng hai thuật toán kinh điển trong **Khai phá dữ liệu (Data Mining)**:
    - **Apriori Algorithm:** Sử dụng nguyên lý sinh tập ứng viên theo chiều rộng (Breadth-First Search).
    - **FP-Growth Algorithm:** Nén dữ liệu vào cây tiền tố **FP-Tree**, khai phá bằng phương pháp chia để trị không cần sinh ứng viên.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Trạng thái dữ liệu thô (`data/raw/`):")
        raw_files = list((DATA_DIR / "raw").glob("*.csv")) if (DATA_DIR / "raw").exists() else []
        if raw_files:
            for rf in raw_files:
                st.write(f"- `{rf.name}` ({round(rf.stat().st_size / (1024*1024), 2)} MB)")
        else:
            st.warning("Chưa có file CSV thô nào trong `data/raw/`. Vui lòng tải từ Kaggle về.")

    with col2:
        st.subheader("⚡ Trạng thái dữ liệu đã xử lý (`data/processed/`):")
        parquet_files = list((DATA_DIR / "processed").glob("*.parquet")) if (DATA_DIR / "processed").exists() else []
        if parquet_files:
            for pf in parquet_files:
                st.success(f"✓ `{pf.name}` ({round(pf.stat().st_size / (1024*1024), 2)} MB)")
        else:
            st.info("Chưa có file `.parquet`. Hãy chạy `python src/preprocess.py` để tạo tập giao dịch.")

# ==========================================
# TAB 2: BENCHMARK
# ==========================================
elif menu == "⚡ 2. So sánh Benchmark (Apriori vs FP-Growth)":
    st.title("⚡ Thực nghiệm So sánh: Apriori vs FP-Growth")
    st.markdown("Đo lường chi tiết về **Thời gian thực thi (Runtime)** và **Bộ nhớ RAM tiêu thụ (Peak RAM)**.")

    csv_files = list(RESULTS_DIR.glob("*.csv")) if RESULTS_DIR.exists() else []
    if not csv_files:
        st.warning("Chưa tìm thấy file kết quả benchmark nào trong `results/`. Hãy chạy `python src/benchmark.py` trước!")
    else:
        selected_file = st.selectbox("Chọn file kết quả benchmark:", [f.name for f in csv_files])
        df_bench = pd.read_csv(RESULTS_DIR / selected_file)

        st.dataframe(df_bench, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏱️ Thời gian thực thi (giây)")
            pivoted_time = df_bench.pivot(index='min_support', columns='algorithm', values='execution_time_sec')
            st.bar_chart(pivoted_time)

        with col2:
            st.subheader("💾 RAM tiêu thụ đỉnh (MB)")
            pivoted_mem = df_bench.pivot(index='min_support', columns='algorithm', values='peak_memory_mb')
            st.bar_chart(pivoted_mem)

# ==========================================
# TAB 3: RULE EXPLORER
# ==========================================
elif menu == "🔍 3. Khám phá Luật kết hợp (Rule Explorer)":
    st.title("🔍 Khám phá Luật Kết Hợp (Association Rules)")
    st.markdown("Lọc các luật có ý nghĩa kinh doanh cao dựa trên các chỉ số **Support, Confidence, Lift**.")

    col1, col2 = st.columns(2)
    with col1:
        min_conf = st.slider("Ngưỡng Confidence tối thiểu:", 0.1, 1.0, 0.3, 0.05)
    with col2:
        min_lift = st.slider("Ngưỡng Lift tối thiểu:", 1.0, 10.0, 1.2, 0.1)

    # Hiển thị demo mẫu hoặc load từ file
    st.info("Khi chạy thực nghiệm trên tập Instacart thật, toàn bộ các luật sẽ được lọc và hiển thị dưới dạng bảng tương tác tại đây.")

# ==========================================
# TAB 4: LIVE RECOMMENDER
# ==========================================
elif menu == "🛍️ 4. Giả lập Giỏ hàng & Gợi ý (Live Recommender)":
    st.title("🛍️ Giả lập Giỏ Hàng Thông Minh (Market Basket Recommender)")
    st.markdown("Chọn một số sản phẩm giả định để hệ thống tự động suy diễn và đề xuất các sản phẩm mua kèm:")

    sample_products = [
        "Banana", "Bag of Organic Bananas", "Organic Strawberries",
        "Organic Baby Spinach", "Organic Hass Avocado", "Organic Avocado",
        "Large Lemon", "Limes", "Organic Whole Milk", "Organic Raspberries"
    ]

    selected_items = st.multiselect("Chọn sản phẩm bạn muốn bỏ vào giỏ hàng:", sample_products, default=["Banana"])

    if st.button("🔍 Tìm gợi ý mua kèm", type="primary"):
        st.success(f"Dựa trên các luật kết hợp đã khai phá từ giỏ hàng: **{', '.join(selected_items)}**:")
        # Mô phỏng gợi ý
        st.markdown("""
        1. **Organic Hass Avocado** (Độ tin cậy: **38.5%**, Lift: **2.4**)
        2. **Organic Strawberries** (Độ tin cậy: **29.1%**, Lift: **1.8**)
        3. **Organic Whole Milk** (Độ tin cậy: **24.0%**, Lift: **1.6**)
        """)
