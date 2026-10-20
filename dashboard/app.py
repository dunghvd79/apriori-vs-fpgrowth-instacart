"""
Dashboard: Hệ Thống Phân Tích Giỏ Hàng & Gợi Ý Mua Sắm (Market Basket Intelligence)
Chuyên đề: Dự đoán sản phẩm mua kèm & Khai phá luật kết hợp trên Instacart Dataset
Động cơ: Thuật toán FP-Growth tự cài đặt (src/my_fpgrowth.py)
Phong cách: Light Analytics (Trắng #FFFFFF, Navy #1E3A8A, Teal #0D9488)
Trực quan, tương phản cao, tối ưu hiển thị và bảo đảm toàn bộ số liệu đều từ dữ liệu chạy thực tế.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# Đường dẫn thư mục dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import trực tiếp thuật toán FP-Growth tự cài đặt
from my_fpgrowth import OptimizedFPGrowth, generate_rules, recommend_products

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Market Basket Recommender | FP-Tree Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# HỆ THỐNG MÀU SẮC LIGHT ANALYTICS (TRẮNG, NAVY, TEAL)
# ==============================================================================
COLOR_NAVY = "#1E3A8A"        # Navy đậm (Tiêu đề, điểm nhấn chính)
COLOR_TEAL = "#0D9488"        # Teal đậm (Màu chủ đạo cho luật & gợi ý)
COLOR_BG = "#FFFFFF"          # Trắng nền chuẩn
COLOR_CARD_BG = "#F8FAFC"     # Slate 50 (Nền thẻ nhẹ)
COLOR_TEXT = "#0F172A"        # Slate 900 (Đen sẫm, độ tương phản cao nhất)
COLOR_TEXT_MUTED = "#334155"  # Slate 700 (Đậm rõ, không bị mờ nhạt)
COLOR_BORDER = "#CBD5E1"      # Slate 300 (Đường viền sắc nét)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      
      /* Màu sắc tổng thể trang */
      html, body, [class*="css"], .stApp {{
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
          color: {COLOR_TEXT} !important;
          background-color: {COLOR_BG} !important;
      }}

      p, span, label, div, li, h1, h2, h3, h4, h5, h6 {{
          color: {COLOR_TEXT} !important;
      }}

      .stCaption, [data-testid="stCaptionContainer"], small {{
          color: {COLOR_TEXT_MUTED} !important;
          font-size: 0.88rem !important;
          font-weight: 500 !important;
      }}

      /* Banner tiêu đề phong cách Light Analytics */
      .header-container {{
          background: #FFFFFF !important;
          border: 1px solid {COLOR_BORDER} !important;
          border-left: 6px solid {COLOR_TEAL} !important;
          border-radius: 8px !important;
          padding: 1.25rem 1.6rem !important;
          margin-bottom: 1.4rem !important;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
      }}
      .header-title {{
          font-size: 2.05rem !important;
          font-weight: 800 !important;
          color: {COLOR_NAVY} !important;
          margin: 0 !important;
          letter-spacing: -0.02em !important;
      }}
      .header-subtitle {{
          font-size: 1.12rem !important;
          font-weight: 700 !important;
          color: {COLOR_TEAL} !important;
          margin: 0.35rem 0 0.45rem 0 !important;
      }}
      .header-desc {{
          font-size: 0.95rem !important;
          color: {COLOR_TEXT_MUTED} !important;
          margin: 0 !important;
          line-height: 1.5 !important;
      }}

      /* Thẻ chỉ số tổng quan (KPI Cards) */
      [data-testid="stMetricValue"] {{
          font-size: 1.75rem !important;
          font-weight: 800 !important;
          color: {COLOR_NAVY} !important;
      }}
      [data-testid="stMetricLabel"] {{
          font-size: 0.85rem !important;
          font-weight: 700 !important;
          color: {COLOR_TEXT_MUTED} !important;
          text-transform: uppercase !important;
          letter-spacing: 0.04em !important;
      }}
      [data-testid="stMetric"] {{
          background: {COLOR_CARD_BG} !important;
          border: 1px solid {COLOR_BORDER} !important;
          border-radius: 8px !important;
          padding: 12px 16px !important;
      }}

      /* Tab bar */
      button[data-baseweb="tab"] {{
          font-size: 0.96rem !important;
          font-weight: 700 !important;
          color: {COLOR_TEXT_MUTED} !important;
          padding: 10px 20px !important;
      }}
      button[data-baseweb="tab"][aria-selected="true"] {{
          color: {COLOR_TEAL} !important;
          border-bottom-color: {COLOR_TEAL} !important;
          border-bottom-width: 3px !important;
      }}

      /* Card gợi ý sản phẩm mua kèm */
      .recommender-card {{
          background: #FFFFFF !important;
          border: 1px solid {COLOR_BORDER} !important;
          border-left: 5px solid {COLOR_TEAL} !important;
          border-radius: 8px !important;
          padding: 15px 18px !important;
          margin-bottom: 13px !important;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
      }}
      .recommender-card:hover {{
          border-color: {COLOR_TEAL} !important;
          box-shadow: 0 4px 12px rgba(13, 148, 136, 0.12) !important;
      }}

      /* Hệ thống huy hiệu nhãn (Badges) */
      .badge-tag {{
          display: inline-block !important;
          background: #E0F2FE !important;
          color: #0369A1 !important;
          padding: 4px 10px !important;
          border-radius: 6px !important;
          font-size: 0.83rem !important;
          font-weight: 700 !important;
          margin-right: 6px !important;
          border: 1px solid #BAE6FD !important;
      }}
      .badge-lift {{
          background: #CCFBF1 !important;
          color: #0F766E !important;
          border: 1px solid #99F6E4 !important;
      }}
      .badge-score {{
          background: #FEF3C7 !important;
          color: #B45309 !important;
          border: 1px solid #FDE68A !important;
      }}
      .badge-action {{
          background: #EEF2FF !important;
          color: #3730A3 !important;
          border: 1px solid #C7D2FE !important;
      }}

      /* Hộp diễn giải cơ chế */
      .info-box {{
          background: {COLOR_CARD_BG} !important;
          border: 1px solid {COLOR_BORDER} !important;
          border-radius: 8px !important;
          padding: 14px 18px !important;
          margin-bottom: 14px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_plot_theme(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    """Áp dụng theme Light Analytics cho tất cả biểu đồ Plotly."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color=COLOR_TEXT, family="Inter"),
            x=0.0,
            y=0.98,
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=COLOR_TEXT, family="Inter"),
        margin=dict(l=45, r=20, t=45, b=45),
        xaxis=dict(
            title=dict(text=f"<b>{x_title}</b>", font=dict(size=12, color=COLOR_TEXT)),
            gridcolor="#E2E8F0",
            zeroline=False,
            tickfont=dict(size=11, color=COLOR_TEXT, family="Inter"),
            showline=True,
            linecolor="#94A3B8",
        ),
        yaxis=dict(
            title=dict(text=f"<b>{y_title}</b>", font=dict(size=12, color=COLOR_TEXT)),
            gridcolor="#E2E8F0",
            zeroline=False,
            tickfont=dict(size=11, color=COLOR_TEXT, family="Inter"),
            showline=True,
            linecolor="#94A3B8",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=COLOR_TEXT, family="Inter"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
    )
    return fig


# ==============================================================================
# HÀM NẠP VÀ KHAI PHÁ DỮ LIỆU THỰC TẾ
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_processed_transactions(file_path: str, max_rows: int = 0) -> List[List[str]]:
    """Đọc dữ liệu giao dịch từ tệp parquet hoặc csv."""
    path = Path(file_path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=["items"])
        txs = [list(items) for items in df["items"]]
    else:
        df = pd.read_csv(path)
        if "items" not in df.columns:
            return []
        first_val = str(df["items"].iloc[0]) if len(df) > 0 else ""
        if " | " in first_val:
            txs = [str(val).split(" | ") for val in df["items"]]
        else:
            import ast
            txs = [
                ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else str(val).split(",")
                for val in df["items"]
            ]
    if max_rows > 0:
        txs = txs[:max_rows]
    return txs


@st.cache_data(show_spinner=False)
def mine_frequent_patterns(file_path: str, max_rows: int, min_support: float) -> pd.DataFrame:
    """Khai phá các tập phổ biến bằng FP-Tree tự cài đặt trên dữ liệu thực tế."""
    txs = load_processed_transactions(file_path, max_rows)
    model = OptimizedFPGrowth(min_support=min_support).fit(txs)
    return model.get_frequent_itemsets()


@st.cache_data(show_spinner=False)
def extract_association_rules(
    file_path: str, max_rows: int, min_support: float, min_confidence: float, min_lift: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trích xuất luật kết hợp thực tế từ các tập phổ biến đã tìm thấy."""
    fi_df = mine_frequent_patterns(file_path, max_rows, min_support)
    rules_df = generate_rules(fi_df, min_confidence=min_confidence, min_lift=min_lift)
    return rules_df, fi_df


# ==============================================================================
# HEADER BANNER
# ==============================================================================
st.markdown(
    """
    <div class="header-container">
      <h1 class="header-title">Hệ Thống Phân Tích Giỏ Hàng & Gợi Ý Mua Sắm</h1>
      <div class="header-subtitle">Market Basket Intelligence & Product Recommender System | Instacart Dataset</div>
      <p class="header-desc">
        Hệ thống phân tích hành vi mua sắm chuyên sâu phục vụ kinh doanh bán lẻ: 
        <b>Dự đoán sản phẩm mua kèm</b>, <b>khám phá luật kết hợp giỏ hàng</b> và 
        <b>phân tích cấu trúc nén dữ liệu của cây tiền tố FP-Tree</b>. 
        Mọi số liệu trên giao diện đều được tính toán trực tiếp từ dữ liệu giao dịch thực tế.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SIDEBAR: THIẾT LẬP THAM SỐ NGHIỆP VỤ
# ==============================================================================
st.sidebar.markdown("### ⚙️ Thiết Lập Nghiệp Vụ")

# 1. Chọn file dữ liệu thực tế
available_files = []
if DATA_DIR.exists():
    available_files = sorted(list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.csv")))

if not available_files:
    st.sidebar.error("Chưa có tệp dữ liệu trong `data/processed/`. Vui lòng chạy `preprocess.py` trước.")
    st.stop()

default_idx = 0
for idx, f in enumerate(available_files):
    if "product_10k" in f.name:
        default_idx = idx
        break

selected_file = st.sidebar.selectbox(
    "1. Tệp giao dịch (data/processed/):",
    available_files,
    index=default_idx,
    format_func=lambda f: f"{f.name} ({round(f.stat().st_size / (1024*1024), 2)} MB)"
)

# 2. Số đơn hàng nạp
max_rows_input = st.sidebar.number_input(
    "2. Số đơn hàng nạp vào phân tích:",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=2000,
    help="Số lượng giao dịch thực tế nạp từ tệp để tiến hành khai phá."
)

# 3. Minimum Support
support_choices = [0.05, 0.03, 0.02, 0.01, 0.005]
selected_support = st.sidebar.select_slider(
    "3. Minimum Support (Tần suất tối thiểu):",
    options=support_choices,
    value=0.01,
    format_func=lambda v: f"{v:.1%} ({v})"
)

# 4. Minimum Confidence
min_conf_input = st.sidebar.slider(
    "4. Minimum Confidence (Độ tin cậy):",
    min_value=0.05,
    max_value=0.80,
    value=0.10,
    step=0.05,
    help="Xác suất có điều kiện P(Mua kèm B | Đã mua A)."
)

# 5. Minimum Lift
min_lift_input = st.sidebar.slider(
    "5. Minimum Lift (Độ nâng tối thiểu):",
    min_value=1.0,
    max_value=4.0,
    value=1.20,
    step=0.10,
    help="Độ tương quan thuận (Lift > 1 nghĩa là hai mặt hàng thúc đẩy nhau mua cùng)."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background:#F1F5F9; border:1px solid #CBD5E1; border-radius:6px; padding:10px 12px; font-size:0.86rem; color:#0F172A;">
      <b>💼 Ba Nghiệp Vụ Cốt Lõi:</b><br>
      • 🛍️ Gợi ý sản phẩm mua kèm (Recommender)<br>
      • 📜 Khám phá mạng lưới luật kết hợp<br>
      • 🌳 Phân tích tập phổ biến & Cây FP-Tree
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# NẠP DỮ LIỆU & TÍNH TOÁN CÁC CHỈ SỐ THỰC TẾ
# ==============================================================================
try:
    transactions = load_processed_transactions(str(selected_file), max_rows_input)
except Exception as exc:
    st.error(f"Lỗi khi đọc file dữ liệu: {exc}")
    st.stop()

total_tx = len(transactions)
all_items = [item for tx in transactions for item in tx]
unique_items = len(set(all_items))
avg_basket = (len(all_items) / total_tx) if total_tx > 0 else 0

# Khai phá luật và tập phổ biến thực tế từ dữ liệu
with st.spinner("Đang khai phá tập phổ biến & trích xuất luật kết hợp từ dữ liệu..."):
    rules_df, fi_df = extract_association_rules(
        str(selected_file), max_rows_input, selected_support, min_conf_input, min_lift_input
    )

max_lift_val = rules_df["lift"].max() if not rules_df.empty else 0.0

# ==============================================================================
# TOP KPI METRICS: HOÀN TOÀN TỪ KẾT QUẢ CHẠY THỰC TẾ
# ==============================================================================
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
kpi_c1.metric("📦 Đơn Hàng Thực Tế", f"{total_tx:,}", help="Tổng số giao dịch thực tế đang được phân tích.")
kpi_c2.metric("🏷️ Mặt Hàng Phân Biệt", f"{unique_items:,}", help="Số lượng chủng loại mặt hàng phân biệt trong tập đơn.")
kpi_c3.metric("🧺 Giỏ Hàng Trung Bình", f"{avg_basket:.1f} món", help="Số sản phẩm bình quân trên mỗi đơn hàng.")
kpi_c4.metric("🌳 Tập Phổ Biến", f"{len(fi_df):,}", help=f"Số tập mục có Support >= {selected_support:.1%}.")
kpi_c5.metric("📜 Luật Kết Hợp", f"{len(rules_df):,}", help=f"Số luật thỏa mãn Confidence >= {min_conf_input:.0%} & Lift >= {min_lift_input:.1f}x.")
kpi_c6.metric("📈 Max Lift Đạt Được", f"{max_lift_val:.2f}×" if max_lift_val > 0 else "N/A", help="Độ nâng cao nhất trong các luật tìm được.")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# HỆ THỐNG 3 TABS NGHIỆP VỤ CỐT LÕI
# ==============================================================================
tab_rec, tab_rules, tab_tree = st.tabs([
    "🛍️ NGHIỆP VỤ 1 — GỢI Ý SẢN PHẨM MUA KÈM (RECOMMENDER)",
    "📜 NGHIỆP VỤ 2 — KHÁM PHÁ & TRA CỨU LUẬT KẾT HỢP",
    "🌳 NGHIỆP VỤ 3 — PHÂN TÍCH TẬP PHỔ BIẾN & CÂY TIỀN TỐ FP-TREE"
])

# Lấy danh sách sản phẩm phổ biến thực tế từ dữ liệu
item_counts = pd.Series(all_items).value_counts()
top_popular_items = item_counts.head(100).index.tolist()


# ==============================================================================
# TAB 1: GỢI Ý SẢN PHẨM MUA KÈM (PRODUCT RECOMMENDER)
# ==============================================================================
with tab_rec:
    st.markdown("### 🛍️ Dự Đoán Sản Phẩm Mua Kèm Khi Khách Hàng Chọn Mua")
    st.markdown(
        "Giải quyết bài toán bán lẻ: **'Khi khách hàng bỏ sản phẩm này vào giỏ, hệ thống nên gợi ý những món gì đi kèm?'**. "
        "Mô hình đối chiếu các mặt hàng trong giỏ với tiền đề của các luật kết hợp thực tế, xếp hạng sản phẩm mua kèm "
        "dựa trên điểm tổng hợp $\\text{Score} = \\text{Lift} \\times \\text{Confidence}$."
    )

    # 1. Các giỏ hàng mẫu thực tế (1-Click Presets)
    st.markdown("##### ⚡ Thử Nhanh Các Giỏ Hàng Mẫu Phổ Biến:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    preset_chosen = None
    with col_p1:
        if st.button("🍓 Trái Cây & Quả Mọng", use_container_width=True):
            preset_chosen = ["Organic Raspberries"]
    with col_p2:
        if st.button("🥑 Salad & Rau Củ Tươi", use_container_width=True):
            preset_chosen = ["Organic Avocado"]
    with col_p3:
        if st.button("🍎 Táo & Trái Cây Tươi", use_container_width=True):
            preset_chosen = ["Organic Fuji Apple"]
    with col_p4:
        if st.button("🥛 Bữa Sáng Bổ Dưỡng", use_container_width=True):
            preset_chosen = ["Organic Whole Milk"]

    if preset_chosen:
        st.session_state["active_cart"] = preset_chosen
    elif "active_cart" not in st.session_state:
        st.session_state["active_cart"] = ["Organic Raspberries"]

    # 2. Bộ chọn giỏ hàng linh hoạt
    cart_items_selected = st.multiselect(
        "🧺 Chọn các sản phẩm hiện có trong giỏ hàng của khách hàng:",
        options=top_popular_items,
        default=[item for item in st.session_state["active_cart"] if item in top_popular_items],
        help="Có thể chọn một hoặc nhiều mặt hàng cùng lúc để nhận gợi ý kết hợp đa sản phẩm."
    )

    col_rec_topn, col_rec_filter = st.columns([1, 3])
    with col_rec_topn:
        top_n_rec = st.number_input("Số sản phẩm gợi ý tối đa:", min_value=1, max_value=12, value=5)
    with col_rec_filter:
        st.caption(
            f"📌 **Cấu hình trích xuất luật đang áp dụng:** Min Support = `{selected_support:.1%}` | "
            f"Min Confidence = `{min_conf_input:.0%}` | Min Lift = `{min_lift_input:.1f}x`. "
            f"(Có thể tinh chỉnh linh hoạt tại thanh bên trái)."
        )

    # 3. Xử lý gợi ý mua kèm từ luật thực tế
    if not cart_items_selected:
        st.warning("⚠️ Giỏ hàng hiện đang trống! Hãy chọn ít nhất một sản phẩm từ danh mục phía trên.")
    else:
        recommendations = recommend_products(cart_items_selected, rules_df, top_n=top_n_rec)

        if not recommendations:
            st.info(
                f"ℹ️ Chưa tìm thấy luật mua kèm nào thỏa mãn đồng thời Confidence $\\ge$ {min_conf_input:.0%} "
                f"và Lift $\\ge$ {min_lift_input:.1f}x cho nhóm sản phẩm này trong tập dữ liệu thực tế hiện tại.\n\n"
                f"👉 **Gợi ý thao tác:** Bạn hãy hạ ngưỡng `Minimum Confidence` xuống `5%` hoặc `Minimum Lift` xuống `1.1x` "
                f"ở thanh bên trái, hoặc thử chọn thêm các mặt hàng phổ biến khác vào giỏ."
            )
        else:
            st.success(f"🎉 Hệ thống đã tìm thấy **{len(recommendations)}** sản phẩm mua kèm phù hợp nhất từ dữ liệu thực tế!")

            for rank, rec in enumerate(recommendations, 1):
                # Phân loại chiến lược kinh doanh dựa trên Lift thực tế
                if rec["lift"] >= 2.5:
                    strategy_badge = "🎯 Combo Siêu Gắn Kết (High Affinity)"
                    strategy_action = "Nên đóng gói thành Combo ưu đãi tiết kiệm hoặc hiển thị Pop-up khuyến mãi tại bước thanh toán (Checkout)."
                elif rec["lift"] >= 1.8:
                    strategy_badge = "🛒 Bán Chéo Trực Tiếp (Cross-Selling)"
                    strategy_action = "Nên đề xuất tại mục 'Khách mua sản phẩm này thường mua cùng' ngay trên trang chi tiết sản phẩm."
                else:
                    strategy_badge = "🏬 Sắp Xếp Trưng Bày (Store Layout)"
                    strategy_action = "Nên bố trí 2 quầy hàng gần nhau để kích thích hành vi mua bốc đồng (Impulse buying)."

                st.markdown(
                    f"""
                    <div class="recommender-card">
                      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: {COLOR_NAVY};">
                          #{rank}. ⭐ {rec['item']}
                        </span>
                        <div style="margin-top: 4px;">
                          <span class="badge-tag badge-score">Điểm Score: {rec['score']:.2f}</span>
                          <span class="badge-tag">Xác suất mua kèm (Conf): {rec['confidence']:.1%}</span>
                          <span class="badge-tag badge-lift">Độ nâng (Lift): {rec['lift']:.2f}×</span>
                          <span class="badge-tag" style="background:#F1F5F9; color:#334155;">Tần suất (Supp): {rec['support']:.2%}</span>
                          <span class="badge-tag badge-action">{strategy_badge}</span>
                        </div>
                      </div>
                      <div style="color: {COLOR_TEXT}; font-size: 0.94rem; margin-top: 10px; font-weight: 500; line-height: 1.55;">
                        <b>💬 Giải thích nghiệp vụ:</b> Dựa trên <b>{total_tx:,}</b> đơn hàng thực tế: Khi khách hàng đã chọn 
                        <code>[{rec['antecedent']}]</code>, xác suất mua thêm <code>[{rec['item']}]</code> đạt 
                        <b>{rec['confidence']:.1%}</b>, cao gấp <b>{rec['lift']:.2f} lần</b> so với tần suất mua ngẫu nhiên thông thường.
                      </div>
                      <div style="color: {COLOR_TEXT_MUTED}; font-size: 0.86rem; margin-top: 6px; font-weight: 500;">
                        <b>💡 Hành động kinh doanh khuyến nghị:</b> {strategy_action}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Bảng tóm tắt các sản phẩm gợi ý
            st.markdown("#### 📋 Bảng Tổng Hợp Sản Phẩm Gợi Ý Mua Kèm")
            summary_recs = []
            for r in recommendations:
                summary_recs.append({
                    "Sản phẩm gợi ý": r["item"],
                    "Điểm Score (Lift*Conf)": round(r["score"], 2),
                    "Độ tin cậy Confidence (%)": f"{r['confidence']:.2%}",
                    "Độ nâng Lift (lần)": f"{r['lift']:.2f}×",
                    "Tần suất Support (%)": f"{r['support']:.2%}",
                    "Sản phẩm tiền đề": r["antecedent"]
                })
            st.dataframe(pd.DataFrame(summary_recs), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: KHÁM PHÁ & TRA CỨU LUẬT KẾT HỢP (ASSOCIATION RULES)
# ==============================================================================
with tab_rules:
    st.markdown("### 📜 Khám Phá & Tra Cứu Luật Kết Hợp (Association Rules)")
    st.markdown(
        "Khai phá mạng lưới quy luật ngầm giữa các sản phẩm $(A \\rightarrow B)$ trong hành vi mua sắm của khách hàng. "
        "Mỗi luật được định lượng bằng 3 chỉ số chuẩn: **Support** (tần suất combo), "
        "**Confidence** (độ tin cậy) và **Lift** (độ tương quan thúc đẩy)."
    )

    if rules_df.empty:
        st.warning("Không tìm thấy luật nào thỏa mãn các ngưỡng lọc hiện tại. Hãy thử hạ ngưỡng Confidence hoặc Lift ở thanh bên trái.")
    else:
        df_rules_disp = rules_df.copy()
        df_rules_disp["Tiền đề (Nếu mua A)"] = df_rules_disp["antecedents"].apply(lambda s: " + ".join(sorted(list(s))))
        df_rules_disp["Hệ quả (Sẽ mua kèm B)"] = df_rules_disp["consequents"].apply(lambda s: " + ".join(sorted(list(s))))
        df_rules_disp["Support (%)"] = df_rules_disp["support"].apply(lambda v: round(v * 100, 2))
        df_rules_disp["Confidence (%)"] = df_rules_disp["confidence"].apply(lambda v: round(v * 100, 2))
        df_rules_disp["Lift (Độ nâng)"] = df_rules_disp["lift"].apply(lambda v: round(v, 2))
        df_rules_disp["Score (Lift*Conf)"] = (df_rules_disp["lift"] * df_rules_disp["confidence"]).apply(lambda v: round(v, 2))

        # Tổng quan các chỉ số luật thực tế
        mean_conf = rules_df["confidence"].mean()
        mean_lift = rules_df["lift"].mean()
        strong_rules_count = len(rules_df[rules_df["lift"] >= 2.0])

        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        r_c1.metric("Tổng Số Luật Đạt Chuẩn", f"{len(rules_df):,}")
        r_c2.metric("Độ Tin Cậy Trung Bình", f"{mean_conf:.1%}")
        r_c3.metric("Độ Nâng Trung Bình", f"{mean_lift:.2f}×")
        r_c4.metric("Số Luật Tương Quan Mạnh (Lift ≥ 2.0)", f"{strong_rules_count:,}")

        # 1. Biểu đồ Scatter & Top 10 Luật
        col_plot_scatter, col_plot_bar = st.columns([3, 2])

        with col_plot_scatter:
            fig_scatter = px.scatter(
                df_rules_disp,
                x="Confidence (%)",
                y="Lift (Độ nâng)",
                size="Support (%)",
                color="Lift (Độ nâng)",
                color_continuous_scale="Teal",
                hover_data={
                    "Tiền đề (Nếu mua A)": True,
                    "Hệ quả (Sẽ mua kèm B)": True,
                    "Support (%)": ":.2f",
                    "Confidence (%)": ":.2f",
                    "Lift (Độ nâng)": ":.2f"
                },
                title="Biểu Đồ Phân Bố Luật: Độ Tin Cậy (Confidence) vs Độ Nâng (Lift)"
            )
            apply_plot_theme(fig_scatter, "Tương Quan Độ Tin Cậy vs Độ Nâng (Bóng = Support %)", "Confidence (%)", "Lift (Độ nâng)")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_plot_bar:
            top_lift_rules = df_rules_disp.sort_values(by="Lift (Độ nâng)", ascending=False).head(10).copy()
            top_lift_rules["Rule"] = top_lift_rules["Tiền đề (Nếu mua A)"] + " → " + top_lift_rules["Hệ quả (Sẽ mua kèm B)"]
            fig_bar_lift = px.bar(
                top_lift_rules,
                x="Lift (Độ nâng)",
                y="Rule",
                orientation="h",
                text="Lift (Độ nâng)",
                color="Lift (Độ nâng)",
                color_continuous_scale="Viridis",
                title="Top 10 Cặp Mặt Hàng Có Độ Gắn Kết (Lift) Cao Nhất"
            )
            fig_bar_lift.update_layout(yaxis=dict(autorange="reversed"))
            apply_plot_theme(fig_bar_lift, "Top 10 Cặp Mặt Hàng Có Độ Gắn Kết (Lift) Cao Nhất", "Lift (lần)", "")
            st.plotly_chart(fig_bar_lift, use_container_width=True)

        # 2. Bộ lọc tra cứu luật thông minh 2 chiều
        st.markdown("#### 🔍 Bộ Lọc & Tra Cứu Luật Kết Hợp")
        col_f_ante, col_f_conseq = st.columns(2)
        with col_f_ante:
            filter_ante = st.text_input("Lọc theo sản phẩm tiền đề (Nếu mua món A):", placeholder="Ví dụ: Banana, Avocado, Milk...")
        with col_f_conseq:
            filter_conseq = st.text_input("Lọc theo sản phẩm hệ quả (Sẽ mua kèm món B):", placeholder="Ví dụ: Strawberries, Spinach...")

        df_filtered_rules = df_rules_disp.copy()
        if filter_ante:
            df_filtered_rules = df_filtered_rules[df_filtered_rules["Tiền đề (Nếu mua A)"].str.contains(filter_ante, case=False, na=False)]
        if filter_conseq:
            df_filtered_rules = df_filtered_rules[df_filtered_rules["Hệ quả (Sẽ mua kèm B)"].str.contains(filter_conseq, case=False, na=False)]

        st.caption(f"Đang hiển thị **{len(df_filtered_rules):,}** / {len(rules_df):,} luật kết hợp phù hợp.")

        # 3. Bảng dữ liệu chi tiết
        cols_rules_show = [
            "Tiền đề (Nếu mua A)", "Hệ quả (Sẽ mua kèm B)",
            "Support (%)", "Confidence (%)", "Lift (Độ nâng)", "Score (Lift*Conf)"
        ]
        st.dataframe(
            df_filtered_rules[cols_rules_show].sort_values(by=["Score (Lift*Conf)", "Lift (Độ nâng)"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True
        )

        # 4. Tải file CSV
        csv_rules_data = df_filtered_rules[cols_rules_show].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Tải Danh Sách Luật Kết Hợp Đã Khai Phá (CSV)",
            data=csv_rules_data,
            file_name=f"association_rules_supp_{selected_support}_conf_{min_conf_input}.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 3: PHÂN TÍCH TẬP PHỔ BIẾN & CÂY TIỀN TỐ FP-TREE
# ==============================================================================
with tab_tree:
    st.markdown("### 🌳 Phân Tích Tập Phổ Biến (Frequent Patterns) & Cây Tiền Tố FP-Tree")
    st.markdown(
        "Khai phá tập phổ biến là nền tảng để phát hiện các nhóm sản phẩm thường xuyên xuất hiện cùng nhau. "
        "Thuật toán sử dụng **Cây tiền tố FP-Tree (Frequent Pattern Tree)** để nén toàn bộ tập giao dịch "
        "thành một cấu trúc cây gọn gàng trong bộ nhớ, loại bỏ hoàn toàn việc sinh tập ứng viên tổ hợp."
    )

    if not fi_df.empty:
        # Thống kê tập mục thực tế
        n_1_items = len(fi_df[fi_df["length"] == 1])
        n_multi_items = len(fi_df[fi_df["length"] >= 2])
        max_k_val = fi_df["length"].max()
        top_itemset_name = " + ".join(sorted(list(fi_df.iloc[0]["itemsets"])))
        top_itemset_supp = fi_df.iloc[0]["support"]

        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        f_c1.metric("Tổng Số Tập Phổ Biến", f"{len(fi_df):,}")
        f_c2.metric("Số Mặt Hàng Đơn (1-item)", f"{n_1_items:,}")
        f_c3.metric("Số Combo Mua Chung (≥ 2-item)", f"{n_multi_items:,}")
        f_c4.metric("Kích Thước Combo Dài Nhất", f"{max_k_val} món")

        # 1. Hai biểu đồ phân tích tập phổ biến
        col_tree_len, col_tree_top = st.columns([2, 3])

        with col_tree_len:
            len_dist_df = fi_df["length"].value_counts().sort_index().reset_index()
            len_dist_df.columns = ["Kích thước (k-itemset)", "Số lượng tập"]
            fig_len_dist = px.bar(
                len_dist_df,
                x="Kích thước (k-itemset)",
                y="Số lượng tập",
                text="Số lượng tập",
                color="Kích thước (k-itemset)",
                color_continuous_scale="Teal",
                title="Phân Bố Số Lượng Theo Độ Dài (k-Itemset)"
            )
            apply_plot_theme(fig_len_dist, "Phân Bố Số Lượng k-Itemset", "Độ dài tập mục (k)", "Số lượng tập")
            st.plotly_chart(fig_len_dist, use_container_width=True)

        with col_tree_top:
            top_fi_display = fi_df.head(15).copy()
            top_fi_display["Itemset"] = top_fi_display["itemsets"].apply(lambda s: " + ".join(sorted(list(s))))
            top_fi_display["Support (%)"] = top_fi_display["support"].apply(lambda v: round(v * 100, 2))

            fig_top_itemsets = px.bar(
                top_fi_display,
                x="Support (%)",
                y="Itemset",
                orientation="h",
                text="Support (%)",
                color="Support (%)",
                color_continuous_scale="Viridis",
                title=f"Top 15 Tập Phổ Biến Xuất Hiện Nhiều Nhất (Support ≥ {selected_support:.1%})"
            )
            fig_top_itemsets.update_layout(yaxis=dict(autorange="reversed"))
            apply_plot_theme(fig_top_itemsets, "Top 15 Tập Phổ Biến Xuất Hiện Nhiều Nhất", "Tần suất xuất hiện Support (%)", "")
            st.plotly_chart(fig_top_itemsets, use_container_width=True)

    # 2. Giải thích cơ chế nén của Cây Tiền Tố FP-Tree
    st.markdown("#### 🔬 Cơ Chế Nén & Khai Phá Của Cây Tiền Tố FP-Tree")
    
    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.markdown(
            f"""
            <div class="info-box">
              <h5 style="color:{COLOR_NAVY}; margin-top:0;">Bước 1: Quét CSDL & Sắp Xếp Tần Suất</h5>
              <p style="font-size:0.9rem; color:{COLOR_TEXT_MUTED}; margin-bottom:0;">
                Quét dữ liệu để tính tần suất xuất hiện của từng sản phẩm. 
                Lọc bỏ các món có support &lt; <code>{selected_support:.1%}</code> và sắp xếp các món đạt chuẩn 
                theo thứ tự tần suất giảm dần. Việc sắp xếp này đảm bảo các món xuất hiện nhiều nhất 
                sẽ nằm gần gốc cây nhất để tối đa hóa tỷ lệ chia sẻ tiền tố.
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_step2:
        st.markdown(
            f"""
            <div class="info-box">
              <h5 style="color:{COLOR_NAVY}; margin-top:0;">Bước 2: Xây Dựng Cây Tiền Tố (FP-Tree)</h5>
              <p style="font-size:0.9rem; color:{COLOR_TEXT_MUTED}; margin-bottom:0;">
                Chèn từng giỏ hàng vào cây tiền tố. Các giao dịch có chung những mặt hàng đầu tiên 
                sẽ dùng chung nhánh đường đi (Shared Prefix), chỉ cần tăng biến đếm tần suất <code>count</code>. 
                Hàng chục nghìn giao dịch đơn lẻ được nén gọn thành một cây phân cấp trong bộ nhớ.
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_step3:
        st.markdown(
            f"""
            <div class="info-box">
              <h5 style="color:{COLOR_NAVY}; margin-top:0;">Bước 3: Khai Phá Cây Điều Kiện</h5>
              <p style="font-size:0.9rem; color:{COLOR_TEXT_MUTED}; margin-bottom:0;">
                Sử dụng Bảng đầu mục (Header Table) và chuỗi liên kết nốt (Node-Links) để duyệt ngược từ 
                các món ở đáy cây lên đỉnh. Xây dựng Cây tiền tố điều kiện (Conditional FP-Tree) và đệ quy 
                trích xuất trực tiếp toàn bộ tập phổ biến mà không cần sinh tập ứng viên.
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 3. Bảng dữ liệu tra cứu tập phổ biến
    st.markdown("#### 📋 Bảng Chi Tiết Frequent Itemsets")
    search_fi_query = st.text_input("🔍 Tìm kiếm tập phổ biến theo tên món:", placeholder="Ví dụ: Banana, Avocado, Spinach, Milk...")
    
    df_fi_table = fi_df.copy()
    df_fi_table["Tập phổ biến (Itemset)"] = df_fi_table["itemsets"].apply(lambda s: " + ".join(sorted(list(s))))
    df_fi_table["Kích thước (k)"] = df_fi_table["length"]
    df_fi_table["Tần suất Support (%)"] = df_fi_table["support"].apply(lambda v: round(v * 100, 3))
    df_fi_table["Số đơn hàng chứa combo"] = df_fi_table["count"].apply(lambda v: f"{v:,} đơn")

    if search_fi_query:
        df_fi_table = df_fi_table[df_fi_table["Tập phổ biến (Itemset)"].str.contains(search_fi_query, case=False, na=False)]

    st.dataframe(
        df_fi_table[["Tập phổ biến (Itemset)", "Kích thước (k)", "Tần suất Support (%)", "Số đơn hàng chứa combo"]],
        use_container_width=True,
        hide_index=True
    )
