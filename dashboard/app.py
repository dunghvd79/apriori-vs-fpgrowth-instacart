"""
Dashboard: Gợi ý sản phẩm mua kèm trên Instacart bằng FP-Growth tự cài đặt (src/my_fpgrowth.py).

Ba khu vực chức năng chính:
  1. Gợi ý mua kèm (Recommender) — Chọn giỏ hàng, nhận gợi ý thời gian thực (luật kết hợp + dự phòng món bán chạy).
  2. Luật kết hợp (Rules Explorer) — Biểu đồ phân bố, 10 cặp món gắn kết nhất (đã gộp hai chiều A→B và B→A), tra cứu và tải CSV.
  3. Tập phổ biến (Frequent Itemsets) — Phân bố độ dài k-itemset, danh sách combo phổ biến nhất, tra cứu chi tiết.

Mọi số liệu đều được tính toán trực tiếp từ dữ liệu trong data/processed/. Các ngưỡng do người dùng tùy chỉnh ở thanh bên.
"""

import html
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN & IMPORT THUẬT TOÁN TỰ CÀI ĐẶT
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import 3 thành phần cốt lõi từ module thuật toán tự cài đặt src/my_fpgrowth.py
from my_fpgrowth import OptimizedFPGrowth, generate_rules, recommend_products  # noqa: E402

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="Gợi ý mua kèm · Instacart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="auto",
)

# ==============================================================================
# 2. HỆ THỐNG MÀU SẮC & GIAO DIỆN LIGHT ANALYTICS (QUẦY KỆ SIÊU THỊ)
# ==============================================================================
# Bảng màu lấy cảm hứng từ nông sản tươi sống và nhãn giá trên quầy kệ
PAPER = "#F7F9F4"   # Màu nền trang (trắng xám ngả giấy, dịu mắt, chống chói)
INK = "#1C2B22"     # Màu chữ chính (xanh đen đậm, độ tương phản cao nhất)
MUTED = "#55645A"   # Màu chữ phụ và ghi chú (xám rêu)
RULE = "#D6DED2"    # Màu đường kẻ phân cách sắc nét
KALE = "#2F6B42"    # Màu xanh cải Kale (màu chủ đạo của ứng dụng)
CARROT = "#E2711D"  # Màu cam cà rốt (dùng làm điểm nhấn cho chỉ số Lift và gợi ý trúng)
BEET = "#7E2E58"    # Màu củ dền (dùng cho chuỗi dữ liệu thứ hai trên biểu đồ)
SEED = 42           # Hạt giống ngẫu nhiên cố định để kết quả chia train/test luôn nhất quán

# Áp dụng CSS tùy chỉnh để làm đẹp giao diện và đảm bảo văn bản hiển thị rõ nét
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Public+Sans:wght@400;500;600;700&display=swap');

      .stApp {{ background: {PAPER}; }}
      .stApp, .stApp p, .stApp li, .stApp label, .stApp input {{
          font-family: 'Public Sans', system-ui, sans-serif;
          color: {INK};
          font-variant-numeric: tabular-nums;
      }}
      .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
          font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif;
          color: {INK};
          letter-spacing: -0.01em;
      }}
      [data-testid="stSidebar"] {{ background: #EEF2EA; border-right: 1px solid {RULE}; }}
      [data-testid="stBaseButton-primary"] p {{ color: #FFFFFF; font-weight: 600; }}
      [data-testid="stBaseButton-secondary"] {{ background: #FFFFFF; border-color: {RULE}; }}
      [data-testid="stBaseButton-secondary"]:hover {{ border-color: {KALE}; }}

      /* Giỏ hàng: Cho phép các thẻ tên sản phẩm tự xuống dòng thay vì cuộn ngang */
      [data-testid="stMultiSelect"] div[role="group"]:has(> [data-testid="stMultiSelectTagsContainer"]) {{
          height: auto; max-height: none; min-height: 40px; padding-top: 4px; padding-bottom: 4px;
      }}
      [data-testid="stMultiSelectTagsContainer"] {{
          flex-wrap: wrap; overflow: visible; height: auto; max-height: none;
          flex: 1 1 0; min-width: 0; mask-image: none; -webkit-mask-image: none;
      }}
      [data-testid="stMultiSelectTagsContainer"] > span[role="group"] {{
          display: flex; flex-wrap: wrap; gap: 6px; width: 100%; min-width: 0;
      }}
      [data-testid="stMultiSelectTagsContainer"] [data-tag] {{ max-width: 100%; height: auto; min-height: 28px; }}
      [data-testid="stMultiSelectTagsContainer"] [data-tag] > span {{
          white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.3; padding: 3px 0;
      }}

      /* Tiêu đề đầu trang */
      .masthead {{ display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; margin: 0 0 4px 0; }}
      .masthead h1 {{ font-size: 2.1rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.1; }}
      .masthead p {{ margin: 0; color: {MUTED}; font-size: 0.98rem; max-width: 62ch; }}

      /* Dải thông số tổng quan đầu trang */
      .facts {{ display: flex; flex-wrap: wrap; gap: 6px 28px; padding: 10px 0 14px 0;
               border-bottom: 1px solid {RULE}; margin-bottom: 8px; }}
      .fact b {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 700; color: {KALE}; }}
      .fact span {{ color: {MUTED}; font-size: 0.88rem; margin-left: 4px; }}

      /* Định kiểu thanh tab */
      button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 600; }}
      button[data-baseweb="tab"][aria-selected="true"] p {{ color: {KALE}; }}
      [data-baseweb="tab-highlight"] {{ background-color: {KALE}; }}

      /* Nhãn giá trên kệ (Shelf Tag) — Hiển thị sản phẩm gợi ý như nhãn giá siêu thị */
      .shelf-tag {{
          display: grid; grid-template-columns: 1fr auto; gap: 2px 18px; align-items: start;
          background: #FFFFFF; border: 1.5px solid {INK}; border-radius: 4px;
          padding: 14px 18px 12px 18px; margin-bottom: 12px; position: relative;
      }}
      .shelf-tag::before {{   /* Lỗ xỏ dây nhãn giá */
          content: ""; position: absolute; left: -7px; top: 50%; width: 12px; height: 12px;
          margin-top: -6px; border-radius: 50%; background: {PAPER}; border: 1.5px solid {INK};
      }}
      .shelf-tag .rank {{ color: {MUTED}; font-size: 0.8rem; font-weight: 600; }}
      .shelf-tag .name {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.3rem;
                          font-weight: 700; line-height: 1.2; color: {INK}; }}
      .shelf-tag .lift {{ grid-row: 1 / span 2; grid-column: 2; text-align: right;
                          font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
                          font-size: 2.3rem; line-height: 1; color: {CARROT}; }}
      .shelf-tag .lift small {{ display: block; font-family: 'Public Sans', sans-serif; font-size: 0.72rem;
                                font-weight: 600; color: {MUTED}; margin-top: 4px; }}
      .shelf-tag .why {{ grid-column: 1 / span 2; color: {INK}; font-size: 0.93rem; line-height: 1.5;
                         margin-top: 8px; padding-top: 8px; border-top: 1px dashed {RULE}; }}
      .shelf-tag .fine {{ grid-column: 1 / span 2; color: {MUTED}; font-size: 0.82rem; margin-top: 4px; }}
      .shelf-tag.fallback {{ border-style: dashed; border-color: {MUTED}; background: transparent; }}
      .shelf-tag.fallback .lift {{ color: {MUTED}; font-size: 1rem; font-weight: 600; }}
      .shelf-tag.hit {{ outline: 3px solid {KALE}; outline-offset: 2px; }}

      .hidden-note {{ border-left: 4px solid {CARROT}; background: #FFFFFF; padding: 10px 14px;
                      margin: 6px 0 14px 0; font-size: 0.93rem; }}
      .empty-cart {{ color: {MUTED}; padding: 28px 0; font-size: 1rem; }}

      @media (prefers-reduced-motion: no-preference) {{
          .shelf-tag {{ animation: hang 260ms ease-out both; transform-origin: 0 50%; }}
          @keyframes hang {{ from {{ transform: rotate(-1.5deg); opacity: 0; }} to {{ transform: none; opacity: 1; }} }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 3. HÀM TIỆN ÍCH TRÌNH BÀY & XỬ LÝ CHUỖI
# ==============================================================================
def style_fig(fig: go.Figure, x_title: str = "", y_title: str = "", height: int = 380) -> go.Figure:
    """
    Mục đích: Chuẩn hóa giao diện cho toàn bộ biểu đồ Plotly trong ứng dụng.
    Áp dụng nền trắng, đường kẻ nhạt, phông chữ đồng bộ với hệ thống.
    """
    fig.update_layout(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color=INK, size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
        showlegend=False,
        xaxis=dict(title=x_title, gridcolor="#EDF1EA", zeroline=False, linecolor=RULE),
        yaxis=dict(title=y_title, gridcolor="#EDF1EA", zeroline=False, linecolor=RULE),
        hoverlabel=dict(bgcolor="#FFFFFF", font_family="Public Sans"),
    )
    return fig


def esc(text) -> str:
    """Mục đích: Escape các ký tự đặc biệt trong HTML để tránh lỗi hiển thị trên giao diện."""
    return html.escape(str(text))


def join_items(s) -> str:
    """Mục đích: Chuyển một tập hợp (frozenset/set) các món thành chuỗi hiển thị 'Món A + Món B'."""
    return " + ".join(sorted(s))


# ==============================================================================
# 4. CÁC HÀM XỬ LÝ DỮ LIỆU & THUẬT TOÁN CỐT LÕI
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_transactions(file_path: str, max_rows: int = 0) -> List[List[str]]:
    """
    Mục đích: Đọc dữ liệu giao dịch từ file Parquet hoặc CSV đã tiền xử lý.
    - File Parquet: Nạp siêu tốc danh sách các món trong cột 'items'.
    - File CSV: Tách chuỗi danh sách món phân cách bằng ký tự ' | '.
    - max_rows: Cắt lấy N đơn hàng đầu tiên để phục vụ phân tích theo quy mô.
    - @st.cache_data: Lưu kết quả vào RAM để không phải đọc lại ổ đĩa nhiều lần.
    """
    path = Path(file_path)
    if path.suffix.lower() == ".parquet":
        txs = [list(items) for items in pd.read_parquet(path, columns=["items"])["items"]]
    else:
        txs = [str(v).split(" | ") for v in pd.read_csv(path)["items"]]
    return txs[:max_rows] if max_rows > 0 else txs


@st.cache_data(show_spinner=False)
def mine(file_path: str, max_rows: int, min_support: float, min_conf: float, min_lift: float):
    """
    Mục đích: Hàm cầu nối trung tâm để khai phá tập phổ biến và sinh luật kết hợp.
    
    Quy trình:
      1. Nạp danh sách đơn hàng qua load_transactions.
      2. Khởi tạo OptimizedFPGrowth, gọi .fit() để dựng cây FP-Tree và .get_frequent_itemsets().
      3. Gọi generate_rules() để trích xuất toàn bộ luật kết hợp đạt ngưỡng min_conf và min_lift.
      
    Tại sao dùng @st.cache_data?
      Streamlit luôn tự động chạy lại toàn bộ script khi người dùng tương tác trên web.
      Dùng cache giúp ghi nhớ kết quả ứng với bộ 5 tham số đầu vào. Khi người dùng thao tác ở giỏ hàng
      mà không đổi tham số ở sidebar, hàm mine() trả về kết quả ngay lập tức trong 0.0001s từ RAM.
    """
    txs = load_transactions(file_path, max_rows)
    fi_df = OptimizedFPGrowth(min_support=min_support).fit(txs).get_frequent_itemsets()
    rules_df = generate_rules(fi_df, min_confidence=min_conf, min_lift=min_lift)
    return fi_df, rules_df


def popular_ranking(txs: List[List[str]]) -> pd.Series:
    """
    Mục đích: Đếm tổng tần suất xuất hiện của tất cả sản phẩm trên toàn bộ tập đơn hàng.
    Kết quả được dùng làm phương án dự phòng (fallback) khi một giỏ hàng không kích hoạt được luật nào.
    """
    return pd.Series([i for t in txs for i in t]).value_counts()


def recommend_with_fallback(cart: List[str], rules_df: pd.DataFrame, popular: List[str], top_n: int) -> List[Dict]:
    """
    Mục đích: Chiến lược gợi ý Hybrid (kết hợp Luật kết hợp + Món bán chạy nhất).
    - Ưu tiên 1: Gợi ý theo luật từ hàm recommend_products (gắn nhãn source="rule").
    - Ưu tiên 2: Nếu số luật kích hoạt được ít hơn top_n, tự động bù thêm các món bán chạy nhất
      chưa có trong giỏ (gắn nhãn source="popular") để luôn đảm bảo đủ số lượng gợi ý cho khách.
    """
    recs = [dict(r, source="rule") for r in recommend_products(cart, rules_df, top_n=top_n)]
    taken = set(cart) | {r["item"] for r in recs}
    for item in popular:
        if len(recs) >= top_n:
            break
        if item not in taken:
            recs.append({"item": item, "source": "popular"})
            taken.add(item)
    return recs



def dedupe_pairs(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mục đích: Gộp hai chiều luật A→B và B→A thành một cặp món duy nhất A ⇄ B.
    
    Bản chất toán học:
      - Lift(A→B) = P(A ∩ B) / (P(A) * P(B)) = Lift(B→A). Lift của 2 chiều luôn luôn bằng nhau!
      - Confidence của 2 chiều lại khác nhau vì mẫu số khác nhau: Conf(A→B) = P(A ∩ B) / P(A).
    
    Tại sao cần gộp?
      Khi vẽ biểu đồ Top 10 cặp món gắn kết nhất, nếu không gộp thì biểu đồ sẽ bị vẽ lặp 2 thanh giống hệt nhau
      cho cùng 1 cặp món. Hàm này gộp lại thành 1 dòng, lưu cả Confidence chiều 1 và chiều 2 để hiển thị trên tooltip.
    """
    rows = {}
    for r in rules_df.itertuples(index=False):
        key = frozenset([r.antecedents, r.consequents])
        entry = rows.setdefault(key, {
            "Cặp món": f"{join_items(r.antecedents)}  ⇄  {join_items(r.consequents)}",
            "Lift": r.lift, "Support": r.support, "Chiều 1": None, "Chiều 2": None,
        })
        text = f"{join_items(r.antecedents)} → {join_items(r.consequents)}: {r.confidence:.0%}"
        entry["Chiều 1" if entry["Chiều 1"] is None else "Chiều 2"] = text
    return pd.DataFrame(rows.values()).sort_values("Lift", ascending=False).reset_index(drop=True)


# ==============================================================================
# 5. THANH BÊN (SIDEBAR): ĐIỀU KHIỂN DỮ LIỆU & THAM SỐ
# ==============================================================================
st.sidebar.markdown("### Dữ liệu và ngưỡng")

# 1. Tìm các tệp dữ liệu đã tiền xử lý sẵn trong data/processed/
files = sorted(DATA_DIR.glob("*.parquet")) if DATA_DIR.exists() else []
if not files:
    files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
if not files:
    st.error("Chưa có dữ liệu trong `data/processed/`. Chạy `python src/preprocess.py` rồi tải lại trang.")
    st.stop()


def file_label(f: Path) -> str:
    """Tạo nhãn hiển thị dễ hiểu cho tệp: số đơn và phân loại (sản phẩm hay kệ hàng)."""
    level = "kệ hàng" if "aisle" in f.stem else "sản phẩm"
    size = f.stem.split("_")[-1]
    return f"{size} đơn · mức {level}"


default_file = next((i for i, f in enumerate(files) if "product_10k" in f.name), 0)
selected_file = st.sidebar.selectbox("Bộ dữ liệu", files, index=default_file, format_func=file_label)
file_total = len(load_transactions(str(selected_file)))

# 2. Điều chỉnh số lượng đơn hàng nạp vào bộ nhớ
max_rows = st.sidebar.number_input(
    "Số đơn đưa vào khai phá", min_value=1000, max_value=file_total,
    value=min(10000, file_total), step=1000,
    help=f"Lấy N đơn đầu tiên của bộ dữ liệu (tối đa {file_total:,}).",
)

# 3. Chọn ngưỡng Minimum Support
min_support = st.sidebar.select_slider(
    "Support tối thiểu", options=[0.05, 0.03, 0.02, 0.01, 0.005, 0.002], value=0.01,
    format_func=lambda v: f"{v:.1%}",
    help="Tỉ lệ đơn tối thiểu phải chứa một tập món để tập đó được coi là phổ biến.",
)

# 4. Chọn ngưỡng Minimum Confidence
min_conf = st.sidebar.slider(
    "Confidence tối thiểu", 0.05, 0.80, 0.10, 0.05, format="%.2f",
    help="P(mua B | đã mua A). Luật có confidence thấp hơn bị bỏ.",
)

# 5. Chọn ngưỡng Minimum Lift
min_lift = st.sidebar.slider(
    "Lift tối thiểu", 1.0, 4.0, 1.2, 0.1, format="%.1f",
    help="Lift > 1: mua A làm khả năng mua B cao hơn mức bình thường.",
)
st.sidebar.caption("Support thấp cho nhiều luật hơn nhưng khai phá lâu hơn. Kết quả được lưu đệm sau lần chạy đầu.")


# ==============================================================================
# 6. THỰC THI KHAI PHÁ & HIỂN THỊ DẢI SỐ LIỆU ĐẦU TRANG
# ==============================================================================
transactions = load_transactions(str(selected_file), max_rows)
with st.spinner("Đang dựng FP-Tree và khai phá luật…"):
    fi_df, rules_df = mine(str(selected_file), max_rows, min_support, min_conf, min_lift)

popular = popular_ranking(transactions)
popular_list = popular.index.tolist()
n_tx = len(transactions)
avg_basket = sum(len(t) for t in transactions) / n_tx if n_tx else 0

# Header chính và dải thông số thống kê tổng quan (Facts Banner)
st.markdown(
    f"""
    <div class="masthead">
      <h1>Khách bỏ món này vào giỏ, nên gợi ý món gì?</h1>
      <p>Luật kết hợp khai phá bằng FP-Growth tự cài đặt trên đơn hàng Instacart.</p>
    </div>
    <div class="facts">
      <div class="fact"><b>{n_tx:,}</b><span>đơn hàng</span></div>
      <div class="fact"><b>{len(popular):,}</b><span>mặt hàng</span></div>
      <div class="fact"><b>{avg_basket:.1f}</b><span>món mỗi giỏ</span></div>
      <div class="fact"><b>{len(fi_df):,}</b><span>tập phổ biến (support ≥ {min_support:.1%})</span></div>
      <div class="fact"><b>{len(rules_df):,}</b><span>luật (conf ≥ {min_conf:.0%}, lift ≥ {min_lift:.1f})</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Khởi tạo 3 Tab chức năng
tab_rec, tab_rules, tab_fi = st.tabs(["Gợi ý mua kèm", "Luật kết hợp", "Tập phổ biến"])


# ==============================================================================
# 7. TAB 1: GỢI Ý MUA KÈM (LIVE RECOMMENDER)
# ==============================================================================
# Lấy ra 4 món xuất hiện nhiều nhất ở vế "nếu mua" để làm nút bấm chọn nhanh (Quick picks)
ante_counts = pd.Series([i for a in rules_df["antecedents"] for i in a]).value_counts() if not rules_df.empty else pd.Series(dtype=int)
quick_picks = ante_counts.head(4).index.tolist()

if "cart" not in st.session_state or not set(st.session_state["cart"]) <= set(popular_list):
    st.session_state["cart"] = quick_picks[:1]
    st.session_state["hidden"] = None


def set_cart(items: List[str], hidden=None):
    """Cập nhật giỏ hàng trong session_state khi người dùng bấm nút chọn nhanh."""
    st.session_state["cart"] = items
    st.session_state["hidden"] = hidden


def pick_real_order():
    """Lấy ngẫu nhiên một đơn hàng thật trong CSDL, giấu đi 1 món để thử thách khả năng dự đoán của hệ thống."""
    rng = random.Random()
    basket = list(rng.choice(transactions))
    hidden = basket.pop(rng.randrange(len(basket)))
    set_cart(basket, hidden)


with tab_rec:
    col_cart, col_tags = st.columns([2, 3], gap="large")

    # Cột trái: Quản lý giỏ hàng của người dùng
    with col_cart:
        st.markdown("#### Giỏ hàng")
        if quick_picks:
            st.caption("Chọn nhanh một món có nhiều luật nhất:")
            qcols = st.columns(2)
            for i, item in enumerate(quick_picks):
                qcols[i % 2].button(item, key=f"qp_{i}", on_click=set_cart, args=([item],), width="stretch")
        st.button(
            "Lấy một đơn hàng thật và giấu 1 món", on_click=pick_real_order, width="stretch", type="primary",
            help="Lấy ngẫu nhiên một đơn trong dữ liệu, giấu đi một món rồi xem hệ thống có đoán ra món đó không.",
        )
        st.multiselect(
            "Các món trong giỏ", options=popular_list, key="cart",
            placeholder="Gõ tên sản phẩm, ví dụ Banana",
            on_change=lambda: st.session_state.update(hidden=None),
        )
        top_n = st.slider("Số gợi ý", 1, 10, 5)

    cart = st.session_state["cart"]
    hidden = st.session_state.get("hidden")

    # Cột phải: Trình bày danh sách gợi ý dưới dạng nhãn giá kệ hàng siêu thị (Shelf Tags)
    with col_tags:
        st.markdown("#### Gợi ý cho giỏ này")
        if not cart:
            st.markdown('<div class="empty-cart">Thêm ít nhất một món vào giỏ để xem gợi ý.</div>', unsafe_allow_html=True)
        else:
            # Gọi hàm gợi ý kết hợp luật + fallback món bán chạy
            recs = recommend_with_fallback(cart, rules_df, popular_list, top_n)
            n_rule = sum(r["source"] == "rule" for r in recs)
            
            # Nếu đang ở chế độ thử nghiệm giấu 1 món: Thông báo xem món bị giấu có nằm trong gợi ý không
            if hidden is not None:
                hit = any(r["item"] == hidden for r in recs)
                verdict = "nằm trong danh sách gợi ý." if hit else "không nằm trong danh sách gợi ý."
                st.markdown(
                    f'<div class="hidden-note">Món bị giấu của đơn này là <b>{esc(hidden)}</b>, và {verdict}</div>',
                    unsafe_allow_html=True,
                )
            if n_rule == 0:
                st.caption(
                    "Không có luật nào có vế trái nằm trong giỏ này với ngưỡng hiện tại, "
                    "nên dưới đây là các món bán chạy. Hạ support hoặc confidence để có thêm luật."
                )
            elif n_rule < len(recs):
                st.caption(f"Chỉ có {n_rule} gợi ý từ luật; các vị trí còn lại là món bán chạy (viền nét đứt).")
            
            # Dựng nhãn thẻ HTML cho từng sản phẩm gợi ý
            tags = []
            for rank, r in enumerate(recs, 1):
                is_hit = " hit" if hidden is not None and r["item"] == hidden else ""
                if r["source"] == "rule":
                    # Nhãn gợi ý từ luật kết hợp thực tế (có đầy đủ Lift, Confidence, Support, diễn giải)
                    tags.append(f"""
                    <div class="shelf-tag{is_hit}">
                      <div class="rank">Gợi ý {rank}</div>
                      <div class="lift">×{r['lift']:.2f}<small>lift</small></div>
                      <div class="name">{esc(r['item'])}</div>
                      <div class="why">Trong các đơn có <b>{esc(r['antecedent'])}</b>,
                        <b>{r['confidence']:.1%}</b> đơn mua thêm <b>{esc(r['item'])}</b>, cao gấp
                        {r['lift']:.2f} lần tỉ lệ mua món này nói chung.</div>
                      <div class="fine">{r['support']:.2%} số đơn chứa cả hai · score (lift × confidence) {r['score']:.2f}</div>
                    </div>""")
                else:
                    # Nhãn gợi ý dự phòng từ danh sách bán chạy (fallback)
                    share = popular[r["item"]] / n_tx
                    tags.append(f"""
                    <div class="shelf-tag fallback{is_hit}">
                      <div class="rank">Gợi ý {rank}</div>
                      <div class="lift">bán chạy</div>
                      <div class="name">{esc(r['item'])}</div>
                      <div class="fine">Có mặt trong {share:.1%} số đơn.</div>
                    </div>""")
            st.markdown("".join(tags), unsafe_allow_html=True)


# ==============================================================================
# 8. TAB 2: LUẬT KẾT HỢP (ASSOCIATION RULES EXPLORER)
# ==============================================================================
with tab_rules:
    if rules_df.empty:
        st.info("Không có luật nào đạt ngưỡng hiện tại. Hạ support, confidence hoặc lift ở thanh bên.")
    else:
        # Gộp hai chiều A->B và B->A để phục vụ biểu đồ cặp món
        pairs = dedupe_pairs(rules_df)
        st.markdown(
            f"**{len(rules_df):,} luật** ứng với **{len(pairs):,} cặp món**. "
            f"Luật A→B và B→A luôn có cùng lift nên biểu đồ bên phải gộp thành một cặp."
        )
        c1, c2 = st.columns([3, 2], gap="large")
        
        # Biểu đồ Scatter Plot: Phân bố Confidence vs Lift
        with c1:
            st.markdown("##### Confidence và lift của từng luật")
            disp = rules_df.assign(
                rule=rules_df["antecedents"].map(join_items) + " → " + rules_df["consequents"].map(join_items),
                conf_pct=rules_df["confidence"] * 100,
                supp_pct=rules_df["support"] * 100,
            )
            fig = px.scatter(
                disp, x="conf_pct", y="lift", size="supp_pct", size_max=26,
                hover_name="rule", hover_data={"conf_pct": ":.1f", "lift": ":.2f", "supp_pct": ":.2f"},
                labels={"conf_pct": "Confidence (%)", "lift": "Lift", "supp_pct": "Support (%)"},
            )
            fig.update_traces(marker=dict(color=KALE, opacity=0.65, line=dict(color="#FFFFFF", width=1)))
            st.plotly_chart(style_fig(fig, "Confidence (%)", "Lift"), config={"displayModeBar": False}, use_container_width=True)
            st.caption("Kích thước điểm là support: luật càng to càng xuất hiện nhiều.")
        
        # Biểu đồ Bar Chart: 10 cặp món có độ gắn kết (Lift) cao nhất
        with c2:
            st.markdown("##### 10 cặp món gắn kết nhất")
            top_pairs = pairs.head(10)
            fig = px.bar(top_pairs, x="Lift", y="Cặp món", orientation="h", text="Lift",
                         hover_data={"Chiều 1": True, "Chiều 2": True, "Lift": ":.2f"})
            fig.update_traces(marker_color=CARROT, texttemplate="×%{x:.2f}", textposition="inside",
                              insidetextanchor="end", textfont=dict(color="#FFFFFF"))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, "Lift", ""), config={"displayModeBar": False}, use_container_width=True)

        # Bảng tra cứu luật 2 chiều và xuất file CSV
        st.markdown("##### Tra cứu luật")
        f1, f2 = st.columns(2)
        q_a = f1.text_input("Nếu khách mua", placeholder="Ví dụ: Avocado")
        q_b = f2.text_input("thì mua kèm", placeholder="Ví dụ: Banana")
        table = pd.DataFrame({
            "Nếu mua": rules_df["antecedents"].map(join_items),
            "Thì mua kèm": rules_df["consequents"].map(join_items),
            "Support (%)": (rules_df["support"] * 100).round(2),
            "Confidence (%)": (rules_df["confidence"] * 100).round(1),
            "Lift": rules_df["lift"].round(2),
        })
        if q_a:
            table = table[table["Nếu mua"].str.contains(q_a, case=False, regex=False)]
        if q_b:
            table = table[table["Thì mua kèm"].str.contains(q_b, case=False, regex=False)]
        st.caption(f"{len(table):,} / {len(rules_df):,} luật")
        st.dataframe(table, width="stretch", hide_index=True, height=360)
        st.download_button(
            "Tải các luật đang hiển thị (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"luat_{selected_file.stem}_s{min_support}_c{min_conf}_l{min_lift}.csv", mime="text/csv",
        )


# ==============================================================================
# 9. TAB 3: TẬP PHỔ BIẾN (FREQUENT PATTERNS & FP-TREE)
# ==============================================================================
with tab_fi:
    if fi_df.empty:
        st.info("Không có tập phổ biến nào ở mức support này. Hạ support ở thanh bên.")
    else:
        combos = fi_df[fi_df["length"] >= 2]
        st.markdown(
            f"**{len(fi_df):,} tập phổ biến**: {(fi_df['length'] == 1).sum():,} món đơn lẻ và "
            f"{len(combos):,} combo từ 2 món trở lên, dài nhất {fi_df['length'].max()} món."
        )
        c1, c2 = st.columns([2, 3], gap="large")
        
        # Biểu đồ phân bố độ dài k-itemset
        with c1:
            st.markdown("##### Số tập theo độ dài")
            dist = fi_df["length"].value_counts().sort_index().rename_axis("k").reset_index(name="n")
            fig = px.bar(dist, x="k", y="n", text="n")
            fig.update_traces(marker_color=KALE, textposition="outside", cliponaxis=False)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(style_fig(fig, "Số món trong tập (k)", "Số tập"), config={"displayModeBar": False}, use_container_width=True)
        
        # Biểu đồ Top các combo xuất hiện nhiều nhất trong siêu thị
        with c2:
            st.markdown("##### Combo xuất hiện nhiều nhất")
            if combos.empty:
                st.caption("Chưa có combo nào ở mức support này.")
            else:
                top = combos.head(12).assign(label=lambda d: d["itemsets"].map(join_items), pct=lambda d: d["support"] * 100)
                fig = px.bar(top, x="pct", y="label", orientation="h", text="count",
                             hover_data={"pct": ":.2f", "count": True})
                fig.update_traces(marker_color=BEET, texttemplate="%{text:,} đơn", textposition="inside",
                                  insidetextanchor="end", textfont=dict(color="#FFFFFF"))
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(style_fig(fig, "Support (%)", ""), config={"displayModeBar": False}, use_container_width=True)

        # Hộp mở rộng giải thích nguyên lý hoạt động 3 bước của FP-Tree
        with st.expander("FP-Growth tìm các tập này như thế nào"):
            st.markdown(
                f"1. Quét dữ liệu lần 1, đếm số đơn chứa từng món, bỏ món có support dưới {min_support:.1%} "
                "và sắp các món còn lại theo tần suất giảm dần.\n"
                "2. Quét lần 2, chèn từng giỏ vào FP-Tree. Các giỏ có chung tiền tố dùng chung một nhánh, "
                "chỉ tăng bộ đếm, nên cả tập đơn được nén thành một cây.\n"
                "3. Đi từ món ít phổ biến nhất trong bảng header, gom các nhánh chứa món đó thành cây điều kiện "
                "rồi khai phá đệ quy. Không cần sinh tập ứng viên như Apriori."
            )

        # Bảng tra cứu danh sách tập phổ biến
        st.markdown("##### Tra cứu tập phổ biến")
        q = st.text_input("Tìm theo tên món", placeholder="Ví dụ: Spinach")
        fi_table = pd.DataFrame({
            "Tập món": fi_df["itemsets"].map(join_items),
            "k": fi_df["length"],
            "Số đơn": fi_df["count"],
            "Support (%)": (fi_df["support"] * 100).round(3),
        })
        if q:
            fi_table = fi_table[fi_table["Tập món"].str.contains(q, case=False, regex=False)]
        st.dataframe(fi_table, width="stretch", hide_index=True, height=360)
