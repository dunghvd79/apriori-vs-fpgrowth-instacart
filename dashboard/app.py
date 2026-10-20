"""
Dashboard: Gợi ý sản phẩm mua kèm trên Instacart bằng FP-Growth tự cài đặt (src/my_fpgrowth.py).

Bốn khu vực:
  1. Gợi ý mua kèm  — chọn giỏ hàng, nhận gợi ý (luật kết hợp + dự phòng món bán chạy).
  2. Luật kết hợp    — phân bố, cặp món gắn kết nhất (đã gộp chiều A→B / B→A), tra cứu.
  3. Tập phổ biến    — phân bố độ dài, combo phổ biến nhất, tra cứu.
  4. Đánh giá        — holdout 80/20: giấu 1 món mỗi đơn, đo tỉ lệ đoán trúng trong top-k.

Mọi số liệu đều tính từ file trong data/processed/. Các ngưỡng do người dùng chọn ở thanh bên.
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

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from my_fpgrowth import OptimizedFPGrowth, generate_rules, recommend_products  # noqa: E402

st.set_page_config(
    page_title="Gợi ý mua kèm · Instacart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="auto",
)

# ==============================================================================
# BẢNG MÀU & KIỂU CHỮ — lấy cảm hứng từ quầy rau củ và nhãn giá trên kệ
# ==============================================================================
PAPER = "#F7F9F4"   # nền giấy
INK = "#1C2B22"     # chữ chính
MUTED = "#55645A"   # chữ phụ
RULE = "#D6DED2"    # đường kẻ
KALE = "#2F6B42"    # màu chủ đạo
CARROT = "#E2711D"  # chỉ dùng cho con số lift / điểm nhấn
BEET = "#7E2E58"    # chuỗi dữ liệu thứ hai trong biểu đồ
SEED = 42

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

      /* Giỏ hàng: các món xuống dòng thay vì cuộn ngang, để thấy hết cả giỏ */
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

      /* Đầu trang */
      .masthead {{ display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; margin: 0 0 4px 0; }}
      .masthead h1 {{ font-size: 2.1rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.1; }}
      .masthead p {{ margin: 0; color: {MUTED}; font-size: 0.98rem; max-width: 62ch; }}

      /* Dải số liệu: một hàng chữ, không đóng thẻ */
      .facts {{ display: flex; flex-wrap: wrap; gap: 6px 28px; padding: 10px 0 14px 0;
               border-bottom: 1px solid {RULE}; margin-bottom: 8px; }}
      .fact b {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 700; color: {KALE}; }}
      .fact span {{ color: {MUTED}; font-size: 0.88rem; margin-left: 4px; }}

      /* Tabs */
      button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 600; }}
      button[data-baseweb="tab"][aria-selected="true"] p {{ color: {KALE}; }}
      [data-baseweb="tab-highlight"] {{ background-color: {KALE}; }}

      /* Nhãn giá trên kệ — điểm nhấn duy nhất của trang */
      .shelf-tag {{
          display: grid; grid-template-columns: 1fr auto; gap: 2px 18px; align-items: start;
          background: #FFFFFF; border: 1.5px solid {INK}; border-radius: 4px;
          padding: 14px 18px 12px 18px; margin-bottom: 12px; position: relative;
      }}
      .shelf-tag::before {{   /* lỗ xỏ nhãn */
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


def style_fig(fig: go.Figure, x_title: str = "", y_title: str = "", height: int = 380) -> go.Figure:
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
    return html.escape(str(text))


def join_items(s) -> str:
    return " + ".join(sorted(s))


# ==============================================================================
# DỮ LIỆU & KHAI PHÁ
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_transactions(file_path: str, max_rows: int = 0) -> List[List[str]]:
    path = Path(file_path)
    if path.suffix.lower() == ".parquet":
        txs = [list(items) for items in pd.read_parquet(path, columns=["items"])["items"]]
    else:  # CSV do preprocess.py xuất: các món phân cách bởi " | "
        txs = [str(v).split(" | ") for v in pd.read_csv(path)["items"]]
    return txs[:max_rows] if max_rows > 0 else txs


@st.cache_data(show_spinner=False)
def mine(file_path: str, max_rows: int, min_support: float, min_conf: float, min_lift: float):
    txs = load_transactions(file_path, max_rows)
    fi_df = OptimizedFPGrowth(min_support=min_support).fit(txs).get_frequent_itemsets()
    rules_df = generate_rules(fi_df, min_confidence=min_conf, min_lift=min_lift)
    return fi_df, rules_df


def popular_ranking(txs: List[List[str]]) -> pd.Series:
    return pd.Series([i for t in txs for i in t]).value_counts()


def recommend_with_fallback(cart: List[str], rules_df: pd.DataFrame, popular: List[str], top_n: int) -> List[Dict]:
    """Gợi ý theo luật; nếu không đủ top_n thì bổ sung món bán chạy chưa có trong giỏ."""
    recs = [dict(r, source="rule") for r in recommend_products(cart, rules_df, top_n=top_n)]
    taken = set(cart) | {r["item"] for r in recs}
    for item in popular:
        if len(recs) >= top_n:
            break
        if item not in taken:
            recs.append({"item": item, "source": "popular"})
            taken.add(item)
    return recs


def split_holdout(txs: List[List[str]], test_ratio: float = 0.2):
    """Chia 80/20 cố định theo seed; mỗi đơn kiểm tra giấu ngẫu nhiên 1 món."""
    rng = random.Random(SEED)
    idx = list(range(len(txs)))
    rng.shuffle(idx)
    n_test = int(len(idx) * test_ratio)
    train = [txs[i] for i in idx[n_test:]]
    tests = []
    for i in idx[:n_test]:
        basket = list(txs[i])
        hidden = basket.pop(rng.randrange(len(basket)))
        tests.append((basket, hidden))
    return train, tests


@st.cache_data(show_spinner=False)
def evaluate(file_path: str, max_rows: int, min_support: float, min_conf: float, min_lift: float, k: int):
    txs = load_transactions(file_path, max_rows)
    train, tests = split_holdout(txs)
    fi = OptimizedFPGrowth(min_support=min_support).fit(train).get_frequent_itemsets()
    rules = generate_rules(fi, min_confidence=min_conf, min_lift=min_lift)
    popular = popular_ranking(train).index.tolist()

    stats = {"rule": [0, 0], "popular": [0, 0], "hybrid": [0, 0]}  # [số đơn có gợi ý, số đơn đoán trúng]
    for cart, hidden in tests:
        rule_items = [r["item"] for r in recommend_products(cart, rules, top_n=k)]
        cart_set = set(cart)
        pop_items = [p for p in popular[: k + len(cart)] if p not in cart_set][:k]
        hybrid_items = rule_items + [p for p in pop_items if p not in rule_items][: k - len(rule_items)]
        for name, items in (("rule", rule_items), ("popular", pop_items), ("hybrid", hybrid_items)):
            stats[name][0] += bool(items)
            stats[name][1] += hidden in items
    n = len(tests)
    rows = [
        {"method": "Chỉ luật kết hợp", "key": "rule"},
        {"method": "Chỉ món bán chạy", "key": "popular"},
        {"method": "Luật + dự phòng bán chạy", "key": "hybrid"},
    ]
    for r in rows:
        covered, hit = stats[r["key"]]
        r.update(coverage=covered / n, hit=hit / n, precision_on_covered=hit / covered if covered else 0.0)
    return pd.DataFrame(rows), len(train), n, len(rules)


def dedupe_pairs(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Gộp A→B và B→A thành một cặp (lift của hai chiều luôn bằng nhau)."""
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
# THANH BÊN: THAM SỐ
# ==============================================================================
st.sidebar.markdown("### Dữ liệu và ngưỡng")

files = sorted(DATA_DIR.glob("*.parquet")) if DATA_DIR.exists() else []
if not files:  # chỉ dùng CSV khi không có parquet, tránh liệt kê trùng một bộ dữ liệu
    files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
if not files:
    st.error("Chưa có dữ liệu trong `data/processed/`. Chạy `python src/preprocess.py` rồi tải lại trang.")
    st.stop()


def file_label(f: Path) -> str:
    level = "kệ hàng" if "aisle" in f.stem else "sản phẩm"
    size = f.stem.split("_")[-1]
    return f"{size} đơn · mức {level}"


default_file = next((i for i, f in enumerate(files) if "product_10k" in f.name), 0)
selected_file = st.sidebar.selectbox("Bộ dữ liệu", files, index=default_file, format_func=file_label)
file_total = len(load_transactions(str(selected_file)))

max_rows = st.sidebar.number_input(
    "Số đơn đưa vào khai phá", min_value=1000, max_value=file_total,
    value=min(10000, file_total), step=1000,
    help=f"Lấy N đơn đầu tiên của bộ dữ liệu (tối đa {file_total:,}).",
)
min_support = st.sidebar.select_slider(
    "Support tối thiểu", options=[0.05, 0.03, 0.02, 0.01, 0.005, 0.002], value=0.01,
    format_func=lambda v: f"{v:.1%}",
    help="Tỉ lệ đơn tối thiểu phải chứa một tập món để tập đó được coi là phổ biến.",
)
min_conf = st.sidebar.slider(
    "Confidence tối thiểu", 0.05, 0.80, 0.10, 0.05, format="%.2f",
    help="P(mua B | đã mua A). Luật có confidence thấp hơn bị bỏ.",
)
min_lift = st.sidebar.slider(
    "Lift tối thiểu", 1.0, 4.0, 1.2, 0.1, format="%.1f",
    help="Lift > 1: mua A làm khả năng mua B cao hơn mức bình thường.",
)
st.sidebar.caption("Support thấp cho nhiều luật hơn nhưng khai phá lâu hơn. Kết quả được lưu đệm sau lần chạy đầu.")

# ==============================================================================
# KHAI PHÁ
# ==============================================================================
transactions = load_transactions(str(selected_file), max_rows)
with st.spinner("Đang dựng FP-Tree và khai phá luật…"):
    fi_df, rules_df = mine(str(selected_file), max_rows, min_support, min_conf, min_lift)

popular = popular_ranking(transactions)
popular_list = popular.index.tolist()
n_tx = len(transactions)
avg_basket = sum(len(t) for t in transactions) / n_tx if n_tx else 0

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

tab_rec, tab_rules, tab_fi, tab_eval = st.tabs(["Gợi ý mua kèm", "Luật kết hợp", "Tập phổ biến", "Đánh giá"])

# ==============================================================================
# TAB 1: GỢI Ý MUA KÈM
# ==============================================================================
# Món có nhiều luật nhất ở vế "nếu mua" — dùng làm lối tắt chọn nhanh, luôn ra gợi ý
ante_counts = pd.Series([i for a in rules_df["antecedents"] for i in a]).value_counts() if not rules_df.empty else pd.Series(dtype=int)
quick_picks = ante_counts.head(4).index.tolist()

if "cart" not in st.session_state or not set(st.session_state["cart"]) <= set(popular_list):
    st.session_state["cart"] = quick_picks[:1]
    st.session_state["hidden"] = None


def set_cart(items: List[str], hidden=None):
    st.session_state["cart"] = items
    st.session_state["hidden"] = hidden


def pick_real_order():
    rng = random.Random()
    basket = list(rng.choice(transactions))
    hidden = basket.pop(rng.randrange(len(basket)))
    set_cart(basket, hidden)


with tab_rec:
    col_cart, col_tags = st.columns([2, 3], gap="large")

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

    with col_tags:
        st.markdown("#### Gợi ý cho giỏ này")
        if not cart:
            st.markdown('<div class="empty-cart">Thêm ít nhất một món vào giỏ để xem gợi ý.</div>', unsafe_allow_html=True)
        else:
            recs = recommend_with_fallback(cart, rules_df, popular_list, top_n)
            n_rule = sum(r["source"] == "rule" for r in recs)
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
            tags = []
            for rank, r in enumerate(recs, 1):
                is_hit = " hit" if hidden is not None and r["item"] == hidden else ""
                if r["source"] == "rule":
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
# TAB 2: LUẬT KẾT HỢP
# ==============================================================================
with tab_rules:
    if rules_df.empty:
        st.info("Không có luật nào đạt ngưỡng hiện tại. Hạ support, confidence hoặc lift ở thanh bên.")
    else:
        pairs = dedupe_pairs(rules_df)
        st.markdown(
            f"**{len(rules_df):,} luật** ứng với **{len(pairs):,} cặp món**. "
            f"Luật A→B và B→A luôn có cùng lift nên biểu đồ bên phải gộp thành một cặp."
        )
        c1, c2 = st.columns([3, 2], gap="large")
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
            st.plotly_chart(style_fig(fig, "Confidence (%)", "Lift"), width="stretch")
            st.caption("Kích thước điểm là support: luật càng to càng xuất hiện nhiều.")
        with c2:
            st.markdown("##### 10 cặp món gắn kết nhất")
            top_pairs = pairs.head(10)
            fig = px.bar(top_pairs, x="Lift", y="Cặp món", orientation="h", text="Lift",
                         hover_data={"Chiều 1": True, "Chiều 2": True, "Lift": ":.2f"})
            fig.update_traces(marker_color=CARROT, texttemplate="×%{x:.2f}", textposition="inside",
                              insidetextanchor="end", textfont=dict(color="#FFFFFF"))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, "Lift", ""), width="stretch")

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
# TAB 3: TẬP PHỔ BIẾN
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
        with c1:
            st.markdown("##### Số tập theo độ dài")
            dist = fi_df["length"].value_counts().sort_index().rename_axis("k").reset_index(name="n")
            fig = px.bar(dist, x="k", y="n", text="n")
            fig.update_traces(marker_color=KALE, textposition="outside", cliponaxis=False)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(style_fig(fig, "Số món trong tập (k)", "Số tập"), width="stretch")
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
                st.plotly_chart(style_fig(fig, "Support (%)", ""), width="stretch")

        with st.expander("FP-Growth tìm các tập này như thế nào"):
            st.markdown(
                f"1. Quét dữ liệu lần 1, đếm số đơn chứa từng món, bỏ món có support dưới {min_support:.1%} "
                "và sắp các món còn lại theo tần suất giảm dần.\n"
                "2. Quét lần 2, chèn từng giỏ vào FP-Tree. Các giỏ có chung tiền tố dùng chung một nhánh, "
                "chỉ tăng bộ đếm, nên cả tập đơn được nén thành một cây.\n"
                "3. Đi từ món ít phổ biến nhất trong bảng header, gom các nhánh chứa món đó thành cây điều kiện "
                "rồi khai phá đệ quy. Không cần sinh tập ứng viên như Apriori."
            )

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

# ==============================================================================
# TAB 4: ĐÁNH GIÁ HOLDOUT
# ==============================================================================
with tab_eval:
    st.markdown(
        "Gợi ý chỉ có ích nếu đoán đúng món khách sẽ mua. Cách kiểm tra: chia ngẫu nhiên các đơn thành "
        "80% để khai phá luật và 20% để kiểm tra (seed cố định 42). Mỗi đơn kiểm tra bị giấu đi 1 món, "
        "rồi đo xem món đó có nằm trong k gợi ý không."
    )
    k_eval = st.select_slider("Số gợi ý được tính (k)", options=[1, 3, 5, 10], value=5)
    if st.button("Chạy đánh giá với ngưỡng hiện tại", type="primary"):
        st.session_state["eval_key"] = (str(selected_file), max_rows, min_support, min_conf, min_lift, k_eval)

    key = st.session_state.get("eval_key")
    if key is None:
        st.caption("Bấm nút để chạy. Với 10.000 đơn, việc này mất vài giây.")
    else:
        if key != (str(selected_file), max_rows, min_support, min_conf, min_lift, k_eval):
            st.caption("Ngưỡng đã thay đổi sau lần chạy trước. Bấm nút để chạy lại.")
        with st.spinner("Đang khai phá trên phần huấn luyện và kiểm tra…"):
            res, n_train, n_test, n_rules = evaluate(*key)
        k = key[-1]
        st.markdown(
            f"Khai phá trên **{n_train:,}** đơn ra **{n_rules:,}** luật, kiểm tra trên **{n_test:,}** đơn, k = {k}."
        )
        long = res.melt(id_vars="method", value_vars=["coverage", "hit"], var_name="metric", value_name="v")
        long["metric"] = long["metric"].map({"coverage": "Đơn nhận được gợi ý", "hit": f"Đoán trúng món bị giấu (top {k})"})
        fig = px.bar(long, x="v", y="method", color="metric", barmode="group", orientation="h", text="v",
                     color_discrete_sequence=[RULE, KALE])
        fig.update_traces(texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0, title=None),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, "", "", height=300).update_layout(showlegend=True, margin=dict(t=40)),
                        width="stretch")
        st.dataframe(
            res.drop(columns="key").rename(columns={
                "method": "Cách gợi ý", "coverage": "Đơn nhận được gợi ý",
                "hit": f"Hit@{k}", "precision_on_covered": f"Hit@{k} trên các đơn có gợi ý",
            }).style.format({"Đơn nhận được gợi ý": "{:.1%}", f"Hit@{k}": "{:.2%}", f"Hit@{k} trên các đơn có gợi ý": "{:.2%}"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Hit@k = tỉ lệ đơn kiểm tra có món bị giấu nằm trong k gợi ý. "
            "Món bán chạy là mốc so sánh tối thiểu: cách gợi ý nào cũng nên vượt được mốc này."
        )
