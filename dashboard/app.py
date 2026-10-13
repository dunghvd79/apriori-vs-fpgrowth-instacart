"""Interactive Instacart experiment dashboard for Apriori and FP-Growth."""

import ast
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from my_apriori import MyApriori
from my_fpgrowth import MyFPGrowth
from mlxtend_apriori import MlxtendApriori
from mlxtend_fpgrowth import MlxtendFPGrowth
from association_rules import generate_association_rules

st.set_page_config(page_title="Instacart | Apriori vs FP-Growth", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
      :root { --ink: #182230; --muted: #667085; --line: #e7ebf0; --accent: #187a67; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f7f8fa; color: var(--ink); }
      .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
      h1 { letter-spacing: -0.04em; font-size: 2rem !important; font-weight: 700 !important; }
      h2, h3 { letter-spacing: -0.025em; color: var(--ink); }
      [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; }
      [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
      [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.65rem; font-weight: 700; }
      [data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] h2 { font-size: 1rem; }
      div.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); border-radius: 9px; font-weight: 600; }
      div.stButton > button[kind="primary"]:hover { background: #126552; border-color: #126552; }
      [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
      [data-testid="stTabs"] button { font-weight: 600; }
      hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_transactions(file_path: str, file_mtime: float):
    path = Path(file_path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "items" not in frame.columns:
        raise ValueError("File dữ liệu cần có cột 'items'.")

    def parse_items(value):
        if isinstance(value, list):
            items = value
        elif pd.isna(value):
            items = []
        else:
            items = ast.literal_eval(value)
        if not isinstance(items, (list, tuple, set)):
            raise ValueError("Mỗi giá trị trong cột 'items' phải là một danh sách sản phẩm.")
        return list(dict.fromkeys(str(item) for item in items if str(item).strip()))

    return [parse_items(value) for value in frame["items"]]


def execute(algo_class, transactions, min_support):
    tracemalloc.start()
    started = time.perf_counter()
    model = algo_class(min_support=min_support).fit(transactions)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    itemsets = model.get_frequent_itemsets()
    return {
        "itemsets": itemsets,
        "time_sec": elapsed,
        "peak_mb": peak / (1024 * 1024),
    }


def compare_itemsets(left, right):
    left_map = {frozenset(row["itemsets"]): row["support"] for row in left}
    right_map = {frozenset(row["itemsets"]): row["support"] for row in right}
    common = left_map.keys() & right_map.keys()
    return {
        "left_only": len(left_map.keys() - right_map.keys()),
        "right_only": len(right_map.keys() - left_map.keys()),
        "support_mismatches": sum(abs(left_map[key] - right_map[key]) > 1e-12 for key in common),
    }


st.title("Market Basket Analysis")
st.caption("Instacart · So sánh Apriori và FP-Growth trên cùng dữ liệu")

with st.sidebar:
    st.header("Thiết lập")
    st.caption("Chọn dữ liệu và ngưỡng chạy")
    data_files = sorted(
        list((DATA_DIR / "processed").glob("*.csv")) + list((DATA_DIR / "processed").glob("*.parquet"))
    )
    source_mode = st.radio("Nguồn dữ liệu", ["File đã xử lý", "Tải file lên"], horizontal=True)
    uploaded_file = None
    selected_path = None
    if source_mode == "File đã xử lý":
        if data_files:
            selected_path = st.selectbox("Dữ liệu", data_files, format_func=lambda path: path.name)
        else:
            st.warning("Chưa có file trong data/processed. Chọn ‘Tải file lên’ hoặc chạy src/preprocess.py.")
    else:
        uploaded_file = st.file_uploader("Tải CSV hoặc Parquet (cần cột items)", type=["csv", "parquet"])
    max_rows = st.number_input("Số giao dịch dùng (0 = tất cả)", min_value=0, max_value=1000000, value=1000, step=500)
    implementation = st.selectbox("Cách cài đặt", ["mlxtend", "Tự cài đặt"])
    support_values = st.multiselect(
        "Ngưỡng minimum support",
        [0.05, 0.03, 0.02, 0.01, 0.005, 0.001],
        default=[0.02, 0.01],
        format_func=lambda value: f"{value:.1%}",
    )
    min_confidence = st.slider("Confidence tối thiểu cho luật", 0.0, 1.0, 0.3, 0.05)
    min_lift = st.number_input("Lift tối thiểu cho luật", min_value=0.0, value=1.0, step=0.1)
    st.divider()
    run = st.button("Chạy so sánh", type="primary", use_container_width=True)

try:
    if source_mode == "Tải file lên":
        if uploaded_file is None:
            st.info("Tải lên file đã tiền xử lý có cột 'items' để bắt đầu.")
            st.stop()
        uploaded_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".parquet"):
            from io import BytesIO
            frame = pd.read_parquet(BytesIO(uploaded_bytes))
        else:
            from io import BytesIO
            frame = pd.read_csv(BytesIO(uploaded_bytes))
        if "items" not in frame.columns:
            raise ValueError("File dữ liệu cần có cột 'items'.")
        all_transactions = [
            list(dict.fromkeys(str(item) for item in (value if isinstance(value, (list, tuple, set)) else ast.literal_eval(value)) if str(item).strip()))
            for value in frame["items"]
        ]
    elif selected_path is not None:
        all_transactions = load_transactions(str(selected_path), selected_path.stat().st_mtime)
    else:
        st.stop()
except Exception as error:
    st.error(f"Không đọc được dữ liệu: {error}")
    st.stop()

transactions = all_transactions[:int(max_rows)] if max_rows else all_transactions
st.markdown("#### Quy mô dữ liệu")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Giao dịch trong file", f"{len(all_transactions):,}")
col_b.metric("Giao dịch thí nghiệm", f"{len(transactions):,}")
col_c.metric("Sản phẩm phân biệt", f"{len({item for tx in transactions for item in tx}):,}")

if not support_values:
    st.info("Chọn ít nhất một ngưỡng support ở thanh bên để chạy thí nghiệm.")
elif run:
    if not transactions:
        st.error("Dữ liệu không có giao dịch để phân tích.")
        st.stop()
    classes = (MyApriori, MyFPGrowth) if implementation == "Tự cài đặt" else (MlxtendApriori, MlxtendFPGrowth)
    names = ("Apriori", "FP-Growth")
    output = {}
    progress = st.progress(0, text="Đang chạy các thuật toán…")
    total = len(support_values) * 2
    completed = 0
    for support in sorted(support_values, reverse=True):
        output[support] = {}
        for name, algo_class in zip(names, classes):
            with st.spinner(f"Đang chạy {name} với support {support:.1%}…"):
                output[support][name] = execute(algo_class, transactions, support)
            completed += 1
            progress.progress(completed / total, text=f"Đã chạy {completed}/{total} lượt")
    progress.empty()
    st.session_state["experiment"] = {
        "output": output,
        "implementation": implementation,
        "transactions": len(transactions),
        "min_confidence": min_confidence,
        "min_lift": min_lift,
    }

experiment = st.session_state.get("experiment")
if experiment:
    output = experiment["output"]
    st.divider()
    st.subheader("Hiệu năng")
    st.caption("Thời gian chạy và số tập phổ biến theo từng ngưỡng support")
    summary = []
    for support, algorithms in output.items():
        for name, result in algorithms.items():
            summary.append({
                "min_support": support,
                "algorithm": name,
                "itemsets": len(result["itemsets"]),
                "runtime_sec": result["time_sec"],
                "peak_python_memory_mb": result["peak_mb"],
            })
    summary_frame = pd.DataFrame(summary)
    st.dataframe(
        summary_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "min_support": st.column_config.NumberColumn("Min support", format="%.1%%"),
            "algorithm": st.column_config.TextColumn("Thuật toán"),
            "itemsets": st.column_config.NumberColumn("Số tập phổ biến", format="%d"),
            "runtime_sec": st.column_config.NumberColumn("Thời gian (giây)", format="%.3f"),
            "peak_python_memory_mb": st.column_config.NumberColumn("Peak bộ nhớ (MB)", format="%.2f"),
        },
    )

    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.markdown("**Thời gian chạy**")
        st.bar_chart(summary_frame.pivot(index="min_support", columns="algorithm", values="runtime_sec"))
    with metric_right:
        st.markdown("**Bộ nhớ Python đỉnh**")
        st.bar_chart(summary_frame.pivot(index="min_support", columns="algorithm", values="peak_python_memory_mb"))

    st.divider()
    st.subheader("Đối chiếu kết quả")
    st.caption("Kiểm tra hai thuật toán có tìm được cùng các tập phổ biến hay không")
    support_tabs = st.tabs([f"Support {support:.1%}" for support in output])
    for tab, (support, algorithms) in zip(support_tabs, output.items()):
        with tab:
            apriori_rows = algorithms["Apriori"]["itemsets"]
            fpgrowth_rows = algorithms["FP-Growth"]["itemsets"]
            comparison = compare_itemsets(apriori_rows, fpgrowth_rows)
            c1, c2, c3 = st.columns(3)
            c1.metric("Chỉ có trong Apriori", comparison["left_only"])
            c2.metric("Chỉ có trong FP-Growth", comparison["right_only"])
            c3.metric("Support khác nhau", comparison["support_mismatches"])
            if any(comparison.values()):
                st.warning("Hai kết quả chưa khớp hoàn toàn ở ngưỡng này.")
            else:
                st.success("Hai thuật toán trả về cùng tập phổ biến và support.")
            left, right = st.columns(2)
            left.markdown("**Apriori — tập phổ biến hàng đầu**")
            left.dataframe(pd.DataFrame(apriori_rows).head(100), use_container_width=True, hide_index=True)
            right.markdown("**FP-Growth — tập phổ biến hàng đầu**")
            right.dataframe(pd.DataFrame(fpgrowth_rows).head(100), use_container_width=True, hide_index=True)

    chosen_support = st.selectbox("Ngưỡng support dùng để xem luật", list(output), format_func=lambda value: f"{value:.1%}")
    st.divider()
    st.subheader("Luật kết hợp")
    st.caption(f"Lọc theo confidence ≥ {experiment['min_confidence']:.0%} và lift ≥ {experiment['min_lift']:.1f}")
    rule_cols = st.columns(2)
    for col, name in zip(rule_cols, names):
        rows = output[chosen_support][name]["itemsets"]
        rules = generate_association_rules(
            rows,
            min_confidence=experiment["min_confidence"],
            min_lift=experiment["min_lift"],
        )
        with col:
            st.markdown(f"**{name}: {len(rules):,} luật**")
            rule_frame = pd.DataFrame(rules)
            if not rule_frame.empty:
                rule_frame["antecedents"] = rule_frame["antecedents"].apply(lambda x: " + ".join(x))
                rule_frame["consequents"] = rule_frame["consequents"].apply(lambda x: " + ".join(x))
                st.dataframe(rule_frame.head(100), use_container_width=True, hide_index=True)
            else:
                st.info("Không có luật thỏa ngưỡng đã chọn.")

    st.caption("Bộ nhớ là peak allocation Python do tracemalloc đo; không đại diện toàn bộ RSS của tiến trình.")
else:
    st.info("Thiết lập dữ liệu và ngưỡng support ở thanh bên, sau đó nhấn **Chạy so sánh**.")
