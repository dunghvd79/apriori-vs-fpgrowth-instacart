"""
Chạy toàn bộ thực nghiệm cho Chương 4 và xuất bảng (CSV) + biểu đồ (PNG) vào results/.

    python scripts/run_experiments.py            # chạy tất cả
    python scripts/run_experiments.py --only eval

Các phần:
  env      — cấu hình máy và phiên bản thư viện
  verify   — đối chiếu FP-Growth tự cài đặt với mlxtend
  mining   — số tập phổ biến / số luật theo support, mức sản phẩm và mức kệ
  rules    — luật tiêu biểu (cặp gắn kết nhất theo lift) + biểu đồ
  eval     — đánh giá gợi ý holdout 80/20: Coverage, Hit@k, so với món bán chạy
  case     — tình huống Soda: món có liên quan nhưng không có luật

Mọi ngẫu nhiên dùng seed 42; cách chia holdout giống hệt tab Đánh giá của dashboard.
"""

import argparse
import os
import platform
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from my_fpgrowth import OptimizedFPGrowth, generate_rules, recommend_products  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "results"
FIG = OUT / "figures"
SEED = 42
MIN_CONF, MIN_LIFT = 0.10, 1.2
KS = (1, 3, 5, 10)

# Bảng màu phân loại mặc định của skill dataviz (thứ tự cố định: xanh dương, cam, xanh ngọc)
C_RULE, C_POP, C_HYB = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
METHODS = [("rule", "Chỉ luật kết hợp", C_RULE), ("popular", "Chỉ món bán chạy", C_POP),
           ("hybrid", "Luật + dự phòng bán chạy", C_HYB)]


def load(name):
    return [list(x) for x in pd.read_parquet(DATA / f"{name}.parquet", columns=["items"])["items"]]


def mine(txs, sup, conf=MIN_CONF, lift=MIN_LIFT):
    t = time.perf_counter()
    fi = OptimizedFPGrowth(sup).fit(txs).get_frequent_itemsets()
    t_mine = time.perf_counter() - t
    rules = generate_rules(fi, min_confidence=conf, min_lift=lift)
    return fi, rules, t_mine


def join(s):
    return " + ".join(sorted(s))


def save(df, name):
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"  -> results/{name}.csv")
    print(df.to_string(index=False))


# ------------------------------------------------------------------ biểu đồ
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 11, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 100,
    })
    return plt


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  -> results/figures/{name}.png")


# ------------------------------------------------------------------ các phần
def part_env():
    import matplotlib, mlxtend, numpy, pyarrow
    rows = [
        ("Hệ điều hành", f"{platform.system()} {platform.release()}"),
        ("CPU", f"{platform.processor()} ({os.cpu_count()} luồng)"),
        ("Python", platform.python_version()),
        ("pandas", pd.__version__), ("NumPy", numpy.__version__), ("PyArrow", pyarrow.__version__),
        ("mlxtend", mlxtend.__version__), ("Matplotlib", matplotlib.__version__),
    ]
    try:
        import psutil
        rows.insert(2, ("RAM", f"{psutil.virtual_memory().total / 2**30:.1f} GB"))
    except ImportError:
        pass
    save(pd.DataFrame(rows, columns=["Thành phần", "Giá trị"]), "env")


def part_verify():
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    configs = [("transactions_product_10k", None, 0.01), ("transactions_product_50k", None, 0.005),
               ("transactions_aisle_100k", None, 0.05), ("transactions_aisle_100k", 30000, 0.02)]
    rows = []
    for name, n, sup in configs:
        txs = load(name)[:n] if n else load(name)
        fi, rules, t_mine = mine(txs, sup, 0.2, 1.2)
        te = TransactionEncoder()
        X = pd.DataFrame.sparse.from_spmatrix(te.fit(txs).transform(txs, sparse=True), columns=te.columns_)
        t = time.perf_counter()
        ref = fpgrowth(X, min_support=sup, use_colnames=True)
        t_ref = time.perf_counter() - t
        a = {s: round(v, 9) for s, v in zip(fi.itemsets, fi.support)}
        b = {s: round(v, 9) for s, v in zip(ref.itemsets, ref.support)}
        r2 = association_rules(ref, metric="confidence", min_threshold=0.2)
        r2 = r2[r2.lift >= 1.2]
        k1 = set(zip(rules.antecedents, rules.consequents))
        k2 = set(zip(r2.antecedents, r2.consequents))
        rows.append({
            "Dữ liệu": f"{'Sản phẩm' if 'product' in name else 'Kệ hàng'}, " + f"{len(txs):,} giỏ".replace(",", "."),
            "Support": f"{sup:.1%}".replace(".", ","),
            "Tập phổ biến (tự cài đặt)": len(a), "Tập phổ biến (mlxtend)": len(b),
            "Khớp tập & support": "Có" if a == b else "Không",
            "Dài nhất": int(fi["length"].max()),
            "Luật (tự cài đặt)": len(k1), "Luật (mlxtend)": len(k2), "Khớp luật": "Có" if k1 == k2 else "Không",
            "Thời gian tự cài đặt (s)": round(t_mine, 2), "Thời gian mlxtend (s)": round(t_ref, 2),
        })
    save(pd.DataFrame(rows), "verify_mlxtend")


def part_mining():
    plan = [(f"transactions_product_{s}", (0.01, 0.005, 0.002)) for s in ("10k", "50k", "100k")]
    plan.append(("transactions_aisle_100k", (0.10, 0.05, 0.02)))
    rows = []
    for name, sups in plan:
        txs = load(name)
        for sup in sups:
            fi, rules, t_mine = mine(txs, sup)
            lens = fi["length"].value_counts()
            antes = set().union(*rules.antecedents) if len(rules) else set()
            n_items = len({i for t in txs for i in t})
            rows.append({
                "Mức": "Kệ hàng" if "aisle" in name else "Sản phẩm", "Số giỏ": len(txs), "Support": sup,
                "Tập 1 item": int(lens.get(1, 0)), "Tập 2 item": int(lens.get(2, 0)),
                "Tập ≥3 item": int(lens[lens.index >= 3].sum()), "Tổng tập": len(fi),
                "Số luật": len(rules), "Item có luật": len(antes),
                "Tỉ lệ item có luật (%)": round(100 * len(antes) / n_items, 2),
                "Thời gian khai phá (s)": round(t_mine, 2),
            })
    save(pd.DataFrame(rows), "mining_summary")


def top_pairs(rules, n=10):
    seen = {}
    for r in rules.itertuples(index=False):
        key = frozenset([r.antecedents, r.consequents])
        e = seen.setdefault(key, {"A": join(r.antecedents), "B": join(r.consequents), "Support (%)": round(100 * r.support, 2),
                                  "Lift": round(r.lift, 2), "Conf A→B (%)": None, "Conf B→A (%)": None})
        if join(r.antecedents) == e["A"]:
            e["Conf A→B (%)"] = round(100 * r.confidence, 1)
        else:
            e["Conf B→A (%)"] = round(100 * r.confidence, 1)
    return pd.DataFrame(seen.values()).sort_values("Lift", ascending=False).head(n).reset_index(drop=True)


def part_rules():
    plt = plt_setup()
    prod = load("transactions_product_100k")
    _, rules_p, _ = mine(prod, 0.002)
    tp = top_pairs(rules_p)
    save(tp, "top_pairs_product_100k_s0.2")
    aisle = load("transactions_aisle_100k")
    _, rules_a, _ = mine(aisle, 0.02)
    ta = top_pairs(rules_a[(rules_a.antecedents.map(len) == 1) & (rules_a.consequents.map(len) == 1)])
    save(ta, "top_pairs_aisle_100k_s2")

    # Hình: phân tán confidence – lift (sản phẩm 100k, support 0,2%)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(rules_p.confidence * 100, rules_p.lift, s=8 + rules_p.support * 100 * 60,
               color=C_RULE, alpha=0.55, edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Lift (thang log)")
    savefig(fig, "rules_scatter_product_100k")

    # Hình: 10 cặp kệ hàng gắn kết nhất (tên kệ ngắn nên vẽ được; cặp sản phẩm trình bày bằng bảng)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [f"{a} + {b}" for a, b in zip(ta.A, ta.B)]
    ax.barh(range(len(ta)), ta.Lift, color=C_RULE, height=0.62)
    ax.set_yticks(range(len(ta)), labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(ta.Lift):
        ax.text(v + 0.04, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=10, color=INK)
    ax.set_xlabel("Lift")
    ax.set_xlim(0, ta.Lift.max() * 1.12)
    savefig(fig, "top_pairs_aisle_100k")


def split_holdout(txs, ratio=0.2):
    rng = random.Random(SEED)
    idx = list(range(len(txs)))
    rng.shuffle(idx)
    n_test = int(len(idx) * ratio)
    train = [txs[i] for i in idx[n_test:]]
    tests = []
    for i in idx[:n_test]:
        b = list(txs[i])
        hidden = b.pop(rng.randrange(len(b)))
        tests.append((b, hidden))
    return train, tests


def evaluate(txs, sup):
    train, tests = split_holdout(txs)
    _, rules, _ = mine(train, sup)
    popular = pd.Series([i for t in train for i in t]).value_counts().index.tolist()
    kmax = max(KS)
    agg = {m: {"cov": 0, **{k: 0 for k in KS}} for m, _, _ in METHODS}
    t = time.perf_counter()
    for cart, hidden in tests:
        rule = [r["item"] for r in recommend_products(cart, rules, top_n=kmax)]
        cs = set(cart)
        pop = [p for p in popular[: kmax + len(cart)] if p not in cs][:kmax]
        hyb = rule + [p for p in pop if p not in rule][: kmax - len(rule)]
        for m, lst in (("rule", rule), ("popular", pop), ("hybrid", hyb)):
            agg[m]["cov"] += bool(lst)
            for k in KS:
                agg[m][k] += hidden in lst[:k]
    n = len(tests)
    rows = []
    for m, label, _ in METHODS:
        rows.append({"Phương pháp": label, "key": m, "Support": sup, "Số luật": len(rules), "Số giỏ kiểm tra": n,
                     "Coverage (%)": round(100 * agg[m]["cov"] / n, 1),
                     **{f"Hit@{k} (%)": round(100 * agg[m][k] / n, 2) for k in KS}})
    print(f"    sup={sup}: {len(rules)} luật, đánh giá {time.perf_counter() - t:.0f}s")
    return rows


def part_eval():
    plt = plt_setup()
    plan = [("Sản phẩm", "transactions_product_100k", (0.01, 0.005, 0.002)),
            ("Kệ hàng", "transactions_aisle_100k", (0.10, 0.05, 0.02))]
    rows = []
    for level, name, sups in plan:
        txs = load(name)
        print(f"  {level}: {len(txs):,} giỏ")
        for sup in sups:
            rows += [dict(r, **{"Mức": level}) for r in evaluate(txs, sup)]
    df = pd.DataFrame(rows)
    save(df.drop(columns="key"), "eval_holdout")

    # Hình: Hit@5 theo support, hai mức (small multiples, mỗi mức một trục riêng)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
    for ax, (level, _, sups) in zip(axes, plan):
        sub = df[df["Mức"] == level]
        xs = range(len(sups))
        for m, label, color in METHODS:
            ys = [sub[(sub.key == m) & (sub.Support == s)]["Hit@5 (%)"].iloc[0] for s in sups]
            ls = "--" if m == "popular" else "-"
            ax.plot(xs, ys, ls, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.set_xticks(list(xs), [f"{s:.1%}".replace(".", ",") for s in sups])
        ax.set_title(f"Mức {level.lower()}", color=INK)
        ax.set_xlabel("Support tối thiểu")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Hit@5 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    savefig(fig, "eval_hit5_by_support")

    # Hình: Hit@k theo k, ở ngưỡng thấp nhất của mỗi mức
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (level, _, sups) in zip(axes, plan):
        sub = df[(df["Mức"] == level) & (df.Support == sups[-1])]
        for m, label, color in METHODS:
            r = sub[sub.key == m].iloc[0]
            ls = "--" if m == "popular" else "-"
            ax.plot(KS, [r[f"Hit@{k} (%)"] for k in KS], ls, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.set_xticks(KS)
        ax.set_title(f"Mức {level.lower()}, support {sups[-1]:.1%}".replace(".", ","), color=INK)
        ax.set_xlabel("Số gợi ý k")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Hit@k (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    savefig(fig, "eval_hitk_by_k")


def part_case():
    txs = load("transactions_product_10k")
    n = len(txs)
    pop = pd.Series([i for t in txs for i in t]).value_counts()
    soda = [t for t in txs if "Soda" in t]
    co = pd.Series([i for t in soda for i in t if i != "Soda"]).value_counts().head(5)
    rows = [{"Món mua cùng Soda": item, "Số đơn có cả hai": c, "Support cặp (%)": round(100 * c / n, 2),
             "Confidence (%)": round(100 * c / len(soda), 1), "Lift": round((c / len(soda)) / (pop[item] / n), 1)}
            for item, c in co.items()]
    print(f"  Soda: {len(soda)} đơn / {n} ({100 * len(soda) / n:.2f}%)")
    save(pd.DataFrame(rows), "case_soda_10k")


PARTS = {"env": part_env, "verify": part_verify, "mining": part_mining, "rules": part_rules,
         "eval": part_eval, "case": part_case}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(PARTS), help="chỉ chạy các phần này")
    args = ap.parse_args()
    for name in args.only or PARTS:
        print(f"\n=== {name} ===")
        t = time.perf_counter()
        PARTS[name]()
        print(f"  ({time.perf_counter() - t:.0f}s)")
