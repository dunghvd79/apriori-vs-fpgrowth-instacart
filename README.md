# 🛒 Hệ Thống Dự Đoán & Gợi Ý Sản Phẩm Mua Kèm Bằng Thuật Toán FP-Growth (Instacart Market Basket Analysis)

> **Đề tài:** Khai phá luật kết hợp (Association Rule Mining) và Xây dựng hệ thống gợi ý bán kèm sản phẩm (Cross-Selling Recommender System) sử dụng thuật toán **FP-Growth** trên bộ dữ liệu **Instacart Market Basket Analysis**.

---

## 📌 1. Giới thiệu Đề tài & Mục tiêu Bài toán

Trong ngành bán lẻ và thương mại điện tử (E-commerce / Supermarket), việc thấu hiểu hành vi mua sắm của khách hàng đóng vai trò sống còn để tối ưu hóa doanh thu. 

Bài toán **Phân tích Giỏ hàng (Market Basket Analysis - MBA)** đặt ra câu hỏi cốt lõi:
> *"Khi một khách hàng đã bỏ sản phẩm **A** vào giỏ hàng, họ có xu hướng mua kèm thêm sản phẩm **B** nào nhất?"*

### Mục tiêu chính của dự án:
1. **Khai phá các quy luật mua sắm ẩn (Association Rules):** Tìm ra các cặp/nhóm sản phẩm thường xuyên được mua cùng nhau trong hơn 3 triệu đơn hàng của Instacart.
2. **Dự đoán & Gợi ý sản phẩm mua kèm thời gian thực (Live Recommender):** Dựa trên giỏ hàng hiện tại của người dùng, hệ thống tự động suy diễn và đề xuất các sản phẩm mua kèm có độ tin cậy (*Confidence*) và độ nâng (*Lift*) cao nhất.
3. **Tối ưu hóa hiệu năng bằng thuật toán FP-Growth:** Sử dụng cấu trúc cây tiền tố **FP-Tree** để khai phá dữ liệu quy mô lớn một cách siêu tốc, tiết kiệm hơn 90% bộ nhớ RAM so với phương pháp sinh ứng viên cổ điển (Apriori).
4. **Xây dựng Dashboard Web trực quan:** Cung cấp giao diện tương tác Streamlit chuyên nghiệp giúp người quản lý siêu thị tra cứu luật mua sắm và mô phỏng giỏ hàng gợi ý sản phẩm.

---

## ⚡ 2. Tại sao chọn duy nhất thuật toán FP-Growth?

Khai phá luật kết hợp có hai thuật toán kinh điển là **Apriori** và **FP-Growth**. Dự án này **lựa chọn FP-Growth làm thuật toán hạt nhân duy nhất** để triển khai hệ thống dự đoán và gợi ý vì những ưu thế vượt trội:

| Tiêu chí | Apriori (Cổ điển) | FP-Growth (Được chọn trong dự án) |
| :--- | :--- | :--- |
| **Độ chính xác của luật** | Chuẩn xác 100% | **Chuẩn xác 100% (cho kết quả y hệt Apriori)** |
| **Số lần quét cơ sở dữ liệu** | Quét $k$ lần (với $k$ là độ dài giỏ lớn nhất) $\rightarrow$ rất chậm | **Chỉ quét đúng 2 lần duy nhất** |
| **Cơ chế tìm kiếm** | Sinh tập ứng viên tổ hợp $C_k$ khổng lồ $\rightarrow$ nghẽn RAM | **Không sinh ứng viên**, nén dữ liệu vào cây FP-Tree |
| **Bộ nhớ RAM tiêu thụ** | Rất tốn RAM (**6,674 MB** ở ngưỡng support 0.5%) | **Tiết kiệm vượt trội (528 MB RAM, giảm hơn 12 lần)** |
| **Tốc độ thực thi** | Giảm theo cấp số nhân khi dữ liệu lớn | **Nhanh gấp nhiều lần**, phù hợp gợi ý thời gian thực |

> **Kết luận:** FP-Growth giữ nguyên 100% độ chính xác toán học của tập luật, nhưng giải quyết triệt để bài toán tài nguyên bộ nhớ và thời gian tính toán trên tập dữ liệu hàng trăm nghìn đơn hàng.

---

## 🧠 3. Cơ chế Hoạt động của FP-Growth & Dự Đoán Gợi Ý

### 3.1. Quy trình khai phá của FP-Growth (Frequent Pattern Growth)
Thuật toán hoạt động qua 3 bước cốt lõi:
1. **Lần quét 1:** Quét toàn bộ lịch sử đơn hàng để đếm tần suất xuất hiện của từng sản phẩm. Loại bỏ các sản phẩm có tần suất nhỏ hơn `min_support`. Sắp xếp các sản phẩm còn lại theo thứ tự tần suất giảm dần.
2. **Lần quét 2 (Dựng cây FP-Tree):** Đọc lại từng giỏ hàng và chèn các sản phẩm vào cây tiền tố (FP-Tree). Các đơn hàng có chung chuỗi sản phẩm đầu sẽ dùng chung nhánh cây (nén dữ liệu tối đa). Đồng thời tạo bảng liên kết đỉnh (**Header Table & Node Links**) để duyệt nhanh các nút cùng tên.
3. **Khai phá đệ quy (Divide-and-Conquer):** Duyệt bảng Header Table từ dưới lên (từ món ít phổ biến đến phổ biến nhất), xây dựng các **Cây tiền tố điều kiện (Conditional FP-Tree)** để trích xuất toàn bộ tập phổ biến (*Frequent Itemsets*) mà không cần sinh tập ứng viên.

### 3.2. Trích xuất Luật Kết Hợp (Association Rules)
Từ các tập phổ biến, hệ thống sinh ra các luật dạng $A \rightarrow B$ (Nếu mua $A$ thì sẽ mua kèm $B$):
* **Support (Độ hỗ trợ):** Tỷ lệ các đơn hàng có chứa cả $A$ và $B$.
  $$\text{Support}(A \rightarrow B) = P(A \cup B) = \frac{\text{Số đơn có cả } A \text{ và } B}{\text{Tổng số đơn hàng}}$$
* **Confidence (Độ tin cậy):** Xác suất khách mua $B$ khi đã biết khách mua $A$.
  $$\text{Confidence}(A \rightarrow B) = P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
* **Lift (Độ nâng):** Thước đo mức độ tương quan giữa việc mua $A$ và việc mua $B$.
  $$\text{Lift}(A \rightarrow B) = \frac{P(A \cup B)}{P(A) \times P(B)} = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
  * $\text{Lift} > 1$: Tương quan dương mạnh mẽ (mua $A$ làm tăng đột biến khả năng mua $B$).
  * $\text{Lift} = 1$: Hai sản phẩm hoàn toàn độc lập.
  * $\text{Lift} < 1$: Tương quan âm (mua $A$ làm giảm khả năng mua $B$).

### 3.3. Thuật toán Dự đoán Sản phẩm Mua kèm (Live Recommender)
Khi khách hàng đang có giỏ hàng $C = \{item_1, item_2, ...\}$:
1. Hệ thống lọc tất cả các luật $A \rightarrow B$ thỏa mãn điều kiện $A \subseteq C$ (tiền đề là tập con của giỏ hàng).
2. Lấy tập các sản phẩm mục tiêu gợi ý: $B_{\text{candidate}} = B \setminus C$ (chưa có trong giỏ).
3. Xếp hạng các sản phẩm gợi ý theo thứ tự ưu tiên:
   $$\text{Score} = \text{Lift} \times \text{Confidence}$$
4. Xuất kết quả kèm theo câu giải thích kinh doanh trực quan cho người dùng.

---

## 📁 4. Cấu trúc Thư mục Dự án

```text
apriori-vs-fpgrowth-instacart/
├── data/
│   ├── raw/                 # Dữ liệu gốc tải từ Kaggle (orders, products, order_products...)
│   └── processed/           # Dữ liệu giỏ hàng đã lọc và mẫu lồng nhau (.parquet & .csv)
│       ├── transactions_product_10k.parquet / .csv   # Mẫu 10k đơn (mức sản phẩm)
│       ├── transactions_product_50k.parquet / .csv   # Mẫu 50k đơn (mức sản phẩm)
│       ├── transactions_product_100k.parquet / .csv  # Mẫu 100k đơn (mức sản phẩm)
│       └── transactions_aisle_100k.parquet / .csv    # Mẫu 100k đơn (mức kệ hàng)
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Tiền xử lý, lọc giỏ >= 2 món, nested sampling, xuất file
│   ├── mining.py            # Khai phá FP-Growth (cắt tỉa món hiếm, ma trận thưa, sinh luật)
│   └── benchmark.py         # Đo lường kiểm chứng hiệu năng và bộ nhớ RAM
├── dashboard/
│   └── app.py               # Ứng dụng Streamlit Dashboard: Gợi ý giỏ hàng & Khám phá luật
├── notebooks/
│   ├── 01_eda.ipynb         # Khám phá phân bố dữ liệu Instacart và hiện tượng đuôi dài
│   └── 02_experiments.ipynb # Thử nghiệm kiểm chứng luật và trích xuất gợi ý
├── results/                 # Bảng số liệu benchmark và danh sách luật xuất ra
├── report/                  # Báo cáo Word / PDF và Slide thuyết trình
├── requirements.txt         # Danh sách thư viện cần thiết
├── .streamlit/
│   └── config.toml          # Cấu hình theme sáng (Light Theme) và bảo mật cho Dashboard
└── README.md
```

---

## 🚀 5. Hướng dẫn Cài đặt & Chuẩn bị Môi trường

### Bước 1: Khởi tạo môi trường ảo Python
Khuyến nghị sử dụng Python 3.9, 3.10 hoặc 3.11:

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo
# Trên macOS / Linux:
source .venv/bin/activate

# Trên Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

---

## 📊 6. Chuẩn bị Dữ liệu (Instacart Dataset)

1. Tải bộ dữ liệu từ Kaggle: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data).
2. Giải nén đầy đủ 6 tệp `.csv` vào thư mục `data/raw/`:
   * `order_products__prior.csv`
   * `order_products__train.csv`
   * `products.csv`
   * `orders.csv`
   * `aisles.csv`
   * `departments.csv`

3. Chạy quy trình tiền xử lý và sinh mẫu dữ liệu:
   ```bash
   python src/preprocess.py
   ```
   * **Cơ chế tiền xử lý:**
     * Lọc bỏ các đơn hàng chỉ có 1 món (vì đơn 1 món không thể tạo ra luật mua kèm $A \rightarrow B$).
     * Áp dụng **Lấy mẫu lồng nhau (Nested Sampling)** với `seed=42`: Tập 10k là tập con của 50k, 50k là tập con của 100k ($10k \subset 50k \subset 100k$), giúp tần suất sản phẩm luôn ổn định giữa các quy mô.
     * Tự động xuất song song file `.parquet` (cho thuật toán nạp siêu tốc) và `.csv` (để tra cứu trực tiếp trong Excel).

---

## 🖥️ 7. Hướng Dẫn Sử Dụng Web Dashboard (Streamlit)

Khởi chạy ứng dụng Web trực quan bằng lệnh:

```bash
streamlit run dashboard/app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

### Các chức năng nổi bật trên Dashboard:
1. **🛍️ Tab 1 — Giả Lập Giỏ Hàng & Gợi Ý Mua Kèm (Live Recommender):**
   * Người dùng chọn các món đồ đang có trong giỏ (ví dụ: *Chuối*, *Bơ hữu cơ*...).
   * Nhấn nút **"✨ Tìm sản phẩm gợi ý mua kèm"**.
   * Hệ thống lập tức hiển thị các sản phẩm được dự đoán mua kèm với đầy đủ: Tên sản phẩm, Độ tin cậy (Confidence), Độ nâng (Lift) và câu giải thích nghiệp vụ bằng ngôn ngữ tự nhiên.
2. **📜 Tab 2 — Khám Phá Toàn Bộ Tập Luật (Association Rules Explorer):**
   * Cho phép tùy chỉnh thanh trượt `Min Confidence` và `Min Lift`.
   * Bảng hiển thị: `Nếu mua [A] -> Sẽ mua kèm [B]`.
   * Ô tìm kiếm thông minh theo tên sản phẩm và nút tải toàn bộ luật về file `.csv`.
   * Biểu đồ Scatter Plot tương tác phân bố chất lượng luật (Confidence vs Lift).
3. **🌳 Tab 3 — Phân Tích Tập Phổ Biến FP-Growth (Frequent Itemsets):**
   * Xem danh sách các tập $k$-itemsets phổ biến nhất và kiểm tra độ dài tập món.
4. **⚡ Tab 4 — So Sánh & Thống Kê Hiệu Năng (Benchmark Performance):**
   * Trực quan hóa các biểu đồ cột Runtime và RAM tiêu thụ, chứng minh lý do FP-Growth được lựa chọn thay thế Apriori.
5. **📊 Tab 5 — Lịch Sử Thực Nghiệm Offline:**
   * Tra cứu lại bảng số liệu đo đạc thực nghiệm từ thư mục `results/`.

---

## 📈 8. Một Số Phát Hiện & Luật Kết Hợp Tiêu Biểu trên Instacart

Chạy FP-Growth trên tập dữ liệu Instacart ghi nhận những quy luật mua sắm có giá trị kinh doanh rất cao:

* **Nhóm rau củ hữu cơ & gia vị:**
  $$\text{Organic Cilantro} \rightarrow \text{Limes} \quad (\text{Confidence: } 27.4\%, \text{Lift: } 5.69\times)$$
  $$\text{Organic Garlic} \rightarrow \text{Organic Yellow Onion} \quad (\text{Confidence: } 21.4\%, \text{Lift: } 5.68\times)$$
  * *Ý nghĩa kinh doanh:* Khách hàng mua rau mùi hữu cơ có xu hướng mua chanh gấp gần 5.7 lần người bình thường (nhóm khách nấu món salad/món cuốn).
* **Nhóm bữa sáng & trái cây tươi:**
  $$\text{Organic Hass Avocado} \rightarrow \text{Bag of Organic Bananas} \quad (\text{Confidence: } 38.5\%, \text{Lift: } 2.40\times)$$
  $$\text{Organic Strawberries} \rightarrow \text{Banana} \quad (\text{Confidence: } 29.1\%, \text{Lift: } 1.82\times)$$
  * *Ý nghĩa kinh doanh:* Bơ và chuối là combo ăn sáng và làm sinh tố phổ biến nhất tại siêu thị.

---

## 👥 9. Thông Tin Đồ Án
* **Môn học:** Khai phá dữ liệu (Data Mining)
* **Chủ đề:** Dự đoán giỏ hàng và Khai phá luật kết hợp bằng thuật toán FP-Growth
* **Ngôn ngữ & Công cụ:** Python 3.10+, mlxtend, Streamlit, Pandas, Plotly, PyArrow.
