# 🛒 Hệ Thống Dự Đoán & Gợi Ý Sản Phẩm Mua Kèm Bằng Thuật Toán FP-Growth (Instacart Market Basket Analysis)

> **Đề tài:** Khai phá luật kết hợp (Association Rule Mining) và Xây dựng hệ thống gợi ý bán kèm sản phẩm (Cross-Selling Recommender System) sử dụng thuật toán **FP-Growth** tự cài đặt trên bộ dữ liệu **Instacart Market Basket Analysis**.

---

## 📌 1. Giới thiệu Đề tài & Mục tiêu Bài toán

Trong ngành bán lẻ và thương mại điện tử (E-commerce / Supermarket), việc thấu hiểu hành vi mua sắm của khách hàng đóng vai trò sống còn để tối ưu hóa doanh thu và nâng cao trải nghiệm mua hàng.

Bài toán **Phân tích Giỏ hàng (Market Basket Analysis - MBA)** đặt ra câu hỏi nghiệp vụ cốt lõi:
> *"Khi khách hàng đã chọn mua sản phẩm **A**, họ có xu hướng mua kèm thêm sản phẩm **B** nào nhất?"*

### Mục tiêu chính của dự án:
1. **Tự cài đặt thuật toán FP-Growth tối ưu (`src/my_fpgrowth.py`):** Xây dựng cấu trúc cây tiền tố **FP-Tree** thuần Python với các kỹ thuật tối ưu hóa cao độ (`__slots__`, Integer ID Mapping, chuẩn hóa ngưỡng trần `math.ceil`), khai phá đệ quy không cần sinh tập ứng viên ($C_k$).
2. **Khai phá các quy luật mua sắm ẩn (Association Rules):** Tìm ra mạng lưới quan hệ giữa các mặt hàng từ hàng triệu đơn hàng của Instacart thông qua các chỉ số định lượng: *Support*, *Confidence*, và *Lift*.
3. **Dự đoán & Gợi ý sản phẩm mua kèm (Product Recommender):** Dựa trên giỏ hàng hiện tại của khách, hệ thống tính toán và xếp hạng các sản phẩm mua kèm tiềm năng nhất theo điểm $\text{Score} = \text{Lift} \times \text{Confidence}$, kèm câu diễn giải kinh doanh tự nhiên.
4. **Xây dựng Dashboard phân tích trực quan:** Cung cấp giao diện Streamlit phong cách **Light Analytics** (Trắng, Navy, Teal) với 3 nghiệp vụ tập trung, đảm bảo 100% số liệu hiển thị đều lấy từ kết quả chạy thực tế.

---

## ⚡ 2. Tại sao chọn duy nhất thuật toán FP-Growth?

Khai phá luật kết hợp có hai trường phái kinh điển là **Apriori** và **FP-Growth**. Dự án **lựa chọn FP-Growth làm thuật toán hạt nhân duy nhất** để triển khai hệ thống dự đoán và gợi ý vì những ưu thế vượt trội:

| Tiêu chí | Apriori (Cổ điển) | FP-Growth (Lựa chọn của dự án) |
| :--- | :--- | :--- |
| **Độ chính xác của luật** | Chuẩn xác 100% | **Chuẩn xác 100% (kết quả toán học tương đương)** |
| **Số lần quét cơ sở dữ liệu** | Quét $k$ lần (với $k$ là độ dài giỏ hàng lớn nhất) | **Chỉ quét đúng 2 lần duy nhất** |
| **Cơ chế tìm kiếm** | Sinh tập ứng viên tổ hợp $C_k$ khổng lồ $\rightarrow$ bùng nổ tổ hợp | **Không sinh ứng viên**, nén dữ liệu vào cây FP-Tree |
| **Bộ nhớ RAM tiêu thụ** | Rất tốn RAM khi ngưỡng support giảm sâu | **Tiết kiệm vượt trội**, nén hàng chục nghìn đơn vào cây phân cấp |
| **Độ phức tạp tính toán** | $O(2^d)$ theo số lượng mặt hàng | Chia để trị trên các cây điều kiện nhỏ gọn (**Divide-and-Conquer**) |

> **Kết luận:** FP-Growth loại bỏ hoàn toàn giai đoạn sinh tập ứng viên tổ hợp tốn kém, giúp hệ thống khai phá tập phổ biến và phản hồi gợi ý mua kèm gần như tức thì.

---

## 🧠 3. Động Cơ Khai Phá & Cơ Chế Hoạt Động (`src/my_fpgrowth.py`)

Thuật toán hạt nhân của dự án được triển khai hoàn chỉnh trong module [`src/my_fpgrowth.py`](src/my_fpgrowth.py) với 4 kỹ thuật tối ưu hóa cốt lõi:

### 3.1. Kỹ thuật tối ưu hóa thuật toán
1. **Integer ID Mapping:** Ánh xạ tên sản phẩm (chuỗi ký tự) sang số nguyên `int32`. Các phép so sánh trên cây chuyển thành so sánh cấp CPU ($O(1)$), giảm hơn 70% bộ nhớ so với việc lưu trữ chuỗi.
2. **Cấu trúc nút `__slots__` trong `FPNode`:** Loại bỏ từ điển `__dict__` nội tại của từng đối tượng Python, tiết kiệm 50% RAM trên mỗi nút cây khi cây chứa hàng chục nghìn nhánh.
3. **Chuẩn hóa ngưỡng đếm bằng `math.ceil`:** 
   $$\text{min\_count} = \max\left(1, \lceil \text{min\_support} \times N \rceil\right)$$
   Đảm bảo điều kiện tần suất xuất hiện nghiêm ngặt, không bị làm tròn xuống làm lọt các món dưới ngưỡng hỗ trợ.
4. **Header Table & Node-Links:** Bảng đầu mục lưu con trỏ danh sách liên kết trỏ tới toàn bộ các nút cùng mã hàng trên cây, cho phép truy vết Conditional Pattern Base tức thì mà không cần duyệt lại toàn bộ cây.

### 3.2. Trích xuất Luật Kết Hợp (Association Rules)
Từ các tập phổ biến, hệ thống sinh ra các luật dạng $A \rightarrow B$ (*Nếu mua $A$ thì sẽ mua kèm $B$*):
* **Support (Độ hỗ trợ):** Tỷ lệ đơn hàng chứa đồng thời cả $A$ và $B$.
  $$\text{Support}(A \rightarrow B) = P(A \cup B) = \frac{\text{Số đơn chứa cả } A \text{ và } B}{\text{Tổng số đơn hàng}}$$
* **Confidence (Độ tin cậy):** Xác suất mua $B$ khi đã chọn mua $A$.
  $$\text{Confidence}(A \rightarrow B) = P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
* **Lift (Độ nâng):** Thước đo mức độ thúc đẩy của $A$ lên hành vi mua $B$.
  $$\text{Lift}(A \rightarrow B) = \frac{P(A \cup B)}{P(A) \times P(B)} = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
  * $\text{Lift} > 1$: Tương quan thuận mạnh mẽ (mua $A$ làm tăng khả năng mua $B$).
  * $\text{Lift} = 1$: Hai sản phẩm độc lập.
  * $\text{Lift} < 1$: Tương quan nghịch.

### 3.3. Thuật toán Dự đoán Sản phẩm Mua kèm (Recommender Engine)
Khi khách hàng có giỏ hàng hiện tại $C = \{item_1, item_2, ...\}$:
1. Lọc tất cả các luật $A \rightarrow B$ thỏa mãn $A \subseteq C$ (tiền đề nằm trọn trong giỏ hàng).
2. Lấy tập sản phẩm đề xuất chưa có trong giỏ: $B_{\text{candidate}} = B \setminus C$.
3. Xếp hạng sản phẩm theo điểm ưu tiên kết hợp:
   $$\text{Score} = \text{Lift} \times \text{Confidence}$$
4. Xuất kết quả kèm theo câu giải thích nghiệp vụ tự nhiên và đề xuất hành động bán lẻ (Combo ưu đãi, Cross-selling, Sắp xếp quầy kệ).

---

## 📁 4. Cấu trúc Thư mục Dự án

```text
apriori-vs-fpgrowth-instacart/
├── data/
│   ├── raw/                 # Dữ liệu gốc Instacart (.csv)
│   └── processed/           # Dữ liệu giỏ hàng lồng nhau (.parquet & .csv: 10k, 50k, 100k)
│       ├── transactions_product_10k.parquet / .csv   # Mẫu 10k đơn (mức sản phẩm)
│       ├── transactions_product_50k.parquet / .csv   # Mẫu 50k đơn (mức sản phẩm)
│       └── transactions_product_100k.parquet / .csv  # Mẫu 100k đơn (mức sản phẩm)
├── src/
│   ├── __init__.py          # Khởi tạo package
│   ├── my_fpgrowth.py       # [CỐT LÕI] Thuật toán FP-Growth tự cài đặt (FP-Tree + slots + Recommender)
│   └── preprocess.py        # Tiền xử lý đơn hàng, lọc giỏ >= 2 món, xuất .parquet & .csv
├── dashboard/
│   └── app.py               # Web Dashboard phân tích giỏ hàng & gợi ý mua sắm (3 nghiệp vụ)
├── notebooks/
│   └── 01_eda.ipynb         # Khám phá phân bố dữ liệu Instacart và hiện tượng đuôi dài
├── results/                 # Dữ liệu thực nghiệm offline lưu trữ (.csv)
├── requirements.txt         # Danh sách thư viện (pandas, numpy, pyarrow, fastparquet, streamlit...)
├── .streamlit/
│   └── config.toml          # Cấu hình giao diện Light Analytics (Trắng, Navy, Teal)
└── README.md                # Tài liệu hướng dẫn đồ án
```

---

## 🚀 5. Hướng dẫn Cài đặt & Chuẩn bị Môi trường

### Bước 1: Khởi tạo môi trường ảo Python
Khuyến nghị sử dụng Python 3.9, 3.10 hoặc 3.11:

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo:
# Trên macOS / Linux:
source .venv/bin/activate

# Trên Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt các thư viện phụ thuộc
Đảm bảo đã bao gồm `pyarrow` và `fastparquet` để đọc ghi dữ liệu Parquet:
```bash
pip install -r requirements.txt
```

---

## 📊 6. Chuẩn bị Dữ liệu (Instacart Dataset)

1. Tải bộ dữ liệu từ Kaggle: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data).
2. Giải nén đầy đủ 6 tệp `.csv` vào thư mục `data/raw/`:
   * `order_products__prior.csv`, `order_products__train.csv`
   * `products.csv`, `orders.csv`, `aisles.csv`, `departments.csv`
3. Chạy quy trình tiền xử lý và sinh mẫu dữ liệu:
   ```bash
   python src/preprocess.py
   ```
   * **Cơ chế tiền xử lý:**
     * Lọc bỏ các đơn hàng chỉ có 1 món (đơn 1 món không thể tạo thành luật mua kèm $A \rightarrow B$).
     * Áp dụng **Lấy mẫu lồng nhau (Nested Sampling)** với `seed=42`: Tập 10k là tập con của 50k, 50k là tập con của 100k ($10k \subset 50k \subset 100k$), giúp tần suất mặt hàng ổn định giữa các quy mô.
     * Tự động xuất đồng thời tệp `.parquet` (nạp siêu tốc) và `.csv` (để tra cứu).

---

## 🖥️ 7. Hướng Dẫn Sử Dụng Web Dashboard (Streamlit)

Khởi chạy ứng dụng Web Dashboard phân tích bằng lệnh:

```bash
streamlit run dashboard/app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

Giao diện được thiết kế theo phong cách **Light Analytics** (Trắng `#FFFFFF`, Navy `#1E3A8A`, Teal `#0D9488`), độ tương phản cao, tập trung trực diện vào **3 nghiệp vụ kinh doanh bán lẻ cốt lõi**:

### 🛍️ Nghiệp vụ 1 — Gợi Ý Sản Phẩm Mua Kèm (Product Recommender)
* **Thử nhanh giỏ hàng mẫu:** 4 nút bấm 1-click chọn nhanh các combo nông sản phổ biến (*Trái cây & Berry*, *Salad & Rau củ tươi*, *Táo & Hoa quả*, *Bữa sáng bổ dưỡng*).
* **Bộ chọn giỏ hàng đa sản phẩm:** Cho phép chọn cùng lúc nhiều món hàng từ top sản phẩm phổ biến nhất.
* **Thẻ gợi ý thông minh:**
  * Xếp hạng món mua kèm (#1, #2, #3...) theo điểm $\text{Score} = \text{Lift} \times \text{Confidence}$.
  * Hệ thống huy hiệu: Score, Xác suất mua kèm (Confidence %), Độ nâng (Lift ×), Tần suất combo (Support %).
  * **Giải thích tự nhiên:** *"Dựa trên N đơn hàng thực tế: Khi khách hàng đã chọn [A], xác suất mua thêm [B] đạt X%, cao gấp Y lần so với mua ngẫu nhiên."*
  * **Hành động kinh doanh khuyến nghị:** Tự động phân loại chiến lược dựa trên Lift (*Combo Siêu Gắn Kết*, *Bán Chéo Trực Tiếp*, hoặc *Sắp Xếp Quầy Kệ*).

### 📜 Nghiệp vụ 2 — Khám Phá & Tra Cứu Luật Kết Hợp (Association Rules Explorer)
* **Tổng quan chỉ số thực tế:** Số luật tìm thấy, Độ tin cậy trung bình, Độ nâng trung bình, Số luật tương quan mạnh (Lift $\ge$ 2.0).
* **Biểu đồ Scatter Plot tương tác:** Trục X là Độ tin cậy (Confidence %), Trục Y là Độ nâng (Lift), kích thước bóng thể hiện Tần suất (Support %).
* **Top 10 Cặp mặt hàng có Lift cao nhất:** Biểu đồ thanh ngang thể hiện các liên kết mua sắm chặt chẽ nhất.
* **Bộ lọc thông minh 2 chiều:** Tìm kiếm theo món tiền đề (*Nếu mua món A*) hoặc món hệ quả (*Sẽ mua kèm món B*).
* **Xuất dữ liệu:** Nút **📥 Tải Danh Sách Luật Kết Hợp (CSV)** phục vụ báo cáo.

### 🌳 Nghiệp vụ 3 — Phân Tích Tập Phổ Biến & Cây Tiền Tố FP-Tree
* **Chỉ số tập mục:** Tổng số tập phổ biến, số mặt hàng đơn lẻ (1-item), số combo mua chung ($\ge 2$-item), kích thước combo dài nhất.
* **Biểu đồ phân bố:** Phân bố số lượng theo độ dài combo ($k$-itemset) và Top 15 tập phổ biến xuất hiện nhiều nhất.
* **Trực quan hóa cơ chế nén 3 bước của FP-Tree:** Minh họa trực quan quy trình nén dữ liệu và chia để trị của cây tiền tố FP-Tree.
* **Bảng tra cứu tập phổ biến:** Tìm kiếm theo tên sản phẩm, xem số lượng đơn hàng thực tế chứa từng combo.

> **Cam kết dữ liệu:** Mọi số liệu trên cả 3 nghiệp vụ đều được tính toán động trực tiếp từ dữ liệu giao dịch thực tế thông qua động cơ `src/my_fpgrowth.py`, không sử dụng bất kỳ số liệu minh họa giả lập nào.

---

## 📈 8. Một Số Phát Hiện & Luật Kết Hợp Tiêu Biểu trên Instacart

Chạy thuật toán FP-Growth trên dữ liệu Instacart ghi nhận những quy luật mua sắm có giá trị kinh doanh rất cao:

* **Nhóm rau củ hữu cơ & gia vị nấu ăn:**
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
* **Chủ đề:** Hệ thống Phân tích Giỏ hàng & Gợi ý Mua sắm bằng Thuật toán FP-Growth
* **Động cơ thuật toán:** `src/my_fpgrowth.py` (Tự cài đặt - Pure Python FP-Tree)
* **Giao diện:** Streamlit (Light Analytics Theme)
* **Thư viện chính:** Python 3.9+, Pandas, Plotly, PyArrow, FastParquet.
