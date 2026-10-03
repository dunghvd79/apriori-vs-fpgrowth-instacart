# Khai Phá Luật Kết Hợp: So Sánh Apriori và FP-Growth trên Instacart Dataset

Dự án môn học **Khai phá dữ liệu (Data Mining)** - Đề tài: So sánh hiệu năng của thuật toán **Apriori** và **FP-Growth** trong bài toán phân tích giỏ hàng (Market Basket Analysis) trên bộ dữ liệu **Instacart Market Basket Analysis**.

---

## 📁 Cấu trúc thư mục

```text
apriori-vs-fpgrowth-instacart/
├── data/
│   ├── raw/                 # File CSV gốc tải từ Kaggle (được gitignore)
│   └── processed/           # File .parquet đã lọc/sample/tiền xử lý
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Tiền xử lý, gom giỏ hàng và lưu parquet
│   ├── my_apriori.py        # Tự cài đặt thuật toán Apriori
│   ├── my_fpgrowth.py       # Tự cài đặt thuật toán FP-Growth (FP-Tree)
│   ├── association_rules.py # Trích xuất và đánh giá luật kết hợp (confidence, lift)
│   └── benchmark.py         # Đo lường thời gian chạy và bộ nhớ RAM
├── dashboard/
│   └── app.py               # Streamlit Dashboard demo trực quan tương tác
├── notebooks/
│   ├── 01_eda.ipynb         # Khám phá và trực quan hóa phân bố dữ liệu
│   └── 02_experiments.ipynb # Thử nghiệm so sánh và kiểm chứng
├── results/                 # Lưu bảng thống kê benchmark, biểu đồ và tập luật
├── report/                  # Slide thuyết trình và file báo cáo (PDF/Word)
├── requirements.txt         # Thư viện cần thiết
├── .gitignore
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt và thiết lập môi trường

### 1. Tạo môi trường ảo Python
Khuyến nghị sử dụng Python 3.10 hoặc 3.11:

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Kích hoạt trên Command Prompt:
.\venv\Scripts\activate.bat
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 📊 Chuẩn bị dữ liệu (Instacart Dataset)

1. Tải bộ dữ liệu từ Kaggle: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data).
2. Giải nén các file `.csv` vào thư mục `data/raw/`:
   - `order_products__prior.csv`
   - `order_products__train.csv`
   - `products.csv`
   - `orders.csv`
   - `aisles.csv`
   - `departments.csv`

3. Kiểm tra trạng thái file:
   ```bash
   python src/preprocess.py --check_only
   ```

4. Tiền xử lý và tạo tập dữ liệu mẫu:
   ```bash
   # Lấy mẫu 50,000 đơn hàng từ tập prior để thử nghiệm
   python src/preprocess.py --source prior --sample 50000 --min_basket_size 2
   ```

---

## 🖥️ Chạy Demo Streamlit Dashboard

Sau khi hoàn thành huấn luyện/chạy luật, khởi chạy dashboard trực quan:
```bash
streamlit run dashboard/app.py
```

---

## 🎯 Các mục tiêu đánh giá & so sánh
1. **Thời gian thực thi (Execution Time):** So sánh theo các ngưỡng `min_support` giảm dần (ví dụ: 5%, 2%, 1%, 0.5%).
2. **Bộ nhớ tiêu thụ (Peak Memory):** Đánh giá mức độ tiết kiệm RAM nhờ nén cây tiền tố của FP-Tree so với sinh tập ứng viên của Apriori.
3. **Phân tích Luật (Actionable Rules):** Đưa ra các gợi ý sản phẩm bán kèm (Cross-selling) có giá trị thực tiễn cao cho siêu thị điện tử.
