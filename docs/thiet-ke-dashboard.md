# Thiết kế dashboard

**Câu hỏi:** Dashboard (`dashboard/app.py`) đã đổi những gì so với bản trước commit `ee3895b`, và vì sao?

Chạy dashboard: `streamlit run dashboard/app.py`, mở `http://localhost:8501`.

## Cấu trúc

| Tab | Dùng để |
|---|---|
| Gợi ý mua kèm | Chọn giỏ hàng và xem gợi ý. Có nút chọn nhanh và nút "Lấy một đơn hàng thật và giấu 1 món" để demo dự đoán. |
| Luật kết hợp | Biểu đồ confidence theo lift, 10 cặp món gắn kết nhất, tra cứu và tải các luật về file CSV. |
| Tập phổ biến | Phân bố số tập theo độ dài, các combo phổ biến nhất, tra cứu tập phổ biến. |
| Đánh giá | Holdout 80/20, so sánh luật với món bán chạy (xem [danh-gia-goi-y.md](danh-gia-goi-y.md)). |

Thanh bên chứa bộ dữ liệu, số đơn đưa vào khai phá và các ngưỡng support, confidence, lift. Mọi tab dùng chung các ngưỡng này.

## Lỗi logic đã sửa

| Vấn đề ở bản cũ | Cách sửa |
|---|---|
| Giỏ không khớp luật nào thì không có gợi ý | Gợi ý dự phòng bằng món bán chạy, hiển thị nhãn viền nét đứt để phân biệt với gợi ý từ luật |
| 4 giỏ mẫu cố định: "Bữa Sáng Bổ Dưỡng" (Organic Whole Milk) ra 0 gợi ý, và cả 4 giỏ hỏng khi chọn dữ liệu mức kệ hàng | Nút chọn nhanh lấy 4 món có nhiều luật nhất từ dữ liệu đang dùng |
| Biểu đồ "Top 10 cặp" có 10 thanh nhưng chỉ là 6 cặp khác nhau, vì A→B và B→A luôn có cùng lift | Gộp hai chiều thành một cặp, kèm confidence của từng chiều khi rê chuột |
| Câu giải thích "cao gấp X lần so với mua ngẫu nhiên" | Đổi thành "cao gấp X lần tỉ lệ mua món này nói chung", đúng với định nghĩa lift = confidence / support(B) |
| Nhãn chiến lược Combo / Bán chéo / Trưng bày dựa trên ngưỡng lift 2,5 và 1,8 do nhóm tự đặt | Bỏ, vì không suy ra từ dữ liệu, mâu thuẫn với cam kết "số liệu từ dữ liệu thực tế" |
| Mỗi bộ dữ liệu xuất hiện 2 lần (parquet và csv) | Chỉ liệt kê parquet, chỉ dùng csv khi không có parquet |
| Chưa có phần đánh giá cho một hệ thống "dự đoán" | Thêm tab Đánh giá |

## Lựa chọn thiết kế

- **Chủ đề:** quầy rau củ và nhãn giá trên kệ siêu thị, gắn với dữ liệu là đơn hàng tạp hóa.
- **Màu sắc:**

  | Vai trò | Màu |
  |---|---|
  | Nền | `#F7F9F4` |
  | Chữ | `#1C2B22` |
  | Màu chủ đạo (xanh kale) | `#2F6B42` |
  | Con số lift (cam cà rốt) | `#E2711D` |
  | Chuỗi thứ hai trong biểu đồ (tím củ dền) | `#7E2E58` |

  Theme trong `.streamlit/config.toml` dùng cùng bảng màu này.
- **Font:** Bricolage Grotesque cho tiêu đề và các con số lớn, Public Sans cho nội dung. Chữ số tabular để cột số thẳng hàng.
- **Một điểm nhấn duy nhất:** mỗi gợi ý là một **nhãn giá trên kệ**, với con số lift to như giá tiền. Các phần còn lại giữ yên tĩnh: không emoji, không chữ in hoa, không bóng đổ trên mọi thẻ.
- **Tiêu đề trang chính là câu hỏi của bài toán:** "Khách bỏ món này vào giỏ, nên gợi ý món gì?"
- **Hiển thị trên điện thoại:** thanh bên tự thu gọn trên màn hình nhỏ. Ô "Các món trong giỏ" cho các món xuống dòng, kể cả tên dài, thay vì cuộn ngang như mặc định của Streamlit 1.65.

## Ghi chú kỹ thuật

- Tên sản phẩm được `html.escape` trước khi chèn vào HTML của nhãn gợi ý.
- `use_container_width` đã đổi sang `width="stretch"`, vì Streamlit bỏ tham số cũ sau 31/12/2025.
- Kết quả khai phá và đánh giá được lưu đệm bằng `st.cache_data`. Đổi ngưỡng sẽ tính lại, giữ nguyên ngưỡng thì dùng lại kết quả đã có.
- CSS cho ô chọn món dựa vào `data-testid="stMultiSelectTagsContainer"` của Streamlit 1.65. Nếu nâng cấp Streamlit mà ô chọn lại cuộn ngang, hãy kiểm tra selector này trước.
