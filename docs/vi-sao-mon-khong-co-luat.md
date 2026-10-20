# Vì sao chọn Soda mà hệ thống lại gợi ý Banana?

**Tình huống:** Trên dashboard, cho Soda vào giỏ thì các gợi ý hiện ra là Banana, Bag of Organic Bananas, Organic Strawberries… Những món này chẳng liên quan gì đến Soda.

**Kết luận:** Hệ thống không có luật nào cho Soda. Khi không có luật, dashboard chuyển sang gợi ý **các món bán chạy chung** (nhãn viền nét đứt, ghi "bán chạy"). Danh sách này không phải gợi ý mua kèm cho Soda. Nguyên nhân là bản chất của luật kết hợp trên một danh mục rất lớn, không phải lỗi thuật toán.

## Vì sao Soda không có luật

Cấu hình: 10k đơn mức sản phẩm, support 1%, confidence 10%, lift 1,2.

- Soda có trong **102 đơn (1,02%)**, vừa qua ngưỡng 1%, nên bản thân Soda là món phổ biến.
- Muốn có luật Soda → X thì **cặp {Soda, X} cũng phải đạt support 1%**, tức xuất hiện trong khoảng 100 đơn.
- Các món hay đi cùng Soda nhất lại rất hiếm:

| Món hay mua cùng Soda | Số đơn có cả hai | Support của cặp | Confidence | Lift |
|---|---|---|---|---|
| Zero Calorie Cola | 16 | 0,16% | 16% | 41,3 |
| Popcorn | 10 | 0,10% | 10% | 36,3 |
| Extra Fancy Unsalted Mixed Nuts | 7 | 0,07% | 7% | 24,5 |
| Clementines | 7 | 0,07% | 7% | 6,1 |

Các cặp này **có liên quan thật**: lift rất cao, soda đi cùng cola, bỏng ngô và hạt. Nhưng chúng quá hiếm nên không qua được ngưỡng support. Hạ support xuống 0,2% (20 đơn) cũng chưa đủ.

## Đây là hạn chế chung, không riêng Soda

Số món có ít nhất một luật (xuất hiện ở vế trái), trên tổng **16.516 mặt hàng**:

| Support | Số món có luật | Tỉ lệ |
|---|---|---|
| 1% (mặc định) | 11 | 0,1% |
| 0,5% | 46 | 0,3% |
| 0,2% | 158 | 1,0% |

Danh mục Instacart có **đuôi dài**: vài món như chuối, dâu, bơ, rau bina xuất hiện trong rất nhiều đơn, còn hàng chục nghìn món khác mỗi món chỉ có trong vài đơn. Ngưỡng support loại hết các cặp có món ở phần đuôi, nên luật chỉ phủ được nhóm món bán chạy nhất.

## Hệ quả và cách xử lý

1. **Cho báo cáo:** nêu rõ hạn chế này trong phần đánh giá. Nó cũng giải thích vì sao gợi ý theo luật chỉ đến được khoảng 47% số đơn kiểm tra (xem [danh-gia-goi-y.md](danh-gia-goi-y.md)).
2. **Khi demo:**
   - Chọn món có trong nút chọn nhanh, vì đó là những món có nhiều luật nhất.
   - Hoặc chuyển sang bộ dữ liệu **mức kệ hàng (aisle)**. Ở mức này chỉ có 133 kệ (đã bỏ nhãn `missing`), nên mỗi kệ xuất hiện trong nhiều đơn hơn và luật phủ được nhiều kệ hơn hẳn.
3. **Có thể cải tiến dashboard (chưa làm):**
   - Khi một món không có luật, nói rõ lý do. Ví dụ: "Soda chưa có luật: cặp phổ biến nhất (Soda + Zero Calorie Cola) chỉ có 16 đơn = 0,16%, dưới ngưỡng 1%." Đồng thời tách hẳn khối "món bán chạy chung" ra khỏi khối gợi ý mua kèm.
   - Đánh dấu và đưa lên đầu danh sách chọn những món có luật.
