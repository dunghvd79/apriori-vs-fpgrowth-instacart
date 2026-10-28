# Kiểm chứng FP-Growth tự cài đặt

**Câu hỏi:** `src/my_fpgrowth.py` có tìm đúng và đủ các tập phổ biến và luật kết hợp không?

**Kết luận:** Đúng. Trên 4 cấu hình thử, các tập phổ biến (kèm giá trị support) và các luật đều trùng khớp 100% với thư viện chuẩn `mlxtend`.

## Cách kiểm tra

Chạy cả hai cài đặt trên cùng một tập giao dịch rồi so sánh:

1. **Tập phổ biến:** so sánh tập hợp `{itemset: support}` (support làm tròn 9 chữ số) giữa `OptimizedFPGrowth(...).get_frequent_itemsets()` và `mlxtend.frequent_patterns.fpgrowth(...)`.
2. **Luật:** so sánh tập hợp các cặp `(antecedents, consequents)` giữa `generate_rules(fi, min_confidence=0.2, min_lift=1.2)` và `mlxtend.frequent_patterns.association_rules(..., metric="confidence", min_threshold=0.2)`, sau đó lọc `lift >= 1.2`.

## Kết quả

| Dữ liệu | Support | Tập phổ biến (của nhóm / mlxtend) | Luật (của nhóm / mlxtend) | Thời gian (của nhóm / mlxtend) |
|---|---|---|---|---|
| Sản phẩm, 10k đơn | 1% | 136 / 136 ✅ | 13 / 13 ✅ | 0,05 s / 2,1 s |
| Sản phẩm, 50k đơn | 0,5% | 368 / 368 ✅ | 57 / 57 ✅ | 0,3 s / 28 s |
| Sản phẩm, 10k đơn | 0,2% | 1.698 / 1.698 ✅ | 332 / 332 ✅ | 0,2 s / 1,2 s |
| Sản phẩm, 100k đơn | 0,2% | 1.538 / 1.538 ✅ | 234 / 234 ✅ | 2,2 s / 11 s |

Hai cấu hình cuối là ca khó nhất: hàng nghìn tập phổ biến, có tập 3 món, cây điều kiện đệ quy nhiều tầng. (Trước 28/10/2026 nhóm còn kiểm trên dữ liệu mức kệ hàng; phần đó đã bỏ cùng với mức kệ.)

Bản của nhóm nhanh hơn mlxtend ở mức sản phẩm vì mlxtend phải mã hóa dữ liệu thành ma trận one-hot có khoảng 16.000 cột. Đây không phải phép đo hiệu năng chuẩn (chỉ chạy một lần, không cô lập tiến trình), nên không nên đưa vào báo cáo như kết quả benchmark.

## Lỗi đã phát hiện và đã sửa

**Ngưỡng `min_count` bị làm tròn xuống** (commit `5389961`):

```python
# Trước: int() làm tròn xuống, nên món dưới ngưỡng vẫn lọt
min_count = max(1, int(self.min_support * self.n_transactions))
# Sau
min_count = max(1, math.ceil(round(self.min_support * self.n_transactions, 9)))
```

Ví dụ: 5 giao dịch với `min_support = 0.3` thì cần ít nhất 1,5, tức 2 giao dịch. Bản cũ lấy 1, nên Apple và Eggs (support 0,2) bị tính nhầm là phổ biến. `round(..., 9)` tránh sai số dấu phẩy động, ví dụ `0.07 * 100 = 7.000000000000001` sẽ bị `ceil` thành 8.

## Các điểm khác đã rà

- **Món trùng trong một giỏ:** không xảy ra, vì `preprocess.py` đã `drop_duplicates` theo `(order_id, item)`.
- **Sinh luật:** dùng đúng công thức. Support(A→B) = Support(A∪B); Confidence = Support(A∪B) / Support(A); Lift = Confidence / Support(B). Mọi tập con của một tập phổ biến đều phổ biến (tính chất Apriori), nên tra cứu support của vế trái và vế phải luôn có kết quả.
- **Hàm gợi ý `recommend_products`:** lọc các luật có vế trái nằm trọn trong giỏ, bỏ những món đã có trong giỏ, rồi xếp hạng theo `lift × confidence`. Nếu nhiều luật cùng gợi ý một món, hàm giữ luật có điểm cao nhất.
