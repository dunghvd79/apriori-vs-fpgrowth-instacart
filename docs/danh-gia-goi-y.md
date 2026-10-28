# Đánh giá chất lượng gợi ý

**Câu hỏi:** Đề tài gọi đây là hệ thống *dự đoán* sản phẩm mua kèm. Vậy gợi ý theo luật có đoán đúng món khách sẽ mua không, và có tốt hơn cách đơn giản nhất là gợi ý món bán chạy không?

**Kết luận** (số liệu chính thức trong báo cáo lấy từ bộ 100k, mục 4.4):
- **Chỉ dùng luật thì chưa vượt** mốc gợi ý món bán chạy, vì độ phủ thấp (xem [vi-sao-mon-khong-co-luat.md](vi-sao-mon-khong-co-luat.md)).
- **Luật kèm gợi ý dự phòng bằng món bán chạy thì vượt mốc.** Ở mức sản phẩm, support 0,2%, Hit@5 đạt **6,17%** so với 5,49%; Hit@10 đạt 8,77% so với 8,01%.
- Ở vị trí gợi ý đầu tiên (Hit@1), riêng luật đã đoán trúng nhiều hơn món bán chạy: 1,82% so với 1,60%.

Tái lập: `python scripts/run_experiments.py --only eval` sẽ ghi kết quả ra `results/eval_holdout.csv` và các hình vào `results/figures/`.

## Phương pháp (holdout, giấu 1 món)

1. Xáo trộn các đơn với seed cố định 42, chia **80% để khai phá luật** và **20% để kiểm tra**.
2. Với mỗi đơn kiểm tra, **giấu ngẫu nhiên 1 món**. Các món còn lại đóng vai trò giỏ hàng hiện tại.
3. Cho hệ thống gợi ý k món (mặc định k = 5), rồi kiểm tra món bị giấu có nằm trong danh sách không.
4. So sánh 3 cách gợi ý:
   - **Chỉ luật kết hợp:** dùng `recommend_products`, xếp hạng theo lift × confidence.
   - **Chỉ món bán chạy:** k món phổ biến nhất trong phần huấn luyện mà chưa có trong giỏ.
   - **Luật + dự phòng:** lấy gợi ý từ luật trước, thiếu bao nhiêu thì bổ sung bằng món bán chạy.

Hai chỉ số:
- **Coverage:** tỉ lệ đơn kiểm tra nhận được ít nhất 1 gợi ý.
- **Hit@k:** tỉ lệ đơn kiểm tra mà món bị giấu nằm trong k gợi ý.

Tab **Đánh giá** trên dashboard chạy đúng quy trình này với các ngưỡng đang chọn ở thanh bên.

## Kết quả

**Bộ 10k đơn** (dashboard mặc định: support 1%, confidence 10%, lift 1,2, k = 5; 8.000 đơn huấn luyện, 2.000 đơn kiểm tra, 28 luật):

| Cách gợi ý | Coverage | Hit@5 | Hit@5 trên các đơn có gợi ý |
|---|---|---|---|
| Chỉ luật kết hợp | 46,8% | 2,70% | 5,77% |
| Chỉ món bán chạy | 100% | 5,50% | 5,50% |
| Luật + dự phòng | 100% | 5,50% | 5,50% |

**Bộ 50k đơn** (40k huấn luyện, 10k kiểm tra; confidence 10%, lift 1,2, k = 5):

| Support | Số luật | Coverage của luật | Hit@5 của luật | Hit@5 của món bán chạy |
|---|---|---|---|---|
| 1% | 28 | 46,2% | 2,76% | 6,20% |
| 0,5% | 104 | 61,4% | 4,14% | 6,20% |
| 0,2% | 647 | 75,3% | 5,77% | 6,20% |

**Bộ 100k (số liệu dùng trong báo cáo):** 80.000 giỏ huấn luyện, 20.000 giỏ kiểm tra.

| Support | Số luật | Coverage của luật | Hit@5: chỉ luật | Hit@5: chỉ bán chạy | Hit@5: luật + dự phòng |
|---|---|---|---|---|---|
| 1% | 30 | 47,4% | 2,41% | 5,49% | 5,51% |
| 0,5% | 104 | 60,2% | 3,58% | 5,49% | 5,66% |
| 0,2% | 626 | 74,3% | 5,08% | 5,49% | **6,17%** |

Mức cải thiện đến từ việc hạ support:
- Ở support 1%, luật + dự phòng gần như bằng mốc món bán chạy: 5,51% so với 5,49% trên bộ 100k, và 5,50% so với 5,50% trên bộ 10k.
- Hạ support xuống 0,2% thì cách này mới vượt mốc rõ rệt.

## Cách đọc kết quả

- **Hạ support thì luật tốt dần lên** (nhiều luật hơn, coverage cao hơn), nhưng đến 0,2% vẫn còn dưới mốc món bán chạy.
- **Trên những đơn có gợi ý**, luật đúng khoảng 6–8%, ngang hoặc nhỉnh hơn món bán chạy. Điểm yếu chính của luật là **không có gì để gợi ý** cho phần lớn giỏ hàng.
- **Món bán chạy là mốc khó vượt trên Instacart**, vì chuối và dâu có trong hàng chục phần trăm số đơn.
- **Giá trị của luật không chỉ nằm ở Hit@k:** luật còn giải thích được *vì sao* nên gợi ý (confidence, lift), tìm ra các cặp gắn kết mạnh dù hiếm (Soda ↔ Zero Calorie Cola, lift 41) và gợi ý cách bày hàng. Báo cáo nên nêu cả hai mặt.

## Hướng thử thêm (chưa làm)

- Hạ support thấp hơn nữa (0,1%) trên bộ 100k đơn.
- Đổi cách xếp hạng: chỉ dùng confidence, hoặc ưu tiên luật có vế trái dài hơn.
