# Tài liệu dự án

Thư mục này ghi lại các vấn đề nhóm đã thảo luận và kiểm chứng trong quá trình làm đề tài, để ai đọc lại cũng hiểu được *vì sao* code và báo cáo được viết như hiện tại.

| Tài liệu | Trả lời câu hỏi |
|---|---|
| [kiem-chung-thuat-toan.md](kiem-chung-thuat-toan.md) | FP-Growth tự cài đặt (`src/my_fpgrowth.py`) có cho kết quả đúng không? |
| [vi-sao-mon-khong-co-luat.md](vi-sao-mon-khong-co-luat.md) | Vì sao chọn Soda mà hệ thống lại gợi ý Banana? |
| [danh-gia-goi-y.md](danh-gia-goi-y.md) | Gợi ý theo luật có tốt hơn gợi ý món bán chạy không? |
| [thiet-ke-dashboard.md](thiet-ke-dashboard.md) | Dashboard đã đổi những gì, và vì sao? |

## Bối cảnh

- **Đề tài hiện tại:** Hệ thống gợi ý sản phẩm mua kèm bằng FP-Growth trên dữ liệu Instacart. Đề tài cũ là "So sánh Apriori và FP-Growth", nhóm đã đổi hướng vào ngày 20/10/2026.
- **Dữ liệu:** các file trong `data/processed/` do `src/preprocess.py` sinh ra. Gồm mẫu lồng nhau 10k ⊂ 50k ⊂ 100k đơn ở mức sản phẩm, seed 42, chỉ giữ giỏ có từ 2 món trở lên.
- **Các con số trong tài liệu** đo ngày 20/10/2026, trên bộ 10k đơn mức sản phẩm, trừ khi có ghi khác.
- **Số liệu Chương 4 của báo cáo** sinh ra bằng `python scripts/run_experiments.py` (khoảng 3 phút). Bảng được ghi vào `results/*.csv`, hình vào `results/figures/`. Thư mục `results/` nằm trong `.gitignore`, chạy lại script là có lại đủ.

## Quy ước khi thêm tài liệu mới

- Mỗi file một vấn đề. Tên file viết không dấu, nối bằng gạch ngang.
- Mở đầu bằng câu hỏi hoặc vấn đề, sau đó là kết luận, rồi mới đến số liệu và cách kiểm tra.
- Ghi rõ cấu hình (bộ dữ liệu, support, confidence, lift) cho mọi con số, để người khác chạy lại được.
- Thêm một dòng vào bảng ở trên.
