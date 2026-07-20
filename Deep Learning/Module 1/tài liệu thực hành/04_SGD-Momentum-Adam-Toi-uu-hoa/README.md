# Bài 04 — Thuật toán cập nhật trọng số: SGD, Momentum, Adam

**Notebook tương ứng:** `thực hành/04_SGD-Momentum-Adam-Toi-uu-hoa/04_Toi_uu_hoa.ipynb`

## 1. Mục tiêu

Bài này thực hành phần nội dung: *Thuật toán cập nhật trọng số SGD* (và các biến thể phổ biến hơn: Momentum, Adam).

Sau khi hoàn thành, học viên có thể:
- Cài đặt Gradient Descent, Mini-batch Gradient Descent, Momentum và Adam từ đầu bằng NumPy.
- Hiểu vai trò của mini-batch (đánh đổi tốc độ/độ ổn định giữa batch GD và stochastic GD).
- Hiểu vì sao Momentum/Adam thường hội tụ nhanh và ổn định hơn GD thuần trên bề mặt loss gập ghềnh.
- Áp dụng learning rate decay (giảm dần learning rate theo epoch) để tinh chỉnh hội tụ ở giai đoạn cuối.

## 2. Chuẩn bị

- Đã hoàn thành bài 01 (forward/backward propagation cho mạng 3 lớp — notebook này dùng lại đúng kiến trúc đó qua `opt_utils_v1a.py`).
- Khái niệm exponentially weighted average (trung bình trượt có trọng số mũ) — nền tảng của Momentum và Adam.

## 3. Cách mở và chạy trên Google Colab

1. Vào `Deep Learning/Module 1/thực hành/04_SGD-Momentum-Adam-Toi-uu-hoa/` trên Drive.
2. Chuột phải file `.ipynb` → **Open with → Google Colaboratory**.
3. Chạy cell Colab setup đầu tiên (mount Drive + `cd`) — sửa `NOTEBOOK_DIR` nếu cần.
4. Run all / chạy tuần tự từng cell.

## 4. Nội dung chính (8 Exercise)

| Mục | Exercise | Nội dung |
|---|---|---|
| 2 | Exercise 1 | `update_parameters_with_gd` — cập nhật trọng số bằng gradient descent thuần |
| 3 | Exercise 2 | `random_mini_batches` — chia dữ liệu thành các mini-batch ngẫu nhiên |
| 4 | Exercise 3 | `initialize_velocity` — khởi tạo vận tốc (velocity) cho Momentum |
| 4 | Exercise 4 | `update_parameters_with_momentum` — cập nhật trọng số bằng Momentum |
| 5 | Exercise 5 | `initialize_adam` — khởi tạo `v`, `s` cho Adam |
| 5 | Exercise 6 | `update_parameters_with_adam` — cập nhật trọng số bằng Adam (bias-corrected) |
| 7.1 | Exercise 7 | `update_lr` — learning rate decay theo từng iteration |
| 7.2 | Exercise 8 | `schedule_lr_decay` — learning rate decay theo mốc cố định (fixed interval) |

Mục 6 huấn luyện cùng một mạng 3 lớp trên bộ dữ liệu "moons" 2D bằng 3 optimizer (mini-batch GD / Momentum / Adam) và so sánh trực tiếp qua cost curve + decision boundary; mục 7.3 lặp lại có kèm learning rate decay.

## 5. Kết quả mong đợi

- Cả 3 optimizer đều giảm được cost, nhưng Adam thường hội tụ nhanh và mượt nhất trên bộ dữ liệu "moons".
- Learning rate decay giúp cost dao động ít hơn ở giai đoạn cuối training so với learning rate cố định.

## 6. Câu hỏi ôn tập / mở rộng

1. Vì sao Momentum cần thêm biến trạng thái `v` (velocity) còn Adam cần cả `v` và `s`? Mỗi biến đại diện cho ước lượng gì (bậc 1 / bậc 2 của gradient)?
2. Batch size nhỏ hơn (mini-batch) đánh đổi điều gì so với dùng toàn bộ dữ liệu (batch GD) mỗi lần cập nhật?
3. Thử nối kỹ thuật **gradient flow visualization** (đã học ở bài 03, mục 8) vào notebook này: ghi lại `||dW||` theo iteration cho cả 3 optimizer — optimizer nào cho gradient flow ổn định nhất?

## 7. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all).
- Dữ liệu là `datasets/data.mat` (bộ "moons" 2D tổng hợp, đã có sẵn trong thư mục `thực hành/04.../datasets/`, đã tách train/val ngay trong file) — không có rủi ro data leakage.
- **Riêng bài này**: nếu chạy trên máy Windows/local có thể gặp 1 assertion lỗi ở cell test `random_mini_batches` do numpy trên Windows mặc định dùng số nguyên 32-bit gây tràn số ở một phép tính rất lớn trong chính cell tự-kiểm-tra của bài gốc — đây là đặc thù của Windows, **không xảy ra trên Google Colab** (chạy Linux, numpy mặc định 64-bit) nên không cần sửa gì.
- Sau khi chạy sạch không lỗi trên Colab, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
