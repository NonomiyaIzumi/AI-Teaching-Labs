# Bài 05 — Batch Normalization & Layer Normalization

**Notebook tương ứng:** `thực hành/05_Batch-Norm-Layer-Norm/05_Batch_Norm_Layer_Norm.ipynb`

> Notebook này do tự biên soạn (không có sẵn trong bộ bài tập gốc của Coursera) để lấp đầy nội dung *Batch Normalization, Layer Normalization* trong đề cương Chương 1. Toàn bộ code đã được **gradient-check bằng numerical gradient** (kỹ thuật học ở bài 02) trước khi đưa vào bài, sai số ~1e-9.

## 1. Mục tiêu

Sau khi hoàn thành, học viên có thể:
- Cài đặt Batch Normalization và Layer Normalization từ đầu (forward + backward) bằng NumPy.
- Hiểu chính xác điểm khác nhau giữa BN và LN: **trục tính thống kê** (mean/var) — BN theo trục vi dụ (batch), LN theo trục feature.
- Tự kiểm chứng backward pass bằng gradient checking (tái sử dụng kỹ thuật bài 02).
- Quan sát bằng thực nghiệm: BN/LN giúp **cứu** một mạng sâu bị khởi tạo tệ (liên hệ trực tiếp tới bài 03) khỏi vanishing/exploding gradient.

## 2. Chuẩn bị

- Đã hoàn thành bài 02 (gradient checking) và bài 03 (vanishing/exploding gradient, gradient flow visualization) — bài này dùng lại cả hai kỹ thuật đó.

## 3. Cách mở và chạy trên Google Colab

1. Vào `Deep Learning/Module 1/thực hành/05_Batch-Norm-Layer-Norm/` trên Drive.
2. Chuột phải file `.ipynb` → **Open with → Google Colaboratory**.
3. Chạy cell Colab setup đầu tiên (mount Drive + `cd`) — sửa `NOTEBOOK_DIR` nếu đường dẫn trên Drive của bạn khác.
4. Run all / chạy tuần tự từng cell. Toàn bộ notebook (bao gồm huấn luyện 3 mô hình so sánh) chạy trong khoảng 15–20 giây.

## 4. Nội dung chính

| Mục | Nội dung |
|---|---|
| 1 | Vì sao cần chuẩn hóa — liên hệ vanishing/exploding gradient (bài 03) |
| 2 | `batchnorm_forward` / `batchnorm_backward` (train/test mode, running mean/var) + gradient check |
| 3 | `layernorm_forward` / `layernorm_backward` (không cần running stats) + gradient check |
| 4 | Bảng so sánh BN vs LN (batch size, train/test, use case CNN/MLP vs RNN/Transformer) |
| 5 | Debug thực tế: mạng 5 lớp, khởi tạo cố tình kém (`W *= 4`), so sánh `none` / `batchnorm` / `layernorm` — vẽ gradient flow, cost curve, decision boundary |

## 5. Kết quả mong đợi

- Cả 2 gradient check (BN và LN) đều báo relative error `< 1e-6` (thực tế đạt ~1e-9 đến 1e-11).
- Mục 5: `none` không học được gì (cost đứng yên ở $\\ln 2 \\approx 0.693$, gradient nổ rồi chết ngay từ đầu); `batchnorm` đạt cost ~0.001; `layernorm` đạt cost ~0.04 — cả hai đều học tốt, minh chứng rõ ràng cho vai trò của normalization.

## 6. Câu hỏi ôn tập / mở rộng

1. Vì sao Layer Normalization **không cần** running mean/running var như Batch Normalization?
2. Nếu batch size = 1, BatchNorm còn hoạt động đúng như lý thuyết không? Vì sao LayerNorm thì vẫn hoạt động bình thường?
3. Trong thực nghiệm mục 5, LayerNorm hội tụ chậm hơn BatchNorm ở cùng learning rate 0.1 trong 3000 iteration đầu — thử tăng learning rate hoặc số iteration cho riêng LayerNorm, quan sát cost có cải thiện không? Điều này nói lên điều gì về việc mỗi kỹ thuật chuẩn hóa có thể cần tinh chỉnh hyperparameter khác nhau?

## 7. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all) — 2 cell gradient check phải luôn in "Tat ca deu dung" trước khi xem tiếp phần sau.
- Dữ liệu là bộ "circles" tổng hợp bằng `sklearn.datasets.make_circles` (seed cố định, không phải dữ liệu thật) — không có rủi ro data leakage.
- Notebook này không thuộc bài tập gốc Coursera nên không có phần "fill in the blank" — nếu muốn giao cho học viên tự làm, có thể ẩn phần code trong `batchnorm_backward`/`layernorm_backward` và yêu cầu tự cài lại, dùng gradient check có sẵn để tự chấm (đã ghi chú ở mục 6 cuối notebook).
- Sau khi chạy sạch không lỗi trên Colab, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
