# Bài 03 — Vanishing/Exploding Gradient & Khởi tạo trọng số (Xavier, He)

**Notebook tương ứng:** `thực hành/03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so/03_Khoi_tao_trong_so.ipynb`

## 1. Mục tiêu

Bài này thực hành phần nội dung: *Vanishing/Exploding gradient: nguyên nhân và cách xử lý*, *Weight initialization: Xavier, He*.

Sau khi hoàn thành, học viên có thể:
- Giải thích vì sao khởi tạo trọng số bằng 0 khiến mạng "chết" (fails to break symmetry).
- Giải thích vì sao khởi tạo ngẫu nhiên với giá trị quá lớn gây mất ổn định / vanishing-exploding gradient.
- Tự cài đặt He initialization và hiểu vì sao nó phù hợp với activation ReLU.
- **Trực quan hóa gradient flow**: tự vẽ và đọc biểu đồ `||dW||` theo từng layer qua các iteration để chẩn đoán vanishing/exploding gradient — đây là phần mở rộng (Mục 8) tự soạn thêm, không có trong bài gốc của Coursera.

## 2. Chuẩn bị

- Đã hoàn thành bài 01 (forward/backward propagation).
- Khái niệm phân phối xác suất (mean, variance) của NumPy: `np.random.randn`, `np.var`.

## 3. Cách mở và chạy trên Google Colab

1. Vào `Deep Learning/Module 1/thực hành/03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so/` trên Drive.
2. Chuột phải file `.ipynb` → **Open with → Google Colaboratory**.
3. Chạy cell Colab setup đầu tiên (mount Drive + `cd`) — sửa `NOTEBOOK_DIR` nếu đường dẫn trên Drive của bạn khác.
4. Run all / chạy tuần tự từng cell.

## 4. Nội dung chính

| Mục | Nội dung |
|---|---|
| 4 (Exercise 1) | `initialize_parameters_zeros` — khởi tạo toàn bộ W, b = 0 |
| 5 (Exercise 2) | `initialize_parameters_random` — khởi tạo ngẫu nhiên nhưng nhân 10 (cố tình quá lớn) |
| 6 (Exercise 3) | `initialize_parameters_he` — He initialization: `randn * sqrt(2/n_prev)` |
| 7 | So sánh 3 cách khởi tạo: train accuracy 50% / 83% / 99% |
| **8 (mở rộng)** | `model_with_grad_tracking` — huấn luyện lại cả 3 cách khởi tạo, ghi lại `\|\|dW\|\|` từng layer mỗi 10 iteration, vẽ biểu đồ gradient flow (symlog scale) + biểu đồ so sánh tổng hợp |

## 5. Kết quả mong đợi

- Zero init: train accuracy ~50% (đoán ngẫu nhiên), gradient của layer 1 và 2 **bằng đúng 0** suốt quá trình train (nhìn thấy rõ trên biểu đồ mục 8 — đường phẳng tại đáy).
- Random init (×10): train accuracy ~83%, gradient dao động mạnh, không ổn định.
- He init: train accuracy ~99%, gradient ổn định trong một dải hợp lý, không tiến về 0 cũng không nổ.

## 6. Câu hỏi ôn tập / mở rộng

1. Vì sao chỉ cần trọng số W được khởi tạo ngẫu nhiên là đủ để "phá vỡ tính đối xứng" (symmetry breaking), còn bias b thì khởi tạo 0 vẫn được?
2. He initialization dùng hệ số `sqrt(2/n)`, còn Xavier dùng `sqrt(1/n)` — thử đổi hệ số trong `initialize_parameters_he` sang Xavier rồi so sánh kết quả train và biểu đồ gradient flow.
3. Nhìn vào biểu đồ mục 8.2 (so sánh gradient theo layer): layer nào bị ảnh hưởng nặng nhất bởi khởi tạo kém? Giải thích bằng cơ chế lan truyền ngược qua nhiều lớp liên tiếp.

## 7. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all) — mục 8 huấn luyện lại 3 mô hình nên sẽ mất khoảng vài chục giây, đây là bình thường.
- Dữ liệu là bộ "circles" tổng hợp bằng `sklearn.datasets.make_circles` (sinh ngẫu nhiên có seed cố định), tách train/test rõ ràng ngay trong `load_dataset()` — không có rủi ro data leakage.
- Sau khi chạy sạch không lỗi, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
