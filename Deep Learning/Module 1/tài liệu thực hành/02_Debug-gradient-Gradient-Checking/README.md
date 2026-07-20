# Bài 02 — Debug gradient bằng Gradient Checking

**Notebook tương ứng:** `thực hành/02_Debug-gradient-Gradient-Checking/02_Kiem_tra_gradient.ipynb`

## 1. Mục tiêu

Bài này thực hành kỹ năng **debug gradient** — phần nội dung liên quan trực tiếp tới *Backpropagation* trong đề cương: làm sao biết backward propagation mình tự cài (ở bài 01) có đúng hay không, *trước khi* dùng nó để train một mạng thật.

Sau khi hoàn thành, học viên có thể:
- Hiểu ý tưởng gradient checking: so sánh gradient tính bằng backprop (đạo hàm giải tích) với gradient ước lượng bằng sai phân hữu hạn (numerical gradient, dùng định nghĩa đạo hàm).
- Tự cài đặt gradient checking 1 chiều và N chiều (`dictionary_to_vector`/`vector_to_dictionary` để làm phẳng toàn bộ parameters thành 1 vector).
- Đọc được **relative error** để kết luận backward propagation đúng hay sai (`< 1e-7`: đúng, `> 1e-3`: có bug).

## 2. Chuẩn bị

- Đã hoàn thành bài 01 (hiểu forward/backward propagation).
- Định nghĩa đạo hàm: $f'(\\theta) \\approx \\dfrac{f(\\theta+\\varepsilon) - f(\\theta-\\varepsilon)}{2\\varepsilon}$.

## 3. Cách mở và chạy trên Google Colab

1. Vào `Deep Learning/Module 1/thực hành/02_Debug-gradient-Gradient-Checking/` trên Drive.
2. Chuột phải file `.ipynb` → **Open with → Google Colaboratory**.
3. Chạy cell Colab setup đầu tiên (mount Drive + `cd` đúng thư mục) — sửa `NOTEBOOK_DIR` nếu cần.
4. Run all / chạy tuần tự từng cell.

## 4. Nội dung chính (4 Exercise)

| Mục | Exercise | Nội dung |
|---|---|---|
| 4 | Exercise 1 | `forward_propagation` — hàm 1 biến đơn giản $J(\\theta) = \\theta x$ |
| 4 | Exercise 2 | `backward_propagation` — đạo hàm giải tích $dJ/d\\theta$ |
| 4 | Exercise 3 | `gradient_check` — so sánh giải tích vs numerical (1 chiều) |
| 5 | Exercise 4 | `gradient_check_n` — tổng quát hoá cho toàn bộ tham số của một mạng nhiều lớp (N chiều) |

## 5. Kết quả mong đợi

- Phần 1 chiều: relative error rất nhỏ (~1e-8 đến 1e-10) → xác nhận công thức đạo hàm tay đúng.
- Phần N chiều: notebook cố tình cài **2 lỗi** trong `backward_propagation_n` (ở `dW2` và `db1`) để học viên thấy gradient checking phát hiện được qua relative error lớn bất thường (~0.285, trong khi bình thường phải ~1e-7). Nhiệm vụ là tự tìm và sửa 2 dòng sai đó, chạy lại `gradient_check_n` cho đến khi relative error nhỏ trở lại — đây là bài tập tìm lỗi thực tế, không phải chỉ chạy code cho có.

## 6. Câu hỏi ôn tập / mở rộng

1. Vì sao gradient checking chỉ nên dùng để **debug**, không dùng trong vòng lặp training thật (gợi ý: tốc độ)?
2. Nếu chọn `epsilon` quá lớn hoặc quá nhỏ, kết quả gradient checking sẽ bị ảnh hưởng thế nào?
3. Thử áp dụng kỹ thuật `numerical_gradient` ở bài này để tự kiểm tra lại `linear_backward` đã cài ở bài 01 — kết quả có khớp không?

## 7. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all).
- Dữ liệu chỉ là giá trị số cố định từ `testCases.py`, không có dữ liệu thật nên không có rủi ro data leakage.
- Sau khi chạy sạch không lỗi, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
