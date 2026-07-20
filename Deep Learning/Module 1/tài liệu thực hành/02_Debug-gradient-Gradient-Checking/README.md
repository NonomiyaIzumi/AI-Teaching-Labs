# LAB 02 - Hướng dẫn tự thực hành

## Debug gradient bằng Gradient Checking

**Notebook tương ứng:** `thực hành/02_Debug-gradient-Gradient-Checking/02_Kiem_tra_gradient.ipynb`

## 1. Phát biểu bài toán

Xây dựng công cụ **debug backward propagation**: so sánh gradient tính bằng đạo hàm giải tích (backprop tự cài) với gradient ước lượng bằng sai phân hữu hạn (numerical gradient), để phát hiện lỗi trong code backward *trước khi* dùng nó huấn luyện một mạng thật.

| Thành phần | Mô tả yêu cầu |
|---|---|
| Input | Phần 1D: một scalar `theta`. Phần N-D: một `parameters` dictionary (W1,b1,W2,b2,W3,b3) của mạng 3 lớp, được làm phẳng thành 1 vector bằng `dictionary_to_vector`. |
| Output | `difference` — sai số tương đối (relative error) giữa gradient giải tích và gradient numerical, dạng `np.float64`. |
| Loại bài toán | Kỹ thuật kiểm thử/debug (không phải bài toán học máy có nhãn) — áp dụng được cho bất kỳ hàm khả vi nào. |
| Mục tiêu nâng cao | Không chỉ chạy `gradient_check` cho ra số — phải **đọc được** con số đó để biết mạng có bug hay không, và tìm ra đúng dòng code sai khi `difference` bất thường lớn. |
| Metric chính | `relative error = \|\|grad - gradapprox\|\| / (\|\|grad\|\| + \|\|gradapprox\|\|)`. Ngưỡng đọc: `< 1e-7` rất tốt, `1e-7`–`1e-5` cần xem xét, `> 1e-3` gần như chắc chắn có bug. |

> **Điểm khác với bài 01:** Ở bài 01, ta *tin* rằng backward tự cài là đúng nếu khớp expected output cố định. Ở bài này, ta có công cụ **tự kiểm chứng** cho bất kỳ kiến trúc/tham số nào, kể cả khi không có expected output sẵn — đây là kỹ năng dùng lại được cho mọi bài sau (03, 04, 05 đều được gradient-check theo đúng kỹ thuật này).

## 2. Dữ liệu sử dụng

Không dùng dataset thật. Input là các test case cố định:

| Thuộc tính | Giá trị cần kiểm tra trong notebook |
|---|---|
| Test case 1D | `x = 2, theta = 4` → hàm `J(theta) = theta * x` |
| Test case N-D | `gradient_check_n_test_case()` — mạng 3 lớp cố định, sinh bởi seed cố định trong `testCases.py` |
| `epsilon` (bước sai phân hữu hạn) | `1e-7` |
| Expected `difference` (N-D, **trước khi sửa bug**) | xấp xỉ `0.2850931567761623` (rất lớn — báo hiệu có bug) |

## 3. Output bắt buộc của bài thực hành

| Nhóm output | Yêu cầu cụ thể |
|---|---|
| Phần 1D | In `J = ...`, `dtheta = ...`, và relative error của `gradient_check` (phải rất nhỏ, ~1e-8 đến 1e-10) |
| Phần N-D — trước khi sửa bug | Chạy `gradient_check_n` trên `backward_propagation_n` gốc (có bug), quan sát `difference` lớn bất thường (~0.285) |
| Phần N-D — sau khi sửa bug | Tìm và sửa 2 dòng sai trong `backward_propagation_n` (gợi ý: `dW2` và `db1`), chạy lại `gradient_check_n`, `difference` phải giảm về mức rất nhỏ |
| Toàn bộ 4 test cell | Mỗi test cell (`..._test(...)`) pass, không có `AssertionError` |

## 4. Cấu hình thí nghiệm khuyến nghị

| Thành phần | Cấu hình khuyến nghị |
|---|---|
| `epsilon` | `1e-7` (đủ nhỏ để xấp xỉ đạo hàm chính xác, đủ lớn để tránh sai số làm tròn số học) |
| Ngưỡng kết luận "đúng" | `difference < 1e-7` |
| Ngưỡng kết luận "có bug" | `difference > 1e-3` |
| `print_msg` | `True` khi debug (để in trực tiếp kết luận PASS/FAIL) |

## 5. Quy trình thực hiện từng bước

1. **Import:** `numpy`, `testCases`, `public_tests`, `gc_utils` (`sigmoid`, `relu`, `dictionary_to_vector`, `vector_to_dictionary`, `gradients_to_vector`).
2. **Đọc "How does Gradient Checking work?"** — nắm định nghĩa đạo hàm một phía: `J'(theta) ≈ (J(theta+eps) − J(theta−eps)) / (2*eps)`.
3. **Exercise 1 - `forward_propagation`:** hàm 1 biến đơn giản `J(theta) = theta * x`.
4. **Exercise 2 - `backward_propagation`:** đạo hàm giải tích `dJ/dtheta = x`.
5. **Exercise 3 - `gradient_check`:** tính `J_plus = J(theta+eps)`, `J_minus = J(theta-eps)`, `gradapprox`, so sánh với `backward_propagation(x, theta)` bằng relative error.
6. **Đọc `backward_propagation_n`** (mạng 3 lớp, có sẵn — cố ý cài 2 lỗi ở `dW2` và `db1`, xem mục 3 "Nội dung chính" ở trên).
7. **Exercise 4 - `gradient_check_n`:** với mỗi phần tử `i` trong vector tham số đã làm phẳng, tính `J_plus[i]`, `J_minus[i]`, `gradapprox[i]`; ghép `grad` từ `gradients_to_vector(gradients)`; tính relative error tổng thể bằng `np.linalg.norm`.
8. **Chạy gradient check trên `backward_propagation_n` gốc:** quan sát `difference` lớn (~0.285) → xác nhận có bug.
9. **Tìm và sửa bug:** đọc lại `backward_propagation_n`, so khớp với công thức đạo hàm chuẩn (giống bài 01) để tìm 2 dòng sai.
10. **Chạy lại gradient check:** xác nhận `difference` đã nhỏ trở lại.

## 6. Kết quả mong đợi

- Phần 1D: relative error ~1e-8 đến 1e-10.
- Phần N-D trước khi sửa: `difference ≈ 0.285` (fail rõ ràng).
- Phần N-D sau khi sửa: `difference` giảm mạnh, đạt ngưỡng "đúng" (`< 1e-7`).

## 7. Bài tập mở rộng

1. **Áp dụng lại cho bài 01:** dùng `numerical_gradient`-style ở bài này để tự viết một gradient check cho `linear_backward` đã cài ở bài 01 (mạng 2 lớp) — kết quả có khớp không?
2. **Thử đổi `epsilon`:** chạy `gradient_check` với `epsilon = 1e-3` và `epsilon = 1e-10`, so sánh relative error — quan sát đánh đổi giữa sai số xấp xỉ (epsilon lớn) và sai số làm tròn số học (epsilon quá nhỏ).
3. **Đo thời gian chạy:** dùng `%timeit` đo thời gian `gradient_check_n` trên mạng 3 lớp — giải thích bằng số tại sao gradient checking "chậm" và không dùng được trong vòng lặp train mỗi iteration.

## 8. Tài liệu bổ sung

| Chủ đề | Liên kết tham khảo |
|---|---|
| Numerical differentiation | https://en.wikipedia.org/wiki/Numerical_differentiation |
| Gradient checking (CS231n notes) | https://cs231n.github.io/neural-networks-3/#gradcheck |

## 9. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all).
- Dữ liệu chỉ là giá trị số cố định từ `testCases.py`, không có dữ liệu thật → không có rủi ro data leakage.
- Sau khi chạy sạch không lỗi, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
