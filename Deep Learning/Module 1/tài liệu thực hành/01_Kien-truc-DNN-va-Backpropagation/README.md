# Bài 01 — Xây dựng mô hình DNN (Kiến trúc & Backpropagation)

**Notebook tương ứng:** `thực hành/01_Kien-truc-DNN-va-Backpropagation/01_Xay_dung_mo_hinh_DNN.ipynb`

## 1. Mục tiêu

Bài này thực hành phần nội dung: *Nguyên lý và kiến trúc của DNN/MLP*, *Backpropagation và các hàm kích hoạt*.

Sau khi hoàn thành, học viên có thể:
- Tự cài đặt forward propagation và backward propagation cho một mạng DNN L lớp bất kỳ, hoàn toàn bằng NumPy (không dùng framework).
- Hiểu rõ luồng dữ liệu: `LINEAR -> ACTIVATION` lặp lại qua từng lớp, và cách gradient lan truyền ngược qua từng lớp bằng chain rule.
- Phân biệt vai trò của ReLU (hidden layers) và Sigmoid (output layer) trong bài toán phân loại nhị phân.

## 2. Chuẩn bị

- Đại số tuyến tính cơ bản (nhân ma trận, chuyển vị).
- Đạo hàm/chain rule.
- NumPy cơ bản (`np.dot`, broadcasting, `shape`).

## 3. Cách mở và chạy trên Google Colab

1. Trong Google Drive, vào đúng thư mục `Deep Learning/Module 1/thực hành/01_Kien-truc-DNN-va-Backpropagation/`.
2. Chuột phải vào file `.ipynb` → **Open with → Google Colaboratory**.
3. Chạy cell đầu tiên (Colab setup) — cell này tự mount Google Drive và `cd` vào đúng thư mục để các lệnh `from dnn_utils import ...`, `from testCases import ...` và ảnh minh họa trong `images/` hoạt động đúng.
   - **Quan trọng:** nếu đường dẫn trong cell (`NOTEBOOK_DIR`) không khớp với vị trí thực tế bạn lưu trên Drive, hãy sửa lại cho đúng trước khi chạy tiếp.
4. Chạy tuần tự từng cell từ trên xuống (Runtime → Run all, hoặc Shift+Enter từng cell).

## 4. Nội dung chính (10 Exercise)

| Mục | Exercise | Nội dung |
|---|---|---|
| 3.1 | Exercise 1 | `initialize_parameters` — khởi tạo W, b cho mạng 2 lớp |
| 3.2 | Exercise 2 | `initialize_parameters_deep` — khởi tạo cho mạng L lớp tổng quát |
| 4.1 | Exercise 3 | `linear_forward` — tính $Z = WA + b$ |
| 4.2 | Exercise 4 | `linear_activation_forward` — Linear + ReLU/Sigmoid |
| 4.3 | Exercise 5 | `L_model_forward` — ghép toàn bộ forward pass L lớp |
| 5 | Exercise 6 | `compute_cost` — cross-entropy loss |
| 6.1 | Exercise 7 | `linear_backward` — dW, db, dA_prev từ dZ |
| 6.2 | Exercise 8 | `linear_activation_backward` — backward qua ReLU/Sigmoid |
| 6.3 | Exercise 9 | `L_model_backward` — ghép toàn bộ backward pass |
| 6.4 | Exercise 10 | `update_parameters` — cập nhật W, b bằng gradient descent |

Mỗi exercise đều có test cell riêng (dùng `testCases.py`/`public_tests.py`) để tự kiểm tra kết quả trước khi qua bước tiếp theo.

## 5. Kết quả mong đợi

- Toàn bộ 10 test cell chạy không lỗi, in ra kết quả khớp expected output ghi sẵn trong notebook.
- Đây là notebook nền tảng — các bài 02–05 đều tái sử dụng đúng công thức forward/backward đã cài ở đây (chỉ thay đổi cách khởi tạo, thêm normalization, hoặc thay optimizer).

## 6. Câu hỏi ôn tập / mở rộng

1. Vì sao lớp output dùng Sigmoid thay vì ReLU trong bài toán phân loại nhị phân?
2. Nếu bỏ activation ở hidden layer (dùng linear thuần), mạng L lớp có còn mạnh hơn 1 lớp logistic regression không? Vì sao?
3. `linear_backward` cần những giá trị nào từ forward pass (cache) để tính được dW, db, dA_prev?

## 7. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại **toàn bộ notebook từ đầu** (Runtime → Restart and run all) trên chính Colab, không chỉ chạy rời rạc từng cell — để đảm bảo không có biến "ẩn" từ lần chạy trước gây lỗi khi chấm.
- Dữ liệu dùng trong bài này chỉ là `testCases.py` (mảng số cố định, không phải dữ liệu thật) nên không có rủi ro data leakage.
- Sau khi chạy sạch, không lỗi, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
