# LAB 03 - Hướng dẫn tự thực hành

## Vanishing/Exploding Gradient & Khởi tạo trọng số (Xavier, He)

**Notebook tương ứng:** `thực hành/03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so/03_Khoi_tao_trong_so.ipynb`

**Chạy bản project (local, không cần Jupyter):** `source/03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so/` — xem hướng dẫn cài đặt/chạy chung tại `source/README.md`. Tóm tắt: `cd "source" && uv sync`, sau đó `cd 03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so && uv run --project .. python main.py`.

## 1. Phát biểu bài toán

Huấn luyện cùng một mạng 3 lớp trên cùng một dataset, chỉ thay đổi **cách khởi tạo trọng số**, để quan sát trực tiếp hậu quả của vanishing/exploding gradient và cách He initialization khắc phục vấn đề này.

| Thành phần | Mô tả yêu cầu |
|---|---|
| Input | `X` shape `(2, m)` — toạ độ 2D của các điểm trong dataset "circles" (2 vòng tròn lồng nhau). |
| Output | `Y` shape `(1, m)` — nhãn nhị phân (0 = đỏ, 1 = xanh). |
| Loại bài toán | Phân loại nhị phân, mạng 3 lớp `[2, 10, 5, 1]` cố định — biến số duy nhất là **cách khởi tạo trọng số**. |
| Mục tiêu nâng cao | Không chỉ so sánh train accuracy cuối cùng, mà phải **trực quan hóa gradient flow** (`\|\|dW\|\|` theo layer, theo iteration) để giải thích *tại sao* một cách khởi tạo thất bại — đây là kỹ năng debug tổng quát, dùng lại được ở bài 04 và 05. |
| Metric chính | Train/test accuracy (qua `predict`), cost qua các iteration, và **gradient norm** `\|\|dW^[l]\|\|` theo layer (phần mở rộng mục 8). |

> **Điểm khác với bài 01/02:** Ở đây có **train thật** trên dataset thật (15.000 iteration full-batch gradient descent), nên lần đầu tiên thấy được sự khác biệt giữa "code đúng" (backward propagation đã kiểm chứng ở bài 01-02) và "mô hình học được" (phụ thuộc thêm vào khởi tạo, tối ưu hoá...).

## 2. Dữ liệu sử dụng

Dataset tổng hợp (synthetic), sinh trực tiếp trong notebook bằng `sklearn.datasets.make_circles` — không cần tải file ngoài.

| Thuộc tính | Giá trị cần kiểm tra trong notebook |
|---|---|
| Nguồn dữ liệu | `sklearn.datasets.make_circles(n_samples=300, noise=.05)` (train), tương tự cho test |
| Số điểm train | 300 |
| Số điểm test | 100 |
| Số chiều input | 2 (toạ độ x, y) |
| Số lớp | 2 (nhị phân: trong vòng tròn nhỏ / vòng tròn lớn) |
| Seed | `np.random.seed(1)` cho train, `np.random.seed(2)` cho test (cố định trong `load_dataset()`) |
| Kiến trúc mạng | `layers_dims = [2, 10, 5, 1]` (2 hidden layer + 1 output), `LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID` |

## 3. Output bắt buộc của bài thực hành

| Nhóm output | Yêu cầu cụ thể |
|---|---|
| Exercise 1-3 | `initialize_parameters_zeros`, `initialize_parameters_random`, `initialize_parameters_he` — mỗi hàm pass test cell tương ứng |
| Huấn luyện 3 mô hình | Chạy `model(train_X, train_Y, initialization=...)` cho cả 3 cách khởi tạo, in cost mỗi 1000 iteration, vẽ decision boundary |
| So sánh accuracy | Bảng train accuracy: zeros ~50%, random ~83%, he ~99% |
| **Mục 8 (mở rộng) - Gradient flow** | `model_with_grad_tracking` chạy lại cả 3 cách khởi tạo, ghi `\|\|dW^[l]\|\|` mỗi 10 iteration, vẽ 2 biểu đồ: (a) gradient theo iteration từng layer, thang `symlog`; (b) so sánh gradient trung bình cuối training theo layer × cách khởi tạo (bar chart) |

## 4. Cấu hình thí nghiệm khuyến nghị

| Thành phần | Cấu hình khuyến nghị |
|---|---|
| Kiến trúc | `layers_dims = [2, 10, 5, 1]` |
| Learning rate (mục 4-7, bài gốc) | `0.01` |
| Số iteration (mục 4-7, bài gốc) | `15000` (full-batch gradient descent, không chia mini-batch) |
| Learning rate (mục 8, gradient flow) | `0.01` |
| Số iteration (mục 8, gradient flow) | `5000`, ghi lại `\|\|dW\|\|` mỗi `track_every=10` iteration (để chạy nhanh hơn khi minh hoạ) |
| Seed khởi tạo | `np.random.seed(3)` trong `initialize_parameters_random`/`_he` |
| Hệ số random init "xấu" | `randn(...) * 10` |
| Hệ số He init | `randn(...) * sqrt(2 / n^[l-1])` |

## 5. Quy trình thực hiện từng bước

1. **Import & load dataset:** `sigmoid, relu, compute_loss, forward_propagation, backward_propagation, update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec` từ `init_utils`.
2. **Đọc hàm `model()` có sẵn:** nắm vòng lặp gradient descent chuẩn (forward → cost → backward → update), tham số `initialization` quyết định gọi hàm khởi tạo nào.
3. **Exercise 1 - `initialize_parameters_zeros`:** toàn bộ `W`, `b` = 0.
4. **Chạy `model(..., initialization="zeros")`:** quan sát cost không giảm, accuracy ~50%, decision boundary là một màu đồng nhất.
5. **Đọc giải thích symmetry breaking** (mục 4, sau kết quả zeros) — hiểu vì sao `W=0` khiến mọi neuron học giống hệt nhau.
6. **Exercise 2 - `initialize_parameters_random`:** `W = randn(...) * 10`, `b = 0`.
7. **Chạy `model(..., initialization="random")`:** quan sát cost khởi đầu rất cao (có thể `inf` do log(0)), accuracy ~83%.
8. **Exercise 3 - `initialize_parameters_he`:** `W = randn(...) * sqrt(2/n_prev)`, `b = 0`.
9. **Chạy `model(..., initialization="he")`:** quan sát cost giảm nhanh và ổn định, accuracy ~99%.
10. **So sánh bảng tổng kết** (mục 7 - Conclusions) — đối chiếu 3 kết quả.
11. **Mục 8.1 - `model_with_grad_tracking`:** viết lại `model()` để lưu thêm `\|\|dW^[l]\|\|` mỗi `track_every` iteration cho cả 3 cách khởi tạo.
12. **Mục 8.1 - Vẽ gradient flow:** 3 subplot (zeros/random/he), trục y `symlog` (vì gradient của "zeros" bằng đúng 0, log thường không vẽ được).
13. **Mục 8.2 - Bar chart tổng hợp:** trung bình `\|\|dW\|\|` ở 20 lần đo cuối, theo layer × cách khởi tạo.

## 6. Kết quả mong đợi

- **zeros**: train accuracy ~50%; `\|\|dW1\|\|`, `\|\|dW2\|\|` = 0 tuyệt đối suốt quá trình train (đường phẳng tại đáy trên biểu đồ symlog).
- **random (×10)**: train accuracy ~83%; gradient dao động mạnh, không ổn định, cost khởi đầu có thể là `inf`/`nan` do log(0).
- **he**: train accuracy ~99%; gradient ổn định trong một dải hợp lý suốt quá trình train, không tiến về 0 cũng không nổ.

## 7. Bài tập mở rộng

1. **Thử Xavier initialization:** đổi hệ số trong `initialize_parameters_he` từ `sqrt(2/n_prev)` sang `sqrt(1/n_prev)` (Xavier gốc), so sánh train accuracy và biểu đồ gradient flow với bản He.
2. **Đổi mức độ "xấu" của random init:** thử hệ số nhân `randn(...) * 3` và `* 30` thay vì `* 10`, quan sát ngưỡng nào bắt đầu gây mất ổn định rõ rệt trên biểu đồ gradient flow.
3. **Layer nào chịu ảnh hưởng nặng nhất?** Từ biểu đồ mục 8.2, so sánh `\|\|dW1\|\|` (gần input) với `\|\|dW3\|\|` (gần output) ở cách khởi tạo `zeros` — giải thích bằng cơ chế lan truyền ngược qua nhiều lớp liên tiếp (chain rule nhân dồn).
4. **Tăng độ sâu mạng:** thử `layers_dims = [2, 10, 10, 10, 5, 1]` (5 lớp thay vì 3) với cả 3 cách khởi tạo — vanishing/exploding có rõ hơn không khi mạng sâu hơn?

## 8. Tài liệu bổ sung

| Chủ đề | Liên kết tham khảo |
|---|---|
| He et al., 2015 — Delving Deep into Rectifiers | https://arxiv.org/abs/1502.01852 |
| Glorot & Bengio, 2010 — Xavier initialization | https://proceedings.mlr.press/v9/glorot10a.html |
| Symmetry breaking vs. zero initialization (deeplearning.ai forum) | https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061 |

## 9. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all) — mục 8 huấn luyện lại 3 mô hình nên sẽ mất thêm vài chục giây, đây là bình thường.
- Dữ liệu là bộ "circles" tổng hợp (`sklearn.datasets.make_circles`, seed cố định), tách train/test rõ ràng ngay trong `load_dataset()` → không có rủi ro data leakage.
- Sau khi chạy sạch không lỗi, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
