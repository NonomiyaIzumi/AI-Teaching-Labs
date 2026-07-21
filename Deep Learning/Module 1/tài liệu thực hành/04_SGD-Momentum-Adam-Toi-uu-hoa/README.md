# LAB 04 - Hướng dẫn tự thực hành

## Thuật toán cập nhật trọng số: SGD, Momentum, Adam

**Notebook tương ứng:** `thực hành/04_SGD-Momentum-Adam-Toi-uu-hoa/04_Toi_uu_hoa.ipynb`

**Chạy bản project (local, không cần Jupyter):** `source/04_SGD-Momentum-Adam-Toi-uu-hoa/` — xem hướng dẫn cài đặt/chạy chung tại `source/README.md`. Tóm tắt: `cd "source" && uv sync`, sau đó `cd 04_SGD-Momentum-Adam-Toi-uu-hoa && uv run --project .. python main.py`.

## 1. Phát biểu bài toán

Cài đặt và so sánh 3 thuật toán cập nhật trọng số (Gradient Descent thuần, Momentum, Adam) trên cùng một kiến trúc mạng và cùng dataset, cộng thêm kỹ thuật learning rate decay, để thấy rõ tốc độ/độ ổn định hội tụ khác nhau giữa các optimizer.

| Thành phần | Mô tả yêu cầu |
|---|---|
| Input | `X` shape `(2, m)` — toạ độ 2D dataset "moons" (2 nửa vòng cung đan xen). |
| Output | `Y` shape `(1, m)` — nhãn nhị phân (0/1). |
| Loại bài toán | Phân loại nhị phân, mạng 3 lớp `[2, 5, 2, 1]` cố định — biến số là **thuật toán cập nhật tham số** (`optimizer ∈ {"gd", "momentum", "adam"}`) và **learning rate decay**. |
| Mục tiêu nâng cao | Không chỉ so sánh accuracy cuối cùng, mà đọc được cost curve/decision boundary để giải thích **vì sao** một optimizer hội tụ nhanh/mượt hơn optimizer khác trên cùng dữ liệu. |
| Metric chính | Train accuracy (`predict`), cost qua các epoch. Không có test set riêng trong bài gốc — toàn bộ đánh giá thực hiện trên chính `train_X, train_Y`. |

> **Điểm khác với bài 03:** Bài 03 chỉ đổi cách khởi tạo và dùng full-batch gradient descent. Bài này đổi **thuật toán cập nhật** (GD → Momentum → Adam) và train theo **mini-batch** (chia dữ liệu thành các lô nhỏ mỗi epoch) — cần phân biệt rõ khái niệm *epoch* (một lượt qua toàn bộ dữ liệu) và *iteration* (một lần cập nhật trên một mini-batch).

## 2. Dữ liệu sử dụng

Dataset tổng hợp (synthetic), sinh trực tiếp bằng `sklearn.datasets.make_moons` — không cần tải file ngoài.

| Thuộc tính | Giá trị cần kiểm tra trong notebook |
|---|---|
| Nguồn dữ liệu | `sklearn.datasets.make_moons(n_samples=300, noise=.2)` |
| Số điểm | 300 (không tách test set riêng trong bài gốc) |
| Số chiều input | 2 (toạ độ x, y) |
| Số lớp | 2 (nhị phân) |
| Seed | `np.random.seed(3)` trong `load_dataset()` |
| Kiến trúc mạng | `layers_dims = [train_X.shape[0], 5, 2, 1] = [2, 5, 2, 1]` |

## 3. Output bắt buộc của bài thực hành

| Nhóm output | Yêu cầu cụ thể |
|---|---|
| Exercise 1 | `update_parameters_with_gd` pass test |
| Exercise 2 | `random_mini_batches` pass test — kiểm tra đúng shape từng mini-batch, đúng thứ tự shuffle |
| Exercise 3-4 | `initialize_velocity`, `update_parameters_with_momentum` pass test |
| Exercise 5-6 | `initialize_adam`, `update_parameters_with_adam` (có bias correction) pass test |
| Mục 6 | Huấn luyện 3 mô hình (mini-batch GD / Momentum / Adam) trên cùng dataset, in cost mỗi 1000 epoch, vẽ decision boundary cho từng optimizer |
| Exercise 7-8 | `update_lr` (decay mỗi iteration), `schedule_lr_decay` (decay theo mốc cố định) pass test |
| Mục 7.3 | Lặp lại 3 optimizer có kèm learning rate decay, so sánh cost curve trước/sau khi thêm decay |

## 4. Cấu hình thí nghiệm khuyến nghị

| Thành phần | Cấu hình khuyến nghị |
|---|---|
| Kiến trúc | `layers_dims = [2, 5, 2, 1]` |
| Mini-batch size | `64` |
| Learning rate — mục 6 (không decay) | `0.0007` (mặc định của `model()`) |
| Learning rate — mục 7.3, GD/Momentum + decay | `0.1` |
| Learning rate — mục 7.3, Adam + decay | `0.01` |
| Số epoch | `5000` |
| `beta` (Momentum) | `0.9` |
| `beta1, beta2, epsilon` (Adam) | `0.9`, `0.999`, `1e-8` |
| `decay_rate` (mặc định khi test `update_lr`/`schedule_lr_decay`) | `1` |
| `time_interval` (`schedule_lr_decay`) | Mặc định trong hàm `= 1000`; test case dùng `time_interval = 100` để thấy rõ hiệu ứng decay trong ít epoch |
| Seed shuffle mini-batch | `seed = 10`, tăng dần mỗi epoch (`seed += 1`) để mỗi epoch có thứ tự batch khác nhau nhưng vẫn tái lập được |

## 5. Quy trình thực hiện từng bước

1. **Import:** `numpy, matplotlib, scipy.io, math, sklearn`, các hàm từ `opt_utils_v1a` (`load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation, compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset`), `testCases`, `public_tests`.
2. **Exercise 1 - `update_parameters_with_gd`:** cập nhật `W := W - lr*dW`, `b := b - lr*db` cho từng layer — chính là bước cuối của bài 01/03, viết lại thành hàm optimizer độc lập.
3. **Exercise 2 - `random_mini_batches`:** shuffle `(X, Y)` theo cùng permutation, cắt thành các batch kích thước `mini_batch_size` (batch cuối có thể nhỏ hơn nếu `m` không chia hết).
4. **Exercise 3 - `initialize_velocity`:** khởi tạo `v["dW"+l]`, `v["db"+l]` bằng 0, cùng shape với `W`, `b`.
5. **Exercise 4 - `update_parameters_with_momentum`:** `v := beta*v + (1-beta)*dW`, `W := W - lr*v` (exponentially weighted average của gradient).
6. **Exercise 5 - `initialize_adam`:** khởi tạo cả `v` (bậc 1, giống Momentum) và `s` (bậc 2, bình phương gradient) bằng 0.
7. **Exercise 6 - `update_parameters_with_adam`:** cập nhật `v`, `s`, **bias-correction** (`v_corrected = v/(1-beta1^t)`, tương tự cho `s`), rồi `W := W - lr * v_corrected/(sqrt(s_corrected)+eps)`.
8. **Mục 6 - Huấn luyện & so sánh:** chạy `model(train_X, train_Y, layers_dims, optimizer="gd"/"momentum"/"adam")`, quan sát cost curve và decision boundary từng optimizer.
9. **Exercise 7 - `update_lr`:** `lr := lr / (1 + decay_rate*epoch_num)` — giảm dần mỗi epoch.
10. **Exercise 8 - `schedule_lr_decay`:** chỉ giảm lr sau mỗi `time_interval` epoch (dùng `np.floor(epoch_num/time_interval)`), thay vì giảm liên tục mỗi epoch.
11. **Mục 7.3 - So sánh có/không decay:** chạy lại cả 3 optimizer với `decay=update_lr` hoặc `decay=schedule_lr_decay`, so sánh cost curve.

## 6. Kết quả mong đợi

- Cả 3 optimizer đều giảm được cost trên dataset "moons"; Adam thường hội tụ nhanh và mượt nhất, Momentum nhanh hơn GD thuần nhưng có thể dao động nhẹ hơn Adam.
- Thêm learning rate decay giúp cost dao động ít hơn ở giai đoạn cuối training so với learning rate cố định (đặc biệt rõ với GD/Momentum ở `lr=0.1`).

## 7. Bài tập mở rộng

1. **Nối kỹ thuật gradient flow (bài 03, mục 8) vào bài này:** ghi lại `\|\|dW\|\|` theo epoch cho cả 3 optimizer — optimizer nào cho gradient flow ổn định nhất?
2. **Thử `mini_batch_size` khác:** so sánh `mini_batch_size = 8`, `64`, `256` (gần bằng cả dataset) với cùng optimizer Adam — batch nhỏ hơn có làm cost dao động nhiều hơn giữa các epoch không?
3. **Tự implement RMSprop:** chỉ dùng phần `s` (không dùng `v`) của Adam — `s := beta2*s + (1-beta2)*dW^2`, `W := W - lr*dW/(sqrt(s)+eps)` — so sánh với Adam đầy đủ trên cùng dataset.
4. **Đo Best Val Accuracy đúng nghĩa:** tự tách 300 điểm moons thành 240 train / 60 validation (dataset gốc không có sẵn phần này), huấn luyện lại và báo cáo accuracy trên phần validation, không chỉ trên train.

## 8. Tài liệu bổ sung

| Chủ đề | Liên kết tham khảo |
|---|---|
| Kingma & Ba, 2014 — Adam: A Method for Stochastic Optimization | https://arxiv.org/abs/1412.6980 |
| Exponentially weighted averages (trực giác Momentum/Adam) | https://www.coursera.org/learn/deep-neural-network |
| Learning rate schedules (tổng quan) | https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate |

## 9. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all).
- Dữ liệu là `sklearn.datasets.make_moons` sinh trực tiếp trong notebook (seed cố định) — không có rủi ro data leakage.
- **Riêng bài này:** nếu chạy trên máy Windows/local có thể gặp 1 `AssertionError` ở cell test `random_mini_batches` do numpy trên Windows mặc định dùng số nguyên 32-bit gây tràn số ở một phép tính rất lớn trong chính cell tự-kiểm-tra của bài gốc — đặc thù của Windows, **không xảy ra trên Google Colab** (chạy Linux, numpy mặc định 64-bit) nên không cần sửa gì.
- Sau khi chạy sạch không lỗi trên Colab, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
