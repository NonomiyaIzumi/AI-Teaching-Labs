# LAB 04 - Hướng dẫn tự thực hành

## Thuật toán cập nhật trọng số: SGD, Momentum, Adam

**Notebook tương ứng:** `thực hành/04_SGD-Momentum-Adam-Toi-uu-hoa/04_Toi_uu_hoa.ipynb`

**Chạy bản project (local, không cần Jupyter):** `source/04_SGD-Momentum-Adam-Toi-uu-hoa/` — xem hướng dẫn cài đặt/chạy chung tại `source/README.md`. Tóm tắt: `cd "source" && uv sync`, sau đó `cd 04_SGD-Momentum-Adam-Toi-uu-hoa && uv run --project .. python main.py`.

**Bản TensorFlow/Keras:** `keras-tensorflow/04_SGD-Momentum-Adam-Toi-uu-hoa/04_Toi_uu_hoa_Keras.ipynb` — bài học độc lập về optimizer (`keras.optimizers.SGD`/`Adam`, `model.fit(..., batch_size=...)`). Cũng có notebook `keras-tensorflow/Pima_Diabetes_Nen_Tang_Deep_Learning_Keras.ipynb` — trình bày nối tiếp 5 chủ đề nền tảng Deep Learning trong 1 file. Xem `keras-tensorflow/README.md` để biết cách cài đặt/chạy local.

**Cách làm bài trong notebook:** Mỗi hàm `# GRADED FUNCTION` đã được để trống (`pass`) ở đoạn `# YOUR CODE STARTS HERE ... # YOUR CODE ENDS HERE` — bạn tự viết code vào đó. Ngay bên dưới mỗi bài là 1 cell **💡 Đáp án** dạng form thu gọn của Colab (bấm mũi tên bên trái hoặc chọn "Show code" ở menu ba chấm để mở ra xem). Vì cell đáp án vẫn tự chạy khi bạn bấm "Runtime → Run all", notebook sẽ **không bao giờ báo lỗi** dù bạn chưa làm bài nào — nhưng lưu ý: chạy Run all như vậy nghĩa là bạn đang xem code của đáp án, không phải code bạn viết. Để tự kiểm tra bài của mình, sau khi viết code vào cell bài tập, hãy **chạy lại đúng cell đó** (không chạy cell đáp án ngay dưới) rồi mới chạy tiếp các cell demo/test phía sau — vì Python luôn dùng định nghĩa hàm được chạy **gần nhất**.

## 1. Phát biểu bài toán

Cài đặt và so sánh 3 thuật toán cập nhật trọng số (Gradient Descent thuần, Momentum, Adam) trên cùng một kiến trúc mạng và cùng dataset **y tế thật**, cộng thêm kỹ thuật learning rate decay, để thấy rõ tốc độ/độ ổn định hội tụ khác nhau giữa các optimizer.

| Thành phần | Mô tả yêu cầu |
|---|---|
| Input | `X` shape `(8, m)` — 8 đặc trưng lâm sàng đã chuẩn hóa từ bộ dữ liệu **Pima Indians Diabetes**. |
| Output | `Y` shape `(1, m)` — nhãn nhị phân (0 = không tiểu đường, 1 = tiểu đường). |
| Loại bài toán | Phân loại nhị phân trên dữ liệu thật, mạng 3 lớp `[8, 5, 2, 1]` cố định — biến số là **thuật toán cập nhật tham số** (`optimizer ∈ {"gd", "momentum", "adam"}`) và **learning rate decay**. |
| Mục tiêu nâng cao | Không chỉ so sánh accuracy cuối cùng, mà đọc được cost curve/confusion matrix/ROC để giải thích **vì sao** một optimizer hội tụ nhanh/mượt hơn optimizer khác trên cùng dữ liệu. |
| Metric chính | Train/test accuracy (`predict`), **Precision/Recall/F1/AUC** trên test set (`evaluate_classification`, xem mục 3), cost qua các epoch. |

> **Điểm khác với bài 03:** Bài 03 chỉ đổi cách khởi tạo và dùng full-batch gradient descent. Bài này đổi **thuật toán cập nhật** (GD → Momentum → Adam) và train theo **mini-batch** (chia dữ liệu thành các lô nhỏ mỗi epoch) — cần phân biệt rõ khái niệm *epoch* (một lượt qua toàn bộ dữ liệu) và *iteration* (một lần cập nhật trên một mini-batch).

## 2. Dữ liệu sử dụng

Dataset thật **Pima Indians Diabetes** — giống hệt bài 03 (dùng chung 1 file `data/pima-indians-diabetes.csv`, cùng cách tiền xử lý), để việc so sánh optimizer diễn ra trên đúng một bài toán xuyên suốt module.

| Thuộc tính | Giá trị cần kiểm tra trong notebook |
|---|---|
| Nguồn dữ liệu | `data/pima-indians-diabetes.csv` (đọc qua `load_dataset()` trong `opt_utils_v1a.py`) |
| Số mẫu train / test | 614 / 154 (chia 80/20, giữ nguyên tỉ lệ lớp bằng `stratify`) |
| Số chiều input | 8 |
| Số lớp | 2 (nhị phân) |
| Tiền xử lý | Giống bài 03: median-impute 5 cột có giá trị `0` ẩn ý nghĩa "thiếu dữ liệu" (fit trên train), rồi chuẩn hóa (fit trên train) |
| Seed | `random_state=1` khi chia train/test (cố định trong `load_dataset()`) |
| Kiến trúc mạng | `layers_dims = [train_X.shape[0], 5, 2, 1] = [8, 5, 2, 1]` |

## 3. Output bắt buộc của bài thực hành

| Nhóm output | Yêu cầu cụ thể |
|---|---|
| Exercise 1 | `update_parameters_with_gd` pass test |
| Exercise 2 | `random_mini_batches` pass test — kiểm tra đúng shape từng mini-batch, đúng thứ tự shuffle |
| Exercise 3-4 | `initialize_velocity`, `update_parameters_with_momentum` pass test |
| Exercise 5-6 | `initialize_adam`, `update_parameters_with_adam` (có bias correction) pass test |
| Mục 6 | Huấn luyện 3 mô hình (mini-batch GD / Momentum / Adam) trên cùng dataset, in cost mỗi 1000 epoch, đánh giá trên **test set** bằng `evaluate_classification` (confusion matrix + ROC, thay cho decision boundary 2D không còn vẽ được với 8 chiều) |
| Exercise 7-8 | `update_lr` (decay mỗi iteration), `schedule_lr_decay` (decay theo mốc cố định) pass test |
| Mục 7.3 | Lặp lại 3 optimizer có kèm learning rate decay, so sánh cost curve và test accuracy trước/sau khi thêm decay |

## 4. Cấu hình thí nghiệm khuyến nghị

| Thành phần | Cấu hình khuyến nghị |
|---|---|
| Kiến trúc | `layers_dims = [8, 5, 2, 1]` |
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

1. **Import:** `numpy, matplotlib, scipy.io, math, sklearn`, các hàm từ `opt_utils_v1a` (`load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation, compute_cost, predict, load_dataset, evaluate_classification`), `testCases`, `public_tests`.
2. **Exercise 1 - `update_parameters_with_gd`:** cập nhật `W := W - lr*dW`, `b := b - lr*db` cho từng layer — chính là bước cuối của bài 01/03, viết lại thành hàm optimizer độc lập.
3. **Exercise 2 - `random_mini_batches`:** shuffle `(X, Y)` theo cùng permutation, cắt thành các batch kích thước `mini_batch_size` (batch cuối có thể nhỏ hơn nếu `m` không chia hết).
4. **Exercise 3 - `initialize_velocity`:** khởi tạo `v["dW"+l]`, `v["db"+l]` bằng 0, cùng shape với `W`, `b`.
5. **Exercise 4 - `update_parameters_with_momentum`:** `v := beta*v + (1-beta)*dW`, `W := W - lr*v` (exponentially weighted average của gradient).
6. **Exercise 5 - `initialize_adam`:** khởi tạo cả `v` (bậc 1, giống Momentum) và `s` (bậc 2, bình phương gradient) bằng 0.
7. **Exercise 6 - `update_parameters_with_adam`:** cập nhật `v`, `s`, **bias-correction** (`v_corrected = v/(1-beta1^t)`, tương tự cho `s`), rồi `W := W - lr * v_corrected/(sqrt(s_corrected)+eps)`.
8. **Mục 6 - Huấn luyện & so sánh:** chạy `model(train_X, train_Y, layers_dims, optimizer="gd"/"momentum"/"adam")`, quan sát cost curve rồi đánh giá trên `test_X, test_Y` bằng `evaluate_classification`.
9. **Exercise 7 - `update_lr`:** `lr := lr / (1 + decay_rate*epoch_num)` — giảm dần mỗi epoch.
10. **Exercise 8 - `schedule_lr_decay`:** chỉ giảm lr sau mỗi `time_interval` epoch (dùng `np.floor(epoch_num/time_interval)`), thay vì giảm liên tục mỗi epoch.
11. **Mục 7.3 - So sánh có/không decay:** chạy lại cả 3 optimizer với `decay=update_lr` hoặc `decay=schedule_lr_decay`, so sánh cost curve và test accuracy.

## 6. Kết quả mong đợi

- Không decay: GD ~72% và Momentum ~72% test accuracy (gần như nhau — đúng như ghi chú gốc "gains từ momentum là nhỏ với ví dụ đơn giản"), Adam vượt trội rõ rệt ~79%.
- Có `schedule_lr_decay`: cả 3 optimizer đều cải thiện so với không decay (GD/Momentum lên ~75%), cho thấy decay giúp ích nhiều nhất khi learning rate ban đầu chưa tối ưu.
- `update_lr` (decay liên tục mỗi epoch) làm learning rate giảm quá nhanh (xuống ~0.0001 chỉ sau 1000 epoch) → GD hội tụ chậm hơn hẳn, minh chứng vì sao `schedule_lr_decay` (giảm theo mốc) thường thực tế hơn `update_lr` (giảm liên tục) cho bài toán ít epoch.

## 7. Bài tập mở rộng

1. **Nối kỹ thuật gradient flow (bài 03, mục 8) vào bài này:** ghi lại `\|\|dW\|\|` theo epoch cho cả 3 optimizer — optimizer nào cho gradient flow ổn định nhất?
2. **Thử `mini_batch_size` khác:** so sánh `mini_batch_size = 8`, `64`, `256` (gần bằng cả dataset) với cùng optimizer Adam — batch nhỏ hơn có làm cost dao động nhiều hơn giữa các epoch không?
3. **Tự implement RMSprop:** chỉ dùng phần `s` (không dùng `v`) của Adam — `s := beta2*s + (1-beta2)*dW^2`, `W := W - lr*dW/(sqrt(s)+eps)` — so sánh với Adam đầy đủ trên cùng dataset.
4. **So sánh Recall giữa các optimizer:** trong bài toán y tế, Recall (tỉ lệ phát hiện đúng ca mắc bệnh) thường quan trọng hơn accuracy thuần. Optimizer/cấu hình decay nào cho Recall tốt nhất trên test set?

## 8. Tài liệu bổ sung

| Chủ đề | Liên kết tham khảo |
|---|---|
| Kingma & Ba, 2014 — Adam: A Method for Stochastic Optimization | https://arxiv.org/abs/1412.6980 |
| Exponentially weighted averages (trực giác Momentum/Adam) | https://www.coursera.org/learn/deep-neural-network |
| Learning rate schedules (tổng quan) | https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate |
| Pima Indians Diabetes Dataset (nguồn gốc) | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |

## 9. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all).
- Dữ liệu là **Pima Indians Diabetes thật** — tập train/test đã tách rõ ràng trong `load_dataset()` (kèm `stratify`), tiền xử lý (median-impute, chuẩn hóa) chỉ fit trên train → không có rủi ro data leakage.
- **Riêng bài này:** nếu chạy trên máy Windows/local có thể gặp 1 `AssertionError` ở cell test `random_mini_batches` do numpy trên Windows mặc định dùng số nguyên 32-bit gây tràn số ở một phép tính rất lớn trong chính cell tự-kiểm-tra của bài gốc — đặc thù của Windows, **không xảy ra trên Google Colab** (chạy Linux, numpy mặc định 64-bit) nên không cần sửa gì.
- Sau khi chạy sạch không lỗi trên Colab, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
