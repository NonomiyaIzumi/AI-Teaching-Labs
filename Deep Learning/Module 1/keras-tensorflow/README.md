# Bản TensorFlow/Keras - Module 1 (Deep Learning)

Đây là **phiên bản minh hoạ dùng framework thật** (`tf.keras`) của 5 bài thực hành trong `thực hành/`. Ở bản NumPy from-scratch, bạn tự viết tay forward/backward propagation, gradient checking, các cách khởi tạo trọng số, các optimizer và BatchNorm/LayerNorm. Ở bản này, cùng bài toán/dữ liệu đó được giải lại bằng `tf.keras` — nơi các phần "hạ tầng" (autodiff, optimizer, các lớp khởi tạo/chuẩn hoá có sẵn) đã được framework lo, để bạn thấy rõ **Keras đang tự động hoá đúng những gì bạn vừa tự cài tay**.

Mỗi bài `0X_.../` là **1 notebook duy nhất** (không có bản `source/` project riêng cho bản Keras này), viết theo đúng phong cách "điền vào chỗ trống" của các notebook chính: mỗi hàm có 1 bài tập nhỏ (`# YOUR CODE STARTS HERE ... # YOUR CODE ENDS HERE`) và 1 cell **Answer** dạng form thu gọn ngay bên dưới.

| Bài | Notebook | Thay thế cho |
|---|---|---|
| 01 | `01_Kien-truc-DNN-va-Backpropagation/01_Xay_dung_mo_hinh_DNN_Keras.ipynb` | `initialize_parameters_deep` + `L_model_forward/backward` + `update_parameters` → `keras.Sequential` + `model.fit()` |
| 02 | `02_Debug-gradient-Gradient-Checking/02_Kiem_tra_gradient_Keras.ipynb` | `backward_propagation`/`gradient_check_n` tự viết → `tf.GradientTape` (autodiff) |
| 03 | `03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so/03_Khoi_tao_trong_so_Keras.ipynb` | `initialize_parameters_zeros/random/he` → `kernel_initializer` (`Zeros`/`RandomNormal`/`HeNormal`) |
| 04 | `04_SGD-Momentum-Adam-Toi-uu-hoa/04_Toi_uu_hoa_Keras.ipynb` | `update_parameters_with_gd/momentum/adam` + `random_mini_batches` → `keras.optimizers.SGD/Adam` + `model.fit(batch_size=...)` |
| 05 | `05_Batch-Norm-Layer-Norm/05_Batch_Norm_Layer_Norm_Keras.ipynb` | `batchnorm_forward/backward` + `layernorm_forward/backward` → `layers.BatchNormalization()`/`layers.LayerNormalization()` |

## Cài đặt (chạy 1 lần)

```bash
cd "keras-tensorflow"
uv sync
```

> Lưu ý (Windows): nếu `uv add tensorflow==<phiên bản cụ thể>` báo lỗi thiếu wheel cho `tensorflow-io-gcs-filesystem` trên `win_amd64`, dùng `uv sync` với `pyproject.toml` đã có sẵn trong thư mục này (không ghim cứng phiên bản tensorflow) — `uv` sẽ tự chọn bản TensorFlow mới nhất tương thích Windows.

## Chạy từng bài

Mở notebook tương ứng bằng Jupyter/VS Code (đã cài qua `uv sync`), hoặc chạy trên **Google Colab** — mỗi notebook có sẵn cell tự động `git clone` repo và `cd` đúng thư mục (Colab đã cài sẵn TensorFlow/Keras, không cần cài thêm).

## Lưu ý

- `load_dataset()` trong mỗi `keras_utils.py` dùng chung bộ dữ liệu **Pima Indians Diabetes** và cách tiền xử lý giống hệt bản NumPy from-scratch, nên kết quả giữa 2 bản có thể so sánh trực tiếp với nhau.
- Dữ liệu ở quy ước `(n_x, m)`/`(1, m)` giống các bài NumPy; `compile_and_train()`/`evaluate_classification()` (đã cung cấp sẵn trong `keras_utils.py`) tự `.T` sang quy ước sample-first mà Keras yêu cầu.
- Số epoch dùng trong các bài 03/04/05 (so sánh nhiều biến thể) được giảm so với bản NumPy gốc (vốn chạy hàng nghìn iteration) để notebook chạy nhanh khi minh hoạ — vẫn đủ để quan sát rõ sự khác biệt định tính giữa các lựa chọn.
