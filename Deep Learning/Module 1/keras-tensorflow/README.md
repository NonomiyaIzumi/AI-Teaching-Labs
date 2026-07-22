# Bản TensorFlow/Keras - Module 1 (Deep Learning)

Bản `tf.keras` của 5 bài thực hành Module 1, viết theo phong cách "dien vao cho trong": mỗi hàm
có 1 bài tập nhỏ (`# YOUR CODE STARTS HERE ... # YOUR CODE ENDS HERE`) và 1 cell **Answer** dạng
form thu gọn ngay bên dưới. Cả 5 chủ đề đều dùng chung một bộ dữ liệu y tế thật: **Pima Indians
Diabetes** (768 bệnh nhân, 8 đặc trưng lâm sàng, dự đoán tiểu đường).

Có **2 lựa chọn**, chọn 1 tuỳ nhu cầu:

## Lựa chọn 1 - `Pima_Diabetes_Nen_Tang_Deep_Learning_Keras.ipynb`

**1 file duy nhất**, trình bày 5 phần nối tiếp nhau: Xây dựng mạng DNN → Gradient Checking/Autodiff
→ Khởi tạo trọng số → Optimizer → BatchNorm/LayerNorm — đi từ kiến trúc mạng cơ bản đến các kỹ
thuật giúp mạng học ổn định và hiệu quả hơn, phù hợp để học liền mạch trong một buổi. Không cần
clone repo - notebook tự tải file dữ liệu CSV về (1 cell duy nhất, xem mục Cài đặt bên dưới) và
chạy độc lập từ đầu đến cuối. Tài liệu hướng dẫn chi tiết:
`../tài liệu thực hành/Pima_Diabetes_Nen_Tang_Deep_Learning_Keras/README.md`.

## Lựa chọn 2 - 5 notebook riêng theo từng chủ đề (tiện học/ôn từng phần một)

| Bài | Notebook | Chủ đề |
|---|---|---|
| 01 | `01_Kien-truc-DNN-va-Backpropagation/01_Xay_dung_mo_hinh_DNN_Keras.ipynb` | Kiến trúc DNN, forward/backward, `keras.Sequential` |
| 02 | `02_Debug-gradient-Gradient-Checking/02_Kiem_tra_gradient_Keras.ipynb` | Autodiff (`tf.GradientTape`), Gradient Checking |
| 03 | `03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so/03_Khoi_tao_trong_so_Keras.ipynb` | Vanishing/Exploding Gradient, `kernel_initializer` |
| 04 | `04_SGD-Momentum-Adam-Toi-uu-hoa/04_Toi_uu_hoa_Keras.ipynb` | Optimizer: GD, Momentum, Adam |
| 05 | `05_Batch-Norm-Layer-Norm/05_Batch_Norm_Layer_Norm_Keras.ipynb` | `BatchNormalization`, `LayerNormalization` |

Mỗi bài là notebook độc lập, tự tải dữ liệu qua `keras_utils.py` riêng của nó.

## Cài đặt (chạy 1 lần, cho cả 2 lựa chọn)

```bash
cd "keras-tensorflow"
uv sync
```

> Lưu ý (Windows): nếu `uv add tensorflow==<phiên bản cụ thể>` báo lỗi thiếu wheel cho `tensorflow-io-gcs-filesystem` trên `win_amd64`, dùng `uv sync` với `pyproject.toml` đã có sẵn trong thư mục này (không ghim cứng phiên bản tensorflow) — `uv` sẽ tự chọn bản TensorFlow mới nhất tương thích Windows.

## Chạy notebook

Mở bằng Jupyter/VS Code (đã cài qua `uv sync`), hoặc chạy trực tiếp trên **Google Colab** (Colab
đã cài sẵn TensorFlow/Keras). File CSV được tự động tải về ở cell đầu notebook — không cần clone
repo hay upload thủ công.

## Lưu ý

- Dữ liệu ở quy ước `(n_x, m)`/`(1, m)`; `compile_and_train()`/`evaluate_classification()` (đã
  cung cấp sẵn) tự `.T` sang quy ước sample-first mà Keras yêu cầu.
- Số epoch dùng ở các phần so sánh nhiều biến thể (khởi tạo, optimizer, BatchNorm/LayerNorm) được
  chọn ở mức vừa đủ để notebook chạy nhanh mà vẫn thấy rõ khác biệt định tính giữa các lựa chọn.
- Các con số cụ thể (accuracy, cost...) có thể lệch nhau vài phần trăm giữa các lần chạy dù đã cố
  định seed — đặc điểm bình thường của gradient descent trên CPU nhiều luồng, không phải lỗi.
