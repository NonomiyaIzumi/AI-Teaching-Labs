# Pima Diabetes - Nền tảng Deep Learning (TensorFlow/Keras) - Hướng dẫn tự thực hành

## Nền tảng mạng nơ-ron và học sâu, thực hành bằng tf.keras

**Notebook tương ứng:** `keras-tensorflow/Pima_Diabetes_Nen_Tang_Deep_Learning_Keras.ipynb`

**Đây là bản gộp** cả 5 chủ đề nền tảng của Module 1 thành **1 file duy nhất**, dùng `tf.keras` và dùng chung một bộ dữ liệu y tế thật (Pima Indians Diabetes) xuyên suốt:

| Phần | Chủ đề | Notebook riêng tương ứng (nếu muốn học tách rời) |
|---|---|---|
| 1 | Xây dựng và huấn luyện mạng DNN (`keras.Sequential`, forward/backward/gradient descent) | `keras-tensorflow/01_Kien-truc-DNN-va-Backpropagation/01_Xay_dung_mo_hinh_DNN_Keras.ipynb` |
| 2 | Tính đạo hàm tự động (`tf.GradientTape`) và Gradient Checking | `keras-tensorflow/02_Debug-gradient-Gradient-Checking/02_Kiem_tra_gradient_Keras.ipynb` |
| 3 | Khởi tạo trọng số và Vanishing/Exploding Gradient (`kernel_initializer`) | `keras-tensorflow/03_Vanishing-Exploding-gradient-va-Khoi-tao-trong-so/03_Khoi_tao_trong_so_Keras.ipynb` |
| 4 | Các thuật toán tối ưu hoá: GD, Momentum, Adam | `keras-tensorflow/04_SGD-Momentum-Adam-Toi-uu-hoa/04_Toi_uu_hoa_Keras.ipynb` |
| 5 | Batch Normalization và Layer Normalization | `keras-tensorflow/05_Batch-Norm-Layer-Norm/05_Batch_Norm_Layer_Norm_Keras.ipynb` |

**Cài đặt (chạy 1 lần):** `cd "keras-tensorflow" && uv sync`. Chi tiết ở `keras-tensorflow/README.md`.

**Không cần clone repo hay tải file thủ công:** cell code đầu tiên của notebook tự tải file `pima-indians-diabetes.csv` về từ GitHub raw URL của repo này (chỉ 1 file CSV ~24KB, không phải clone cả repo) — notebook chạy độc lập trên cả Google Colab lẫn Jupyter local.

**Cách làm bài trong notebook:** Mỗi phần có 1-2 bài tập nhỏ, mỗi hàm có đoạn `# YOUR CODE STARTS HERE ... # YOUR CODE ENDS HERE` để trống (`pass`). Ngay bên dưới mỗi bài là 1 cell **Answer** dạng form thu gọn của Colab (bấm mũi tên bên trái hoặc chọn "Show code" ở menu ba chấm để mở ra xem). Vì cell đáp án vẫn tự chạy khi bấm "Runtime → Run all", notebook **không bao giờ báo lỗi** dù chưa làm bài — nhưng lưu ý: chạy Run all như vậy nghĩa là đang xem code đáp án, không phải code tự viết. Để tự kiểm tra bài của mình, viết code vào cell bài tập rồi **chạy lại đúng cell đó** (không chạy cell đáp án ngay dưới) trước khi chạy tiếp các cell phía sau.

## 1. Phát biểu bài toán tổng thể

Notebook đi qua **5 khối kiến thức nền tảng** cần có trước khi xây dựng một mô hình deep learning "chạy được, ổn định, và học đúng" bằng framework thật:

1. **Kiến trúc & huấn luyện cơ bản** — biết dựng một mạng DNN và hiểu 4 bước forward/cost/backward/update mà `model.fit()` đang tự động hoá.
2. **Công cụ debug** — biết dùng autodiff (`tf.GradientTape`) và gradient checking để tin tưởng (có kiểm chứng) vào một phép tính mới tự viết.
3. **Khởi tạo trọng số** — biết vì sao cùng một kiến trúc có thể "học được" hay "chết hoàn toàn" chỉ vì cách khởi tạo trọng số ban đầu.
4. **Thuật toán tối ưu** — biết chọn giữa GD/Momentum/Adam và hiểu đánh đổi giữa chúng.
5. **Chuẩn hoá trong lúc train** — biết dùng BatchNorm/LayerNorm để "cứu" một mạng bị khởi tạo kém.

Cả 5 phần dùng chung **một bộ công cụ đánh giá** (`evaluate_classification` — Accuracy/Precision/Recall/F1/ROC-AUC/confusion matrix) nên có thể so sánh trực tiếp hiệu quả của từng kỹ thuật trên cùng một bài toán thật.

## 2. Dữ liệu sử dụng (dùng chung cho cả 5 phần)

Bộ dữ liệu y tế thật **Pima Indians Diabetes**: 768 bệnh nhân nữ gốc Ấn Độ (Pima), 8 đặc trưng lâm sàng, nhãn nhị phân (0 = không mắc tiểu đường, 1 = mắc tiểu đường).

| Thuộc tính | Giá trị |
|---|---|
| Nguồn tải | Tự động tải qua cell đầu notebook (raw GitHub URL của chính repo này) |
| Số đặc trưng | 8 (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) |
| Xử lý giá trị thiếu | Median imputation cho 5 cột không thể bằng 0 về sinh lý (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`), tính trên tập train, áp lại cho tập test |
| Chuẩn hoá | Z-score (trừ trung bình, chia độ lệch chuẩn), tính trên tập train |
| Chia tập | 80% train (614 mẫu) / 20% test (154 mẫu), `stratify` theo nhãn, `random_state=1` |
| Quy ước shape | `(n_x, m)` cho `X`, `(1, m)` cho `Y` — giống quy ước dùng trong các bài NumPy from-scratch của Module 1 |

## 3. Output bắt buộc theo từng phần

| Phần | Bài tập chính | Output cần có |
|---|---|---|
| 1 | `build_model(input_dim, hidden_units)` | `model.summary()` đúng shape; huấn luyện xong in cost/accuracy; confusion matrix + ROC trên test set |
| 2 | `dtheta_autodiff`, `compute_gradients_autodiff` | `dtheta` khớp lý thuyết (`= x`); gradient checking mạng `[8,5,3,1]` có `difference` nhỏ (assert `< 5e-3`) |
| 3 | `build_model_with_initializer` | Huấn luyện + đánh giá 3 biến thể `zeros`/`random_bad`/`he`; biểu đồ cost overlay; confusion matrix + ROC cho từng biến thể |
| 4 | `get_optimizer` | Huấn luyện + đánh giá 3 optimizer `gd`/`momentum`/`adam`; biểu đồ cost overlay; confusion matrix + ROC cho từng optimizer |
| 5 | `build_model_with_norm` | Huấn luyện + đánh giá 3 biến thể `none`/`batchnorm`/`layernorm`; biểu đồ cost overlay; confusion matrix + ROC cho từng biến thể |

## 4. Cấu hình thí nghiệm dùng trong notebook

| Phần | Kiến trúc | Optimizer / learning rate | Epoch |
|---|---|---|---|
| 1 | `[8, 7, 1]` | `SGD(lr=0.5)`, full-batch | 3000 |
| 2 | `[8, 5, 3, 1]` (chỉ để kiểm tra gradient, không train) | — | — |
| 3 | `[8, 10, 5, 1]` | `SGD(lr=0.01)`, full-batch | 2000 (mỗi biến thể) |
| 4 | `[8, 5, 2, 1]`, khởi tạo He | `lr=0.0007`, `batch_size=64` | 200 (mỗi optimizer) |
| 5 | `[8, 20, 20, 20, 20, 1]`, khởi tạo kém cố ý (`stddev=4.0`) | `SGD(lr=0.1)`, full-batch | 2000 (mỗi biến thể) |

## 5. Quy trình thực hiện

1. **Chạy cell tải dữ liệu** (đầu notebook) — xác nhận in ra `Da tai du lieu ve: pima-indians-diabetes.csv` (hoặc `Du lieu da co san` nếu chạy lại).
2. **Chạy cell định nghĩa dữ liệu + hàm dùng chung** (`load_dataset`, `compile_and_train`, `evaluate_classification`) — các hàm này được dùng lại ở cả 5 phần phía sau, chỉ định nghĩa 1 lần.
3. **Phần 1:** đọc lý thuyết mục 1.1-1.2, làm bài tập `build_model`, chạy huấn luyện + đánh giá, đọc phần Kết quả & Phân tích.
4. **Phần 2:** đọc lý thuyết mục 2.1-2.2, làm bài tập `dtheta_autodiff` (trường hợp 1D), rồi `compute_gradients_autodiff` (trường hợp mạng nhỏ), chạy gradient checking.
5. **Phần 3:** đọc lý thuyết mục 3.1-3.2, làm bài tập `build_model_with_initializer`, chạy huấn luyện + so sánh 3 cách khởi tạo.
6. **Phần 4:** đọc lý thuyết mục 4.1-4.2, làm bài tập `get_optimizer`, chạy huấn luyện + so sánh 3 optimizer.
7. **Phần 5:** đọc lý thuyết mục 5.1-5.2, làm bài tập `build_model_with_norm`, chạy huấn luyện + so sánh `none`/`batchnorm`/`layernorm`.
8. **Đọc phần Tổng kết chung** ở cuối notebook — liên hệ lại cả 5 kỹ thuật trên cùng một bài toán.

## 6. Kết quả mong đợi

> Các con số dưới đây đã được xác minh bằng cách chạy toàn bộ notebook từ đầu đến cuối; có thể lệch nhau vài phần trăm giữa các lần chạy dù đã cố định seed — đặc điểm bình thường của gradient descent trên CPU nhiều luồng, không phải lỗi.

**Phần 1** — train Accuracy ~81%, test Accuracy ~74% (Precision ~68%, Recall ~48%, F1 ~0.57).

**Phần 2** — `dtheta` khớp chính xác lý thuyết (`= x`); gradient checking mạng nhỏ cho `difference ≈ 7.9e-04` (qua ngưỡng `< 5e-3` thoải mái).

**Phần 3** — `zeros`: test Accuracy 64.9%, **Precision = Recall = 0** (không học được gì). `random_bad` (stddev=10): tương tự, 64.9%/Recall=0. `he`: test Accuracy 68.2%, Recall 51.9% (học được tín hiệu thật).

**Phần 4** — `gd`: test Accuracy 67.5% (Recall 25.9%). `momentum`: 72.1% (Recall 38.9%). `adam`: cao nhất 74.7% (Recall 46.3%) — thứ tự Adam > Momentum > GD đúng lý thuyết.

**Phần 5** — `none`: test Accuracy 64.9%, **Precision = Recall = 0** (khởi tạo kém trên mạng sâu khiến mạng "chết"). `batchnorm`: 69.5% (Recall 59.3%). `layernorm`: 62.3% (Recall 46.3%) — cả hai đều "cứu" được mạng so với `none`, batchnorm hiệu quả hơn ở cấu hình này.

## 7. Bài tập mở rộng

1. **Phần 1:** thử `hidden_units=20` hoặc thêm 1 lớp `Dense`; thử `class_weight` để cải thiện Recall.
2. **Phần 2:** áp dụng lại `dtheta_autodiff`-style để tự viết gradient check cho một hàm loss tuỳ chỉnh khác.
3. **Phần 3:** thử `GlorotNormal` thay vì `HeNormal`; thử kiến trúc sâu hơn (5-6 hidden layer) để xem `zeros`/`random_bad` có "hỏng" nặng hơn.
4. **Phần 4:** tăng `epochs` cho `gd`/`momentum` xem có bắt kịp `adam` không; thử learning rate riêng cho từng optimizer.
5. **Phần 5:** thử `batch_size=1` cho `batchnorm`; so sánh BatchNorm/LayerNorm khi kết hợp với khởi tạo tốt (`HeNormal`) thay vì khởi tạo kém cố ý.
6. **Liên hệ xuyên suốt:** Phần 3 (khởi tạo tốt) và Phần 5 (chuẩn hoá trong lúc train) đều giải quyết chung một vấn đề (vanishing/exploding gradient) theo hai hướng độc lập — thử kết hợp cả hai (khởi tạo He **và** BatchNorm) trên kiến trúc của Phần 5, so sánh với chỉ dùng riêng lẻ từng kỹ thuật.

## 8. Tài liệu tham khảo

| Chủ đề | Liên kết |
|---|---|
| `tf.keras.Sequential` API | https://www.tensorflow.org/guide/keras/sequential_model |
| `tf.GradientTape` guide | https://www.tensorflow.org/guide/autodiff |
| He et al., 2015 - He initialization | https://arxiv.org/abs/1502.01852 |
| Glorot & Bengio, 2010 - Xavier initialization | https://proceedings.mlr.press/v9/glorot10a.html |
| Kingma & Ba, 2015 - Adam | https://arxiv.org/abs/1412.6980 |
| Ioffe & Szegedy, 2015 - Batch Normalization | https://arxiv.org/abs/1502.03167 |
| Ba, Kiros & Hinton, 2016 - Layer Normalization | https://arxiv.org/abs/1607.06450 |
| Pima Indians Diabetes Dataset (nguồn gốc) | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |

## 9. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại **toàn bộ notebook từ đầu** trên Colab (Runtime → Restart and run all) hoặc Jupyter local — xác nhận cell tải dữ liệu chạy được và không có `Exception` ở bất kỳ cell nào.
- Notebook có 5 phần huấn luyện khá dài (Phần 1: 3000 epoch; Phần 3/5: 3 biến thể × 2000 epoch; Phần 4: 3 biến thể × 200 epoch) — chạy hết một lượt trên máy CPU thường mất vài phút, không phải lỗi treo.
- Dữ liệu là **Pima Diabetes thật**, train/test tách rõ (có `stratify`, không rò rỉ dữ liệu test vào bước tiền xử lý).
- Sau khi chạy sạch không lỗi, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
