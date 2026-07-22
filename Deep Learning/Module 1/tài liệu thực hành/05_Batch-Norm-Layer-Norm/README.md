# LAB 05 - Hướng dẫn tự thực hành

## Batch Normalization & Layer Normalization

**Notebook tương ứng:** `thực hành/05_Batch-Norm-Layer-Norm/05_Batch_Norm_Layer_Norm.ipynb`

**Chạy bản project (local, không cần Jupyter):** `source/05_Batch-Norm-Layer-Norm/` — xem hướng dẫn cài đặt/chạy chung tại `source/README.md`. Tóm tắt: `cd "source" && uv sync`, sau đó `cd 05_Batch-Norm-Layer-Norm && uv run --project .. python main.py`.

**Bản TensorFlow/Keras:** `keras-tensorflow/05_Batch-Norm-Layer-Norm/05_Batch_Norm_Layer_Norm_Keras.ipynb` — bài học độc lập về `layers.BatchNormalization()`/`layers.LayerNormalization()`. Cũng có notebook `keras-tensorflow/Pima_Diabetes_Nen_Tang_Deep_Learning_Keras.ipynb` — trình bày nối tiếp 5 chủ đề nền tảng Deep Learning trong 1 file. Xem `keras-tensorflow/README.md` để biết cách cài đặt/chạy local.

**Cách làm bài trong notebook:** Khác với 4 notebook trước (vốn đã có sẵn cấu trúc bài tập), notebook này ban đầu là code hoàn chỉnh — nay đã để trống riêng 2 hàm `batchnorm_backward` và `layernorm_backward` (đúng gợi ý "biến thành bài tập" ghi trong chính notebook) theo cùng cơ chế với 4 bài kia: `# YOUR CODE STARTS HERE ... # YOUR CODE ENDS HERE` để trống (`pass`), ngay dưới có cell **💡 Đáp án** dạng form thu gọn của Colab. Cell đáp án vẫn tự chạy khi bấm "Runtime → Run all" nên notebook không bao giờ báo lỗi dù chưa làm bài; để tự kiểm tra bài của mình, viết code vào cell bài tập rồi **chạy lại đúng cell đó** (không chạy cell đáp án) trước khi chạy tiếp các cell gradient-check phía sau.

> Notebook này do tự biên soạn (không có sẵn trong bộ bài tập gốc của Coursera) để lấp đầy nội dung *Batch Normalization, Layer Normalization* trong đề cương Chương 1. Toàn bộ code đã được **gradient-check bằng numerical gradient** (kỹ thuật học ở bài 02) trước khi đưa vào bài, sai số ~1e-9.

## 1. Phát biểu bài toán

Cài đặt Batch Normalization và Layer Normalization từ đầu (forward + backward), kiểm chứng bằng gradient checking, rồi dùng cả hai để "cứu" một mạng sâu bị khởi tạo tệ khỏi vanishing/exploding gradient trên dữ liệu **y tế thật** — nối tiếp trực tiếp câu chuyện của bài 03.

| Thành phần | Mô tả yêu cầu |
|---|---|
| Input | Phần cài đặt: `Z` (pre-activation) shape `(n, m)`. Phần demo: `X` shape `(8, m)` — 8 đặc trưng từ bộ dữ liệu **Pima Indians Diabetes**, giống bài 03/04. |
| Output | Phần cài đặt: `Z_tilde` đã chuẩn hoá + scale/shift, cùng shape với `Z`. Phần demo: `Y` nhị phân (0 = không tiểu đường, 1 = tiểu đường). |
| Loại bài toán | Cài đặt kỹ thuật normalization (không phải thuật toán học có giám sát riêng) + thực nghiệm so sánh trên bài toán phân loại nhị phân thật. |
| Mục tiêu nâng cao | Phân biệt chính xác **trục tính thống kê** giữa BN (theo vi dụ/batch) và LN (theo feature) — đây là điểm hay nhầm nhất; đồng thời phải tự gradient-check trước khi tin code đúng, không suy luận cảm tính. |
| Metric chính | Relative error của gradient check (`< 1e-6`); cost và `\|\|dW\|\|` theo layer khi train mạng demo; Accuracy/Precision/Recall/AUC trên test set. |

> **Điểm khác với bài 03:** Bài 03 sửa vấn đề gradient bằng cách chọn khởi tạo tốt hơn (một lần, trước khi train). Bài này sửa cùng vấn đề bằng cách chuẩn hoá **liên tục trong lúc train** (ở mỗi bước forward) — hai cách tiếp cận độc lập nhau và có thể dùng cùng lúc trong thực tế.

## 2. Dữ liệu sử dụng

| Thuộc tính | Giá trị cần kiểm tra trong notebook |
|---|---|
| Gradient check | Dữ liệu ngẫu nhiên `np.random.randn(n=4, m=5) * 5`, seed `0` (BatchNorm) và `1` (LayerNorm) — không liên quan tới dữ liệu thật, chỉ để kiểm chứng công thức đạo hàm |
| Demo huấn luyện | `data/pima-indians-diabetes.csv`, đọc qua `load_dataset()` — giống hệt cách tiền xử lý ở bài 03/04 (median-impute 5 cột thiếu dữ liệu, chuẩn hóa, chia 80/20 stratify) |
| Kiến trúc demo | `layers_dims = [train_X.shape[0], 20, 20, 20, 20, 1] = [8, 20, 20, 20, 20, 1]` — 4 hidden layer + 1 output (**sâu hơn bài 03** để vanishing/exploding thể hiện rõ) |

## 3. Output bắt buộc của bài thực hành

| Nhóm output | Yêu cầu cụ thể |
|---|---|
| `batchnorm_forward`/`backward` | Gradient check `dZ`, `dgamma`, `dbeta` — relative error `< 1e-6` |
| `layernorm_forward`/`backward` | Gradient check `dZ`, `dgamma`, `dbeta` — relative error `< 1e-6` |
| Demo — huấn luyện 3 biến thể | `none` / `batchnorm` / `layernorm` trên cùng mạng 5 lớp, khởi tạo cố tình kém (`W *= 4`) |
| Biểu đồ gradient flow | 3 subplot (none/batchnorm/layernorm), `\|\|dW^[l]\|\|` theo iteration, thang `symlog` |
| Biểu đồ cost | So sánh cost theo iteration của cả 3 biến thể trên cùng một trục |
| Confusion matrix + ROC | Lưới 2×3 (hàng trên: confusion matrix, hàng dưới: ROC curve) trên **test set**, 1 cột cho mỗi biến thể — thay cho decision boundary 2D không còn vẽ được với 8 chiều |

## 4. Cấu hình thí nghiệm khuyến nghị

| Thành phần | Cấu hình khuyến nghị |
|---|---|
| Kiến trúc demo | `layers_dims = [8, 20, 20, 20, 20, 1]` |
| Hệ số khởi tạo "xấu" cố tình | `W = randn(...) * 4` |
| Learning rate | `0.1` |
| Số iteration | `3000`, ghi `\|\|dW\|\|` mỗi `track_every=10` iteration |
| `eps` (chống chia 0) | `1e-8` |
| `momentum` (running mean/var của BatchNorm) | `0.9` |
| Seed gradient check | `0` (BatchNorm), `1` (LayerNorm) |
| Ngưỡng gradient check | `relative_error < 1e-6` |

## 5. Quy trình thực hiện từng bước

1. **Đọc lại lý do cần normalization** (mục 1) — liên hệ trực tiếp vanishing/exploding gradient đã học ở bài 03.
2. **`batchnorm_forward`** (đã cho sẵn): tính `mu, var` theo `axis=1` (trên các vi dụ, cho từng feature), chuẩn hoá `Z_norm = (Z-mu)/sqrt(var+eps)`, scale/shift `gamma*Z_norm+beta`; xử lý riêng `mode="train"` (dùng thống kê batch, cập nhật running mean/var) và `mode="test"` (dùng running mean/var).
3. **Exercise 1 - `batchnorm_backward`:** áp chain rule qua từng bước forward để ra `dZ, dgamma, dbeta` (công thức đầy đủ có trong notebook, mục Exercise 2.2).
4. **Gradient check BatchNorm:** dùng `numerical_gradient` (kỹ thuật bài 02) so sánh với `batchnorm_backward`, xác nhận relative error nhỏ.
5. **`layernorm_forward`** (đã cho sẵn): giống hệt BatchNorm nhưng đổi `axis=1` → `axis=0` (chuẩn hoá theo feature, cho từng vi dụ) và bỏ toàn bộ phần train/test/running-stats (không cần thiết với LN).
6. **Exercise 2 - `layernorm_backward`:** tương tự Exercise 1 nhưng đổi trục, tự kiểm chứng bằng gradient check.
7. **Đọc bảng so sánh BN vs LN** (mục 4) — batch size, train/test, use case (CNN/MLP vs RNN/Transformer).
8. **`initialize_parameters`:** khởi tạo `W *= 4` (cố tình kém, giống "random ×10" ở bài 03 nhưng nhẹ hơn để phù hợp mạng sâu hơn), thêm `gamma=1, beta=0` cho mỗi hidden layer.
9. **`forward_propagation`/`backward_propagation`:** mạng L lớp tổng quát, chèn `batchnorm_forward`/`layernorm_forward` (nếu có) giữa `linear` và `relu` ở mỗi hidden layer.
10. **`model()`:** vòng lặp huấn luyện đầy đủ, ghi lại `\|\|dW^[l]\|\|` mỗi 10 iteration (kỹ thuật gradient flow visualization từ bài 03).
11. **Chạy 3 lần** với `norm_type` lần lượt `"none"`, `"batchnorm"`, `"layernorm"`, lưu lại lịch sử để vẽ so sánh.
12. **Vẽ gradient flow, cost curve** cho cả 3 biến thể trên cùng bộ trục để so sánh trực tiếp.
13. **Đánh giá trên test set:** `evaluate_classification` cho từng biến thể, vẽ lưới confusion matrix + ROC curve.

## 6. Kết quả mong đợi

- Cả 2 gradient check (BN và LN) đều báo relative error `< 1e-6` (thực tế đạt ~1e-9 đến 1e-11).
- **`none`**: không học được gì — cost đứng yên ở `ln(2) ≈ 0.693` (đúng bằng baseline đoán lớp đa số), `\|\|dW\|\|` nổ mạnh ngay iteration đầu rồi chết về 0 (dead ReLU) và nằm yên suốt phần còn lại; test set: Accuracy ~65%, **Precision = Recall = 0, AUC = 0.5** (không phân biệt được lớp nào cả).
- **`batchnorm`**: cost giảm xuống ~0.11; test Accuracy ~61%, Precision/Recall khiêm tốn nhưng khác 0, AUC ~0.65.
- **`layernorm`**: cost giảm xuống ~0.31; test Accuracy ~69%, AUC ~0.74 — tốt nhất trong 3 biến thể ở cấu hình này.
- Điểm mấu chốt cần nhận ra: cả BN và LN đều "cứu" được mạng khỏi trạng thái hoàn toàn không học (AUC=0.5 của `none`), dù kết quả cuối cùng khác nhau.

## 7. Bài tập mở rộng

1. **Batch size = 1:** thử train với `mini_batch_size=1` cho biến thể `batchnorm` — dự đoán điều gì sẽ xảy ra với thống kê `mu, var` tính trên batch chỉ có 1 điểm? So sánh với `layernorm` trong cùng tình huống.
2. **Tăng/giảm learning rate riêng cho LayerNorm:** thử `learning_rate = 0.15, 0.2, 0.25` chỉ cho biến thể `layernorm` (giữ nguyên 3000 iteration), so sánh cost/AUC cuối cùng — optimizer/normalization nào cũng cần tinh chỉnh hyperparameter riêng, không phải lúc nào "một cấu hình dùng chung" cũng là tối ưu.
3. **Chuẩn hoá cả layer output:** thử thêm `batchnorm`/`layernorm` ngay trước lớp Sigmoid cuối cùng (bài gốc chỉ chuẩn hoá hidden layer) — quan sát cost/AUC có thay đổi không, giải thích tại sao thực hành chuẩn thường **không** chuẩn hoá layer output.
4. **So sánh với bài 03:** kiến trúc bài này sâu hơn hẳn (5 lớp so với 3 lớp) và khởi tạo tệ hơn (`*4` so với `*10` áp cho toàn 3 lớp) — thử áp He initialization (bài 03) thay vì `random_bad` cho kiến trúc 5 lớp này, so sánh với việc dùng BatchNorm/LayerNorm: hai hướng tiếp cận nào hiệu quả hơn ở đây?

## 8. Tài liệu bổ sung

| Chủ đề | Liên kết tham khảo |
|---|---|
| Ioffe & Szegedy, 2015 — Batch Normalization | https://arxiv.org/abs/1502.03167 |
| Ba, Kiros & Hinton, 2016 — Layer Normalization | https://arxiv.org/abs/1607.06450 |
| CS231n — Batch Normalization notes & backward derivation | https://cs231n.github.io/neural-networks-2/#batchnorm |
| Pima Indians Diabetes Dataset (nguồn gốc) | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |

## 9. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all) — 2 cell gradient check phải luôn in "Tat ca deu dung" trước khi xem tiếp phần sau.
- Dữ liệu là **Pima Indians Diabetes thật** (phần demo huấn luyện) — tập train/test tách rõ trong `load_dataset()` (kèm `stratify`), tiền xử lý chỉ fit trên train → không có rủi ro data leakage. Riêng 2 cell gradient check vẫn dùng dữ liệu ngẫu nhiên cố định (không liên quan dữ liệu thật), đúng mục đích kiểm chứng công thức đạo hàm.
- Sau khi chạy sạch không lỗi trên Colab, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
