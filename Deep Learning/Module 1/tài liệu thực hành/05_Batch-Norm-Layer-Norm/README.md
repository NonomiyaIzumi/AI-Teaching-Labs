# LAB 05 - Hướng dẫn tự thực hành

## Batch Normalization & Layer Normalization

**Notebook tương ứng:** `thực hành/05_Batch-Norm-Layer-Norm/05_Batch_Norm_Layer_Norm.ipynb`

> Notebook này do tự biên soạn (không có sẵn trong bộ bài tập gốc của Coursera) để lấp đầy nội dung *Batch Normalization, Layer Normalization* trong đề cương Chương 1. Toàn bộ code đã được **gradient-check bằng numerical gradient** (kỹ thuật học ở bài 02) trước khi đưa vào bài, sai số ~1e-9.

## 1. Phát biểu bài toán

Cài đặt Batch Normalization và Layer Normalization từ đầu (forward + backward), kiểm chứng bằng gradient checking, rồi dùng cả hai để "cứu" một mạng sâu bị khởi tạo tệ khỏi vanishing/exploding gradient — nối tiếp trực tiếp câu chuyện của bài 03.

| Thành phần | Mô tả yêu cầu |
|---|---|
| Input | Phần cài đặt: `Z` (pre-activation) shape `(n, m)`. Phần demo: `X` shape `(2, m)` — dataset "circles" giống bài 03. |
| Output | Phần cài đặt: `Z_tilde` đã chuẩn hoá + scale/shift, cùng shape với `Z`. Phần demo: `Y` nhị phân, giống bài 03. |
| Loại bài toán | Cài đặt kỹ thuật normalization (không phải thuật toán học có giám sát riêng) + thực nghiệm so sánh trên bài toán phân loại nhị phân sẵn có. |
| Mục tiêu nâng cao | Phân biệt chính xác **trục tính thống kê** giữa BN (theo vi dụ/batch) và LN (theo feature) — đây là điểm hay nhầm nhất; đồng thời phải tự gradient-check trước khi tin code đúng, không suy luận cảm tính. |
| Metric chính | Relative error của gradient check (`< 1e-6`); cost và `\|\|dW\|\|` theo layer khi train mạng demo. |

> **Điểm khác với bài 03:** Bài 03 sửa vấn đề gradient bằng cách chọn khởi tạo tốt hơn (một lần, trước khi train). Bài này sửa cùng vấn đề bằng cách chuẩn hoá **liên tục trong lúc train** (ở mỗi bước forward) — hai cách tiếp cận độc lập nhau và có thể dùng cùng lúc trong thực tế.

## 2. Dữ liệu sử dụng

| Thuộc tính | Giá trị cần kiểm tra trong notebook |
|---|---|
| Gradient check | Dữ liệu ngẫu nhiên `np.random.randn(n=4, m=5) * 5`, seed `0` (BatchNorm) và `1` (LayerNorm) |
| Demo huấn luyện | `sklearn.datasets.make_circles(n_samples=300, noise=.05)`, giống hệt bài 03 (dùng lại `load_dataset()`) |
| Kiến trúc demo | `layers_dims = [2, 20, 20, 20, 20, 1]` — 4 hidden layer + 1 output (**sâu hơn bài 03** để vanishing/exploding thể hiện rõ) |

## 3. Output bắt buộc của bài thực hành

| Nhóm output | Yêu cầu cụ thể |
|---|---|
| `batchnorm_forward`/`backward` | Gradient check `dZ`, `dgamma`, `dbeta` — relative error `< 1e-6` |
| `layernorm_forward`/`backward` | Gradient check `dZ`, `dgamma`, `dbeta` — relative error `< 1e-6` |
| Demo — huấn luyện 3 biến thể | `none` / `batchnorm` / `layernorm` trên cùng mạng 5 lớp, khởi tạo cố tình kém (`W *= 4`) |
| Biểu đồ gradient flow | 3 subplot (none/batchnorm/layernorm), `\|\|dW^[l]\|\|` theo iteration, thang `symlog` |
| Biểu đồ cost | So sánh cost theo iteration của cả 3 biến thể trên cùng một trục |
| Decision boundary | 3 subplot decision boundary tương ứng 3 biến thể |

## 4. Cấu hình thí nghiệm khuyến nghị

| Thành phần | Cấu hình khuyến nghị |
|---|---|
| Kiến trúc demo | `layers_dims = [2, 20, 20, 20, 20, 1]` |
| Hệ số khởi tạo "xấu" cố tình | `W = randn(...) * 4` |
| Learning rate | `0.1` |
| Số iteration | `3000`, ghi `\|\|dW\|\|` mỗi `track_every=10` iteration |
| `eps` (chống chia 0) | `1e-8` |
| `momentum` (running mean/var của BatchNorm) | `0.9` |
| Seed gradient check | `0` (BatchNorm), `1` (LayerNorm) |
| Ngưỡng gradient check | `relative_error < 1e-6` |

## 5. Quy trình thực hiện từng bước

1. **Đọc lại lý do cần normalization** (mục 1) — liên hệ trực tiếp vanishing/exploding gradient đã học ở bài 03.
2. **`batchnorm_forward`:** tính `mu, var` theo `axis=1` (trên các vi dụ, cho từng feature), chuẩn hoá `Z_norm = (Z-mu)/sqrt(var+eps)`, scale/shift `gamma*Z_norm+beta`; xử lý riêng `mode="train"` (dùng thống kê batch, cập nhật running mean/var) và `mode="test"` (dùng running mean/var).
3. **`batchnorm_backward`:** áp chain rule qua từng bước forward để ra `dZ, dgamma, dbeta` (công thức đầy đủ có trong notebook, mục Exercise 2.2).
4. **Gradient check BatchNorm:** dùng `numerical_gradient` (kỹ thuật bài 02) so sánh với `batchnorm_backward`, xác nhận relative error nhỏ.
5. **`layernorm_forward`/`backward`:** giống hệt BatchNorm nhưng đổi `axis=1` → `axis=0` (chuẩn hoá theo feature, cho từng vi dụ) và bỏ toàn bộ phần train/test/running-stats (không cần thiết với LN).
6. **Gradient check LayerNorm:** tương tự bước 4.
7. **Đọc bảng so sánh BN vs LN** (mục 4) — batch size, train/test, use case (CNN/MLP vs RNN/Transformer).
8. **`initialize_parameters`:** khởi tạo `W *= 4` (cố tình kém, giống "random ×10" ở bài 03 nhưng nhẹ hơn để phù hợp mạng sâu hơn), thêm `gamma=1, beta=0` cho mỗi hidden layer.
9. **`forward_propagation`/`backward_propagation`:** mạng L lớp tổng quát, chèn `batchnorm_forward`/`layernorm_forward` (nếu có) giữa `linear` và `relu` ở mỗi hidden layer.
10. **`model()`:** vòng lặp huấn luyện đầy đủ, ghi lại `\|\|dW^[l]\|\|` mỗi 10 iteration (kỹ thuật gradient flow visualization từ bài 03).
11. **Chạy 3 lần** với `norm_type` lần lượt `"none"`, `"batchnorm"`, `"layernorm"`, lưu lại lịch sử để vẽ so sánh.
12. **Vẽ gradient flow, cost curve, decision boundary** cho cả 3 biến thể trên cùng bộ trục để so sánh trực tiếp.

## 6. Kết quả mong đợi

- Cả 2 gradient check (BN và LN) đều báo relative error `< 1e-6` (thực tế đạt ~1e-9 đến 1e-11).
- **`none`**: không học được gì — cost đứng yên ở `ln(2) ≈ 0.693` (tương đương đoán ngẫu nhiên), `\|\|dW\|\|` nổ mạnh (~10⁴) ngay iteration đầu rồi chết về 0 (dead ReLU) và nằm yên suốt phần còn lại.
- **`batchnorm`**: cost giảm xuống ~0.001; `\|\|dW\|\|` ổn định trong một dải hợp lý suốt quá trình train.
- **`layernorm`**: cost giảm xuống ~0.04; `\|\|dW\|\|` cũng ổn định, không nổ/không biến mất, tuy động lực hội tụ khác BN đôi chút.
- Decision boundary: `none` tô một màu đồng nhất toàn bộ mặt phẳng; `batchnorm` và `layernorm` đều vẽ được đường biên gần đúng hình vành khăn của dữ liệu.

## 7. Bài tập mở rộng

1. **Batch size = 1:** thử train với `mini_batch_size=1` cho biến thể `batchnorm` — dự đoán điều gì sẽ xảy ra với thống kê `mu, var` tính trên batch chỉ có 1 điểm? So sánh với `layernorm` trong cùng tình huống.
2. **Tăng/giảm learning rate riêng cho LayerNorm:** thử `learning_rate = 0.15, 0.2, 0.25` chỉ cho biến thể `layernorm` (giữ nguyên 3000 iteration), so sánh cost cuối cùng — optimizer/normalization nào cũng cần tinh chỉnh hyperparameter riêng, không phải lúc nào "một cấu hình dùng chung" cũng là tối ưu.
3. **Chuẩn hoá cả layer output:** thử thêm `batchnorm`/`layernorm` ngay trước lớp Sigmoid cuối cùng (bài gốc chỉ chuẩn hoá hidden layer) — quan sát cost/accuracy có thay đổi không, giải thích tại sao thực hành chuẩn thường **không** chuẩn hoá layer output.

## 8. Tài liệu bổ sung

| Chủ đề | Liên kết tham khảo |
|---|---|
| Ioffe & Szegedy, 2015 — Batch Normalization | https://arxiv.org/abs/1502.03167 |
| Ba, Kiros & Hinton, 2016 — Layer Normalization | https://arxiv.org/abs/1607.06450 |
| CS231n — Batch Normalization notes & backward derivation | https://cs231n.github.io/neural-networks-2/#batchnorm |

## 9. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại toàn bộ notebook từ đầu trên Colab (Restart and run all) — 2 cell gradient check phải luôn in "Tat ca deu dung" trước khi xem tiếp phần sau.
- Dữ liệu là bộ "circles" tổng hợp bằng `sklearn.datasets.make_circles` (seed cố định, không phải dữ liệu thật) → không có rủi ro data leakage.
- Notebook này không thuộc bài tập gốc Coursera nên không có phần "fill in the blank" — nếu muốn giao cho học viên tự làm, có thể ẩn phần code trong `batchnorm_backward`/`layernorm_backward` và yêu cầu tự cài lại, dùng gradient check có sẵn để tự chấm.
- Sau khi chạy sạch không lỗi trên Colab, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
