# LAB 01 - Hướng dẫn tự thực hành

## Xây dựng mô hình DNN (Kiến trúc & Backpropagation)

**Notebook tương ứng:** `thực hành/01_Kien-truc-DNN-va-Backpropagation/01_Xay_dung_mo_hinh_DNN.ipynb`

**Chạy bản project (local, không cần Jupyter):** `source/01_Kien-truc-DNN-va-Backpropagation/` — xem hướng dẫn cài đặt/chạy chung tại `source/README.md`. Tóm tắt: `cd "source" && uv sync`, sau đó `cd 01_Kien-truc-DNN-va-Backpropagation && uv run --project .. python main.py`.

**Bản TensorFlow/Keras:** `keras-tensorflow/01_Kien-truc-DNN-va-Backpropagation/01_Xay_dung_mo_hinh_DNN_Keras.ipynb` — bài học độc lập cùng chủ đề, dùng `tf.keras.Sequential`/`model.fit()`. Cũng có bản gộp cả 5 chủ đề Module 1 thành 1 file: `keras-tensorflow/Pima_Diabetes_Nen_Tang_Deep_Learning_Keras.ipynb`. Xem `keras-tensorflow/README.md` để biết cách cài đặt/chạy local (`uv sync` trong `keras-tensorflow/`).

**Cách làm bài trong notebook:** Mỗi hàm `# GRADED FUNCTION` đã được để trống (`pass`) ở đoạn `# YOUR CODE STARTS HERE ... # YOUR CODE ENDS HERE` — bạn tự viết code vào đó. Ngay bên dưới mỗi bài là 1 cell **💡 Đáp án** dạng form thu gọn của Colab (bấm mũi tên bên trái hoặc chọn "Show code" ở menu ba chấm để mở ra xem). Vì cell đáp án vẫn tự chạy khi bạn bấm "Runtime → Run all", notebook sẽ **không bao giờ báo lỗi** dù bạn chưa làm bài nào — nhưng lưu ý: chạy Run all như vậy nghĩa là bạn đang xem code của đáp án, không phải code bạn viết. Để tự kiểm tra bài của mình, sau khi viết code vào cell bài tập, hãy **chạy lại đúng cell đó** (không chạy cell đáp án ngay dưới) rồi mới chạy tiếp các cell demo/test phía sau — vì Python luôn dùng định nghĩa hàm được chạy **gần nhất**.

## 1. Phát biểu bài toán

Cài đặt từ đầu (chỉ dùng NumPy, không dùng framework) toàn bộ các khối xây dựng của một mạng DNN L lớp: khởi tạo tham số, forward propagation, hàm mất mát, backward propagation và cập nhật tham số — để dùng làm nền tảng cho mọi bài sau trong Module 1.

| Thành phần | Mô tả yêu cầu |
|---|---|
| Input | Vector/ma trận đặc trưng `X`, shape `(n_x, m)` — `n_x` số đặc trưng, `m` số vi dụ. Trong bài này, `X` chỉ là các mảng số cố định lấy từ `testCases.py` để kiểm thử hàm, chưa phải dữ liệu ảnh thật. |
| Output | Với mỗi hàm: giá trị/dictionary tương ứng (`parameters`, `AL`, `cost`, `grads`...) — không có "nhãn dự đoán" theo nghĩa một bài toán train hoàn chỉnh, vì bài này dừng ở mức xây dựng function, chưa train trên dataset thật. |
| Loại bài toán | Cài đặt thuật toán (forward/backward propagation) cho mạng nơ-ron L lớp tổng quát; kiểm thử bằng unit test có sẵn (`testCases.py`, `public_tests.py`). |
| Mục tiêu nâng cao | Không chỉ chép công thức — phải tự suy luận đúng shape của từng ma trận `W^[l]`, `b^[l]`, `Z^[l]`, `A^[l]` ở mỗi bước, vì đây là nguyên nhân phổ biến nhất gây lỗi khi mở rộng mạng lên nhiều lớp. |
| Metric chính | Không có metric huấn luyện (chưa train) — "đúng/sai" được xác định qua so khớp với **expected output** in sẵn trong notebook và qua các cell `..._test(...)`. |

> **Điểm khác với các bài dùng framework (PyTorch):** Ở đây không có `nn.Linear`, không có autograd — mọi phép nhân ma trận, mọi đạo hàm đều tự viết tay bằng NumPy. Vì vậy sai một dấu `.T`, một `axis`, hay quên `keepdims=True` là lỗi thường gặp nhất; đọc kỹ shape ghi trong docstring trước khi code.

## 2. Dữ liệu sử dụng

Bài này **không dùng dataset thật**. Toàn bộ input để kiểm thử hàm được sinh sẵn trong `testCases.py` (mảng số cố định, không phải ảnh/văn bản), và `public_tests.py` chứa các test case đối chiếu kết quả — dùng chung logic với `test_utils.py` (không phụ thuộc gói `dlai_tools` nội bộ của Coursera, nên chạy được bình thường ngoài môi trường Coursera).

| Thuộc tính | Giá trị cần kiểm tra trong notebook |
|---|---|
| Nguồn input test | `testCases.py` (cùng thư mục với notebook) |
| Shape test case — mạng 2 lớp | `initialize_parameters(n_x=3, n_h=2, n_y=1)` |
| Shape test case — mạng L lớp | `initialize_parameters_deep([5, 4, 3])` (2 lớp: 5→4→3) |
| Framework chấm điểm | `public_tests.py` + `test_utils.py` (tự chứa, không cần cài thêm gói ngoài) |

## 3. Output bắt buộc của bài thực hành

Notebook nộp bài cần chạy được tuần tự từ cell đầu đến cell cuối, mọi cell test đều phải **pass** (không có `AssertionError`).

| Nhóm output | Yêu cầu cụ thể |
|---|---|
| Khởi tạo tham số | In `W1, b1, W2, b2` (mạng 2 lớp) và `W1, b1, W2, b2` (mạng L lớp, ví dụ `[5,4,3]`) đúng shape, khớp expected output |
| Forward propagation | `linear_forward`, `linear_activation_forward`, `L_model_forward` chạy đúng, `AL` có shape `(1, m)`, giá trị nằm trong `(0,1)` (do Sigmoid ở lớp cuối) |
| Cost | `compute_cost` trả về một số vô hướng (scalar), không phải mảng |
| Backward propagation | `linear_backward`, `linear_activation_backward`, `L_model_backward` trả đúng `dW`, `db`, `dA_prev` khớp expected output |
| Cập nhật tham số | `update_parameters` trả về `parameters` đã cập nhật đúng công thức `W := W - learning_rate * dW` |
| Toàn bộ 10 test cell | Mỗi test cell (`..._test(...)`) in ra kết quả **pass**, không có exception |

## 4. Cấu hình thí nghiệm khuyến nghị

Bài này không "train" nên không có learning rate/epoch — cấu hình dưới đây là các **shape chuẩn** dùng để tự kiểm tra (đã khớp sẵn trong notebook, không cần đổi khi chạy bản chính):

| Thành phần | Cấu hình khuyến nghị |
|---|---|
| Kiến trúc test — 2 lớp | `n_x=3, n_h=2, n_y=1` |
| Kiến trúc test — L lớp | `layers_dims = [5, 4, 3]` |
| Activation hidden layer | ReLU |
| Activation output layer | Sigmoid |
| Learning rate dùng trong `update_parameters` (test case) | `0.1` |

## 5. Quy trình thực hiện từng bước

1. **Import & kiểm tra môi trường:** import `numpy`, `matplotlib`, `h5py`, `testCases`, `public_tests`, `dnn_utils` (`sigmoid`, `relu` và đạo hàm của chúng).
2. **Đọc lại kiến trúc tổng quát (mục 2 - Outline):** hiểu sơ đồ `LINEAR -> RELU` lặp lại `L-1` lần rồi `LINEAR -> SIGMOID` ở lớp cuối, và quy ước ký hiệu `[l]` (layer), `(i)` (example).
3. **Exercise 1 - `initialize_parameters`:** khởi tạo `W1, b1, W2, b2` cho mạng 2 lớp — `W` khởi tạo ngẫu nhiên nhỏ (`*0.01`), `b` khởi tạo 0.
4. **Exercise 2 - `initialize_parameters_deep`:** tổng quát hoá Exercise 1 cho `layers_dims` bất kỳ độ dài, dùng vòng lặp `for l in range(1, L)`.
5. **Exercise 3 - `linear_forward`:** tính `Z = np.dot(W, A) + b`, trả về `Z` và `cache = (A, W, b)`.
6. **Exercise 4 - `linear_activation_forward`:** gọi `linear_forward` rồi áp activation (`sigmoid` hoặc `relu`), lưu cả `linear_cache` và `activation_cache`.
7. **Exercise 5 - `L_model_forward`:** lặp `linear_activation_forward` với ReLU cho `L-1` lớp đầu, Sigmoid cho lớp cuối; trả về `AL` và danh sách `caches`.
8. **Exercise 6 - `compute_cost`:** cross-entropy loss `-1/m * sum(Y*log(AL) + (1-Y)*log(1-AL))`.
9. **Exercise 7 - `linear_backward`:** từ `dZ` và `cache`, tính `dW = 1/m * dZ.A_prev^T`, `db = 1/m * sum(dZ)`, `dA_prev = W^T.dZ`.
10. **Exercise 8 - `linear_activation_backward`:** áp `sigmoid_backward`/`relu_backward` lên `dA` để ra `dZ`, rồi gọi `linear_backward`.
11. **Exercise 9 - `L_model_backward`:** khởi tạo `dAL` từ đạo hàm cross-entropy, lan truyền ngược qua Sigmoid (lớp cuối) rồi qua từng ReLU layer theo thứ tự ngược.
12. **Exercise 10 - `update_parameters`:** cập nhật toàn bộ `W^[l]`, `b^[l]` bằng gradient descent với `learning_rate`.

## 6. Kết quả mong đợi

- Toàn bộ 10 test cell chạy không lỗi, giá trị in ra khớp **Expected Output** ghi ngay dưới mỗi bài tập trong notebook.
- Đây là "thư viện hàm" nền tảng — các bài 02–05 trong Module 1 đều tái sử dụng đúng công thức forward/backward đã kiểm chứng ở đây (chỉ thay cách khởi tạo, thêm normalization, hoặc đổi thuật toán cập nhật tham số).

## 7. [Bonus] Áp dụng vào dữ liệu thật

Notebook có thêm mục **"11. [Bonus] Áp dụng vào dữ liệu thật"** ở cuối (sau cell "Congratulations!") — dùng chính các hàm vừa cài (`initialize_parameters_deep`, `L_model_forward`, `compute_cost`, `L_model_backward`, `update_parameters`) để lắp thành một hàm huấn luyện đầy đủ `L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost)`, rồi train trên bộ dữ liệu y tế thật **Pima Indians Diabetes** (768 bệnh nhân, 8 đặc trưng lâm sàng, dự đoán tiểu đường) qua `load_dataset()` — cùng bộ dữ liệu và cách tiền xử lý sẽ dùng lại xuyên suốt các bài 03–05.

| Thành phần | Giá trị |
|---|---|
| Kiến trúc | `layers_dims = [train_X.shape[0], 7, 1] = [8, 7, 1]` (1 hidden layer) |
| Learning rate | `0.5` |
| Số iteration | `3000` |
| Kết quả | Train accuracy ≈ **83.1%**, Test accuracy ≈ **74.0%** |

Đây chỉ là bản demo đơn giản (1 hidden layer, khởi tạo `*0.01` mặc định) — thử tăng lên mạng sâu hơn (vd `[8, 20, 7, 5, 1]`) sẽ thấy accuracy tụt lại ngang baseline (~65%, tức đoán toàn bộ về lớp đa số) do vanishing gradient với cách khởi tạo ngây thơ này. Đây chính là vấn đề mà **bài 03** sẽ giải quyết bằng khởi tạo He/Xavier.

## 8. Bài tập mở rộng

1. **Tự vẽ sơ đồ shape:** với `layers_dims = [4, 3, 2, 1]`, viết ra giấy shape của từng `W^[l]`, `b^[l]`, `Z^[l]`, `A^[l]` trước khi chạy code — rồi in shape thật trong notebook (`.shape`) để đối chiếu.
2. **Thêm activation Tanh:** cài thêm nhánh `tanh` trong `linear_activation_forward`/`backward` (dùng `np.tanh` và đạo hàm `1 - tanh(z)^2`), so sánh kết quả `AL` với bản ReLU trên cùng test case.
3. **Gộp forward+backward thành một hàm `L_layer_train_step`:** nhận `X, Y, parameters, layers_dims`, gọi lần lượt forward → cost → backward → update, trả về `parameters` mới và `cost` — đây chính là bước đệm để hiểu vòng lặp huấn luyện đầy đủ sẽ dùng ở bài 03/04.
4. **Kiểm chứng nhận định "mạng sâu hơn sẽ tệ hơn":** tự thử `layers_dims = [8, 20, 7, 5, 1]` trong mục Bonus với nhiều learning rate khác nhau (0.01 đến 0.75), xác nhận accuracy không vượt qua baseline — rồi so sánh với kết quả sau khi áp dụng He initialization ở bài 03.

## 9. Tài liệu bổ sung

| Chủ đề | Liên kết tham khảo |
|---|---|
| NumPy broadcasting (nguồn lỗi shape phổ biến nhất) | https://numpy.org/doc/stable/user/basics.broadcasting.html |
| Cross-entropy loss cho phân loại nhị phân | https://en.wikipedia.org/wiki/Cross-entropy |
| Backpropagation (bài giảng gốc deeplearning.ai) | https://www.coursera.org/learn/neural-networks-deep-learning |
| Pima Indians Diabetes Dataset (nguồn gốc) | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |

## 10. Lưu ý khi kiểm tra bài trước khi nộp

- Chạy lại **toàn bộ notebook từ đầu** trên Colab (Runtime → Restart and run all), không chỉ chạy rời rạc từng cell.
- Dữ liệu chính chỉ là mảng số cố định trong `testCases.py`, không phải dữ liệu thật → không có rủi ro data leakage. Riêng mục Bonus (mục 7) dùng dữ liệu Pima Diabetes thật với train/test tách rõ trong `load_dataset()`.
- Sau khi chạy sạch không lỗi, báo lại cho anh Cường / thầy Thiện theo đúng quy trình đã hướng dẫn.
