# Mã nguồn thực hành - Module 1 (Deep Learning)

Đây là **phiên bản project** (nhiều file `.py`, hàm đóng gói vào `utils.py`/`model.py`) của 5 bài thực hành trong `thực hành/`. Bản này chạy trực tiếp bằng Python trên máy Lab, không cần Jupyter/Colab.

Mỗi bài `0X_.../` trong đây là bản chuyển thể trung thực từ notebook cùng tên trong `../thực hành/0X_.../` — cùng công thức, cùng random seed, cùng thứ tự chạy — nên kết quả in ra khớp với phần "Expected output"/"Kết quả mong đợi" trong tài liệu hướng dẫn tương ứng (`../tài liệu thực hành/0X_.../README.md`).

## Cài đặt (chạy 1 lần)

```bash
cd "source"
uv sync
```

## Chạy từng bài

```bash
cd 01_Kien-truc-DNN-va-Backpropagation
uv run --project .. python main.py
```

Thay `01_...` bằng thư mục bài tương ứng (`02_...` đến `05_...`). Mỗi `main.py` chạy tuần tự toàn bộ các exercise của bài, in kết quả ra console (và mở cửa sổ `matplotlib` cho các bài có vẽ đồ thị: 03, 04, 05).

## Cấu trúc mỗi bài

- `utils.py` — các hàm đã được cung cấp sẵn trong notebook gốc (activation, load dataset, plot...), sinh viên không cần tự viết.
- `test_cases.py` (nếu có) — input cố định dùng để kiểm thử/demo từng hàm, y hệt `testCases.py` trong bản notebook.
- `model.py` — các hàm bài tập (exercise) chính của bài, viết theo đúng thứ tự trong notebook.
- `main.py` — điểm chạy chính, gọi lần lượt từng hàm và in kết quả, tương đương chạy hết notebook từ trên xuống.

## Lưu ý

- Bản này ưu tiên khớp 1-1 với notebook để dễ đối chiếu khi tự học, nên giữ nguyên `print()` thay vì dùng logging framework.
- `public_tests.py` gốc của Coursera (lab 02, 03, 04) phụ thuộc gói nội bộ `dlai_tools` không cài được ngoài môi trường Coursera — chính notebook gốc cũng đã comment-out các lời gọi này, nên bản project không mang theo các file đó (trừ lab 01, nơi `public_tests.py` hoạt động độc lập, không cần `dlai_tools`).
