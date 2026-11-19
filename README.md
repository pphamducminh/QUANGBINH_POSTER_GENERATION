# Quảng Bình Poster Studio

Tạo poster du lịch từ ảnh AI, ảnh tải lên hoặc ảnh mẫu. Bản molab chạy trên GPU miễn phí theo phiên, sử dụng LoRA Quảng Bình đã train. Bạn có thể tải poster dạng PDF và ảnh nền dạng PNG.

## Mở giao diện trên molab

1. Mở [notebook Quảng Bình Poster](https://molab.marimo.io/notebooks/nb_GVhf8vkaa9JQkBPaUDsSem) và đăng nhập molab. Nếu muốn tạo bản riêng, dùng **Import Notebook → Upload File** ở trang chủ molab và chọn `QuangBinh_Molab.py`.
2. Kiểm tra đầu trang có ghi **RTX Pro 6000**. Nếu đang dùng CPU, bấm thông số máy, chọn GPU rồi bấm **Apply and restart**.
3. Thiết lập Hugging Face lần đầu. Đăng nhập, chấp nhận điều kiện truy cập [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), rồi tạo [token Read](https://huggingface.co/settings/tokens) bằng cùng tài khoản. Trong molab, mở **Secrets → +**, lưu token với tên `HF_TOKEN` vào `.env` của notebook. Molab nạp lại secret khi mở phiên mới. Để thử ngay trong phiên đang mở, cũng có thể dùng ô **Kết nối Hugging Face** trên giao diện. [Tài liệu Secrets của molab](https://marimo.io/pages/molab/storage)
4. Nếu chưa thấy giao diện, bấm **Run all** một lần. Notebook tự chuẩn bị thư viện. Khi bạn bấm tạo ảnh lần đầu, ứng dụng tải FLUX.1-dev và [LoRA Quảng Bình](https://huggingface.co/kTon/quangbinh-flux-lora).

## Tạo và tải poster

1. Chọn **Tạo ảnh bằng AI**, chọn địa danh, nhập mô tả rồi bấm **Tạo ảnh nền**. Hoặc chọn **Tải ảnh của bạn** hay **Thử với ảnh mẫu** để dựng poster ngay.
2. Chọn bố cục và sửa tiêu đề, dòng giới thiệu, địa điểm. Bản xem trước cập nhật từ ảnh đang dùng.
3. Bấm **Tải poster PDF** để lấy bản có chữ và bố cục, hoặc **Tải ảnh nền PNG** để lấy ảnh trước khi thêm chữ.

Molab giới hạn phiên tối đa 12 giờ và tự dừng sau 90 phút không hoạt động. Khi quay lại, nếu phải mở phiên mới thì bấm **Run all** một lần. Lần tạo ảnh đầu sau khi mở phiên có thể lâu hơn do phải nạp model. [Giới hạn phiên molab](https://docs.marimo.io/guides/molab/#compute)

Nếu gặp lỗi **401/403**, kiểm tra token có quyền đọc và tài khoản đã được cấp quyền truy cập FLUX.1-dev. Nếu giao diện báo thiếu CUDA, kiểm tra mục GPU ở đầu trang.

## Đưa mã lên GitHub

Tạo một repository trống trên GitHub và chọn chế độ hiển thị phù hợp. Từ thư mục dự án, chạy các lệnh dưới đây, sau đó thay `TAI_KHOAN` và `TEN_REPO` bằng tên thật. Danh sách `git add` chỉ gồm bản molab và mã cần để dựng lại bản đó.

```bash
git init -b main
git add README.md .gitignore QuangBinh_Molab.py app_ui.py core_model.py \
  model_selection.py endpoint_client.py template_manager.py templates/ \
  scripts/build_molab.py poster_quang_phu_final.png
git diff --cached --stat
git commit -m "Add Quang Binh Poster Studio for molab"
git remote add origin https://github.com/TAI_KHOAN/TEN_REPO.git
git push -u origin main
```

Giữ token trong molab Secrets và trọng số LoRA trên Hugging Face. `.gitignore` loại tệp cấu hình cục bộ, dữ liệu train, trọng số, ZIP, PDF đầu ra và `notebook.py` cũ có thông tin đăng nhập W&B ghi trực tiếp. Nếu key cũ còn hiệu lực, hãy thu hồi nó trước khi chia sẻ notebook train. GitHub chặn tệp lớn hơn 100 MiB qua Git thông thường. [Giới hạn tệp GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

Để nối một repo công khai với molab, trên trang chủ molab chọn **Mirror from GitHub** và dán URL GitHub của tệp `QuangBinh_Molab.py`, ví dụ `https://github.com/TAI_KHOAN/TEN_REPO/blob/main/QuangBinh_Molab.py`. Để chạy AI, mở bản chạy trên máy chủ hoặc fork vào workspace của bạn để cấu hình GPU và Secrets. Nếu molab không truy cập được repo riêng tư, nhập trực tiếp tệp `.py` bằng **Import Notebook**. [Tài liệu Mirror from GitHub](https://docs.marimo.io/guides/molab/#mirror-notebooks-from-github)

Sau khi sửa giao diện, code model hoặc mẫu poster, chạy `python3 scripts/build_molab.py` để cập nhật tệp molab rồi commit cả mã nguồn và tệp mới.
