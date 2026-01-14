import gdown
import os

# Thư mục lưu file
save_dir = "/workspace/khanh/SpaMo/features/vit_feat_Phoenix14T"
os.makedirs(save_dir, exist_ok=True)

# File ID từ link Google Drive mới
file_id = "1EbEagkjbUssEFJLen1wYqnfN1fmHcL_3"

# Đặt tên file tải về (đuôi .ckpt)
output_path = os.path.join(save_dir, "mo_feat_p14t.zip")

# URL tải trực tiếp
url = f"https://drive.google.com/uc?id={file_id}"

# Tải file
gdown.download(url, output_path, quiet=False)

print(f"✅ File đã được tải về: {output_path}")
