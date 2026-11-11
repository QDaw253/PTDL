import numpy as np
import pandas as pd

# ==============================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU SẠCH (DATA LOADING)
# ==============================================================================
file_input = "media_all_channels_cleaned.csv"

try:
    df = pd.read_csv(file_input)
    print("========================================================================")
    print("=== 1. ĐỌC DỮ LIỆU SẠCH (MEDIA_ALL_CHANNELS_CLEANED.CSV) ===")
    print("========================================================================")
    print(
        f"-> Kích thước dữ liệu sạch đầu vào: {df.shape[0]} dòng, {df.shape[1]}"
        " cột"
    )
except FileNotFoundError:
    print(
        f"Lỗi: Không tìm thấy file '{file_input}'. Vui lòng chạy file làm"
        " sạch dữ liệu trước!"
    )
    exit()

# Kiểm tra tổng quan phân bố chỉ số sinh lời ROAS
print("\nThống kê mô tả chỉ số sinh lời ROAS:")
print(df["roas"].describe())


# ==============================================================================
# BƯỚC 2: XÂY DỰNG NHÃN MỤC TIÊU DỰA TRÊN ROAS (TARGET LABELING)
# ==============================================================================
print(
    "\n========================================================================"
)
print("=== 2. XÂY DỰNG NHÃN MỤC TIÊU (HIGH_PERFORMANCE_CAMPAIGN) ===")
print(
    "========================================================================"
)

# Lựa chọn giá trị trung vị Median ROAS làm ngưỡng phân loại (Threshold)
# Đảm bảo tỷ lệ nhãn cân bằng tuyệt đối (50% nhãn 1, 50% nhãn 0) để tránh lệch dữ liệu (imbalanced data)
roas_threshold = df["roas"].median()

# Gán nhãn nhị phân: 1 (Chiến dịch hiệu quả cao), 0 (Chiến dịch hiệu quả kém/trung bình)
df["High_Performance_Campaign"] = (df["roas"] >= roas_threshold).astype(int)

print(f"-> Ngưỡng ROAS trung vị làm ranh giới phân loại: {roas_threshold:.4f}")
print("\nPhân bổ tỷ lệ nhãn mục tiêu (Class Balance Audit):")
print(df["High_Performance_Campaign"].value_counts(normalize=True))

print(
    "\n- Tổng số chiến dịch Hiệu quả cao (Nhãn 1):"
    f" {(df['High_Performance_Campaign'] == 1).sum()} chiến dịch"
)
print(
    "- Tổng số chiến dịch Chưa hiệu quả (Nhãn 0):"
    f" {(df['High_Performance_Campaign'] == 0).sum()} chiến dịch"
)


# ==============================================================================
# BƯỚC 3: XUẤT FILE SẠCH ĐÃ GÁN NHÃN (OUTPUT GENERATION)
# ==============================================================================
file_output = "media_all_channels_cleaned_labeled.csv"
df.to_csv(file_output, index=False)

print(
    "\n========================================================================"
)
print("=== KẾT QUẢ CUỐI CÙNG ===")
print(
    "========================================================================"
)
print(f"Kích thước file xuất ra: {df.shape[0]} dòng, {df.shape[1]} cột")
print(
    f" File dữ liệu đã được dán nhãn thành công và lưu tại: '{file_output}'"
)