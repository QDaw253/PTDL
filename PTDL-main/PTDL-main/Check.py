import numpy as np
import pandas as pd

# 1. Đọc tệp dữ liệu
df = pd.read_csv("media_all_channels_dirty_dataset.csv")

print("=== 1. TỔNG QUAN DỮ LIỆU ===")
print(df.info())

print("\n=== 2. KIỂM TRA GIÁ TRỊ THIẾU (NULL/NAN) ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n=== 3. KIỂM TRA BẢN GHI TRÙNG LẬP ===")
print(f"Số lượng dòng trùng lặp hoàn toàn: {df.duplicated().sum()}")

print("\n=== 4. KIỂM TRA BẤT THƯỜNG Ở DỮ LIỆU DẠNG CHUỖI (CATEGORICAL) ===")
print("Các giá trị duy nhất trong cột 'channel':")
print(df["channel"].dropna().unique())

print("\n=== 5. KIỂM TRA LỖI LOGIC & GIÁ TRỊ BẤT THƯỜNG (OUTLIERS) ===")
# Số âm ở cột clicks
print(f"Số dòng có clicks < 0: {(df['clicks'] < 0).sum()}")
# Lỗi logic: clicks > impressions
print(
    "Số dòng có clicks > impressions:"
    f" {((df['clicks'] > df['impressions']) & df['impressions'].notnull()).sum()}"
)
# Outlier ở cột cost (kiểm tra các giá trị vượt quá 3 lần độ lệch chuẩn hoặc quá cao)
print(f"Giá trị cost lớn nhất: {df['cost'].max():,.2f}")
print(f"Trung vị của cost: {df['cost'].median():,.2f}")

print("\n=== 6. KIỂM TRA KIỂU DỮ LIỆU NGHỊCH LÝ (ĐỊNH DẠNG NGÀY THÁNG) ===")
# Kiểm tra các mẫu ngày không theo chuẩn YYYY-MM-DD
non_standard_dates = df[df["date"].astype(str).str.contains("/")]
print(f"Số lượng dòng ngày tháng sai chuẩn YYYY-MM-DD: {len(non_standard_dates)}")

# Tạo bản sao để tiến hành làm sạch
df_clean = df.copy()

# ---------------------------------------------------------
# 1. Xử lý bản ghi trùng lặp (Duplicates)
# ---------------------------------------------------------
df_clean = df_clean.drop_duplicates()

# ---------------------------------------------------------
# 2. Chuẩn hóa dữ liệu dạng chuỗi (Text Normalization)
# ---------------------------------------------------------
# Xóa khoảng trắng thừa hai đầu và chuẩn hóa tên kênh quảng cáo
df_clean["channel"] = df_clean["channel"].astype(str).str.strip()
channel_mapping = {
    "facebook": "Facebook",
    "Google_Ads": "Google Ads",
    "Tik Tok": "TikTok",
    "nan": np.nan,  # Giữ nguyên NaN để xử lý ở bước thiếu dữ liệu
}
df_clean["channel"] = df_clean["channel"].replace(channel_mapping)

# ---------------------------------------------------------
# 3. Chuẩn hóa định dạng ngày tháng (Datetime Inconsistency)
# ---------------------------------------------------------
# Chuyển tất cả về kiểu datetime chuẩn YYYY-MM-DD
df_clean["date"] = pd.to_datetime(df_clean["date"], format="mixed").dt.strftime(
    "%Y-%m-%d"
)

# ---------------------------------------------------------
# 4. Xử lý giá trị âm & Lỗi logic (Anomalies & Logic Errors)
# ---------------------------------------------------------
# Chuyển các giá trị clicks bị âm thành giá trị tuyệt đối
df_clean["clicks"] = df_clean["clicks"].abs()

# Xử lý lỗi logic: Clicks > Impressions -> gán Clicks = Impressions (hoặc drop dòng tùy nghiệp vụ)
invalid_clicks_mask = df_clean["clicks"] > df_clean["impressions"]
df_clean.loc[invalid_clicks_mask, "clicks"] = df_clean.loc[
    invalid_clicks_mask, "impressions"
]

# ---------------------------------------------------------
# 5. Xử lý giá trị ngoại lệ cực đại (Outliers)
# ---------------------------------------------------------
# Thay thế các giá trị cost tăng đột biến (> 50.000) bằng trung vị của kênh đó
cost_threshold = 50000
outlier_mask = df_clean["cost"] > cost_threshold

# Điền giá trị ngoai lệ bằng median
median_cost = df_clean[~outlier_mask]["cost"].median()
df_clean.loc[outlier_mask, "cost"] = median_cost

# ---------------------------------------------------------
# 6. Xử lý giá trị thiếu (Missing Values / Imputation)
# ---------------------------------------------------------
# Cột channel: Điền bằng 'Unknown' hoặc mode (giá trị xuất hiện nhiều nhất)
df_clean["channel"] = df_clean["channel"].fillna("Unknown")

# Cột impressions và cost: Điền bằng trung vị (Median) để tránh ảnh hưởng của ngoại lệ
df_clean["impressions"] = df_clean["impressions"].fillna(
    df_clean["impressions"].median()
)
df_clean["cost"] = df_clean["cost"].fillna(df_clean["cost"].median())

# ---------------------------------------------------------
# 7. Tính toán lại các chỉ số phái sinh (Recalculate Derived Metrics)
# ---------------------------------------------------------
# Do dữ liệu thay đổi (clicks, cost, impressions), cần tính lại CPC, CTR, ROAS, ROI...
df_clean["cpc"] = np.where(
    df_clean["clicks"] > 0, df_clean["cost"] / df_clean["clicks"], 0
)
df_clean["ctr"] = np.where(
    df_clean["impressions"] > 0,
    df_clean["clicks"] / df_clean["impressions"],
    0,
)
df_clean["roas"] = np.where(
    df_clean["cost"] > 0, df_clean["revenue"] / df_clean["cost"], 0
)
df_clean["roi"] = df_clean["roas"] - 1

# ---------------------------------------------------------
# 8. Xuất file dữ liệu sạch
# ---------------------------------------------------------
df_clean.to_csv("media_all_channels_cleaned.csv", index=False)
print("-> Đã xử lý xong toàn bộ dữ liệu và lưu vào tệp media_all_channels_cleaned.csv!")