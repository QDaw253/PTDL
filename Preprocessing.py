import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. ĐỌC DỮ LIỆU & KIỂM TRA MỨC ĐỘ NGUYÊN BẢN (DATA AUDIT)
# ---------------------------------------------------------
df = pd.read_csv("media_all_channels_dt.csv")

print("==================================================")
print("=== 1. KHÁM PHÁ & ĐẾM SỐ LƯỢNG LỖI TRONG DỮ LIỆU ===")
print("==================================================")
print(f"Tổng số dòng dữ liệu thô: {len(df)} dòng")

# 1.1 Đếm số dòng bị trùng lặp (Duplicates)
count_duplicates = df.duplicated().sum()#duplicated là kiểm tra trùng lắp
print(f"-> [Lỗi 1] Số dòng bị trùng lặp hoàn toàn: {count_duplicates} dòng")

# 1.2 Đếm số dòng bị khuyết dữ liệu (Missing values)
count_missing_rows = df.isnull().any(axis=1).sum() #.any(axis=1) nếu có ít nhất 1 giá trị NaN trong dòng thì trả về True
print(
    f"-> [Lỗi 2] Số dòng bị thiếu dữ liệu (Missing/NaN): {count_missing_rows} dòng"
)

# 1.3 Đếm số dòng bị nhiễu / sai số (Noisy Data & Outliers)
mask_noisy = (
    (df["cost"] < 0)
    | (df["cpc"] < 0)
    | (df["conversions"] < 0)
    | (df["clicks"] > 500000)
) # kiểm tra các điều kiện dữ liệu nhiễu: cost, cpc, conversions âm hoặc clicks > 500000
count_noisy_rows = mask_noisy.sum()
print(
    f"-> [Lỗi 3] Số dòng bị nhiễu / sai số (âm, click phóng đại): {count_noisy_rows} dòng"
)

# 1.4 Đếm số dòng sai định dạng / không nhất quán (Inconsistent Data)
mask_inc_date = df["date"].astype(str).str.contains("/")# kiểm tra định dạng ngày tháng có chứa dấu "/"
mask_inc_channel = df["channel"].astype(str).str.contains(r"^[a-z]|^\s+|\s+$", regex=True)# ^\s+ bắt khoảng trắng ở đầu và \s+$ bắt khoảng trắng ở cuối
mask_inc_campaign = df["campaign"].astype(str).str.contains("_|[A-Z]{4,}", regex=True)# kiểm tra định dạng campaign có chứa dấu "_" hoặc có 4 chữ cái in hoa liên tiếp
mask_inconsistent = mask_inc_date | mask_inc_channel | mask_inc_campaign
count_inconsistent_rows = mask_inconsistent.sum()
print(
    f"-> [Lỗi 4] Số dòng sai định dạng văn bản / ngày tháng: {count_inconsistent_rows} dòng"
)

# 1.5 Tổng số dòng dính ít nhất 1 lỗi
mask_total_bad = (
    df.duplicated() | df.isnull().any(axis=1) | mask_noisy | mask_inconsistent
)
print(
    f"\n=> TỔNG SỐ DÒNG BỊ LỖI THỰC TẾ: {mask_total_bad.sum()} dòng (trên tổng {len(df)} dòng)"
)


# ---------------------------------------------------------
# 2. TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# ---------------------------------------------------------
print("\n==================================================")
print("=== 2. THỰC HIỆN TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU ===")
print("==================================================")

df_clean = df.copy()# tạo bản sao thay vì gán tham chiếu để tránh việc sửa 1 dung liệu sẽ ảnh hưởng đến dữ liệu gốc

# Bước 1: Xóa bản ghi trùng lặp (Duplicate Data)
df_clean = df_clean.drop_duplicates()
print(f" Đã loại bỏ {count_duplicates} dòng trùng lặp.")

# Bước 2: Chuẩn hóa dữ liệu văn bản & ngày tháng (Inconsistent Data)
# 2.1 Đưa định dạng ngày về chuẩn YYYY-MM-DD
df_clean["date"] = pd.to_datetime(df_clean["date"], format="mixed").dt.strftime("%Y-%m-%d")
# định dạng chuỗi ở date thành dữ liệu ngày tháng với format mixed (tự động nhận dạng) và sau đó chuyển về định dạng chuẩn YYYY-MM-DD

# 2.2 Đưa tên kênh tiếp thị về viết hoa đầu từ và xóa khoảng trắng thừa
df_clean["channel"] = df_clean["channel"].astype(str).str.strip().str.title()# thực hiện xóa khoảng trắng 2 đầu và đưa về định dạng title case (viết hoa chữ cái đầu tiên của mỗi từ)
channel_map = {
    "Facebook": "Facebook",
    "Google Ads": "Google Ads",
    "Instagram": "Instagram",
    "Linkedin": "LinkedIn",
    "Twitter": "Twitter",
    "Website": "Website",
}
# tạo một bộ từ điển ánh xạ để chuẩn hóa tên kênh tiếp thị, ví dụ "Linkedin" sẽ được chuẩn hóa thành "LinkedIn"
df_clean["channel"] = df_clean["channel"].replace(channel_map)
# replace() sẽ thay thế các giá trị trong cột channel dựa trên ánh xạ trong channel_map

# 2.3 Chuẩn hóa tên chiến dịch (bỏ gạch dưới, định dạng Title)
df_clean["campaign"] = (
    df_clean["campaign"].astype(str).str.strip().str.title().str.replace("_", " ") # xóa bỏ khoảng trắng, viết hoa chữ cái đầu và thay thế những chuỗi có "_" bằng khoảng trắng
)
print(
    f" Đã chuẩn hóa {count_inconsistent_rows} dòng sai định dạng ngày tháng & chữ viết."
)

# Bước 3: Xử lý dữ liệu nhiễu & Ngoại lai (Noisy Data & Outliers)
# Chuyển các giá trị âm bất hợp lý về số dương
num_cols_positive = ["cost", "cpc", "conversions", "clicks", "impressions"]
for col in num_cols_positive:
    df_clean[col] = df_clean[col].abs()# abs tức trị tuyệt đối chuyển số âm thành dương

# Thu nhỏ các giá trị clicks bị ngoại lai phóng đại 50 lần
df_clean.loc[df_clean["clicks"] > 500000, "clicks"] = (
    df_clean.loc[df_clean["clicks"] > 500000, "clicks"] // 50
)# kiểm tra nếu clicks > 500000 thì chia cho 50 lấy nguyên để giảm giá trị ngoại lai
print(f" Đã xử lý {count_noisy_rows} dòng chứa số âm và giá trị ngoại lai.")

# Bước 4: Xử lý dữ liệu thiếu (Missing Data Imputation)
# Điền các giá trị NaN ở các cột chỉ số tiếp thị bằng giá trị Trung vị (Median)
metrics_cols = ["impressions", "clicks", "conversions", "cost", "revenue"]
for col in metrics_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())# tìm các ô đang trông nan và điền bằng giá trị trung vị của cột đó
print(
    f" Đã điền giá trị trung vị (Median) cho {count_missing_rows} dòng khuyết dữ liệu."
)

# Bước 5: Tính toán lại đồng bộ các chỉ số phái sinh (Derived Metrics)
df_clean["cpc"] = np.where(
    df_clean["clicks"] > 0, df_clean["cost"] / df_clean["clicks"], 0
) # np.where(điều kiện, giá trị nếu đúng, giá trị nếu sai) để tránh chia cho 0
df_clean["cpa"] = np.where(
    df_clean["conversions"] > 0, df_clean["cost"] / df_clean["conversions"], 0
)
df_clean["ctr"] = np.where(
    df_clean["impressions"] > 0, df_clean["clicks"] / df_clean["impressions"], 0
)
df_clean["conversion_rate"] = np.where(
    df_clean["clicks"] > 0, df_clean["conversions"] / df_clean["clicks"], 0
)
df_clean["roas"] = np.where(
    df_clean["cost"] > 0, df_clean["revenue"] / df_clean["cost"], 0
)
df_clean["roi"] = df_clean["roas"] - 1
df_clean["profit_margin"] = np.where(
    df_clean["revenue"] > 0,
    (df_clean["revenue"] - df_clean["cost"]) / df_clean["revenue"],
    0,
)
print(" Đã tái tính toán lại toàn bộ các chỉ số phái sinh (CPC, CPA, CTR, ROAS, ROI...).")


# ---------------------------------------------------------
# 3. KIỂM TRA KẾT QUẢ VÀ XUẤT TỆP CLEANED
# ---------------------------------------------------------
output_file = "media_all_channels_cleaned.csv"
df_clean.to_csv(output_file, index=False)

print("\n==================================================")
print("=== 3. KẾT QUẢ SAU KHI LÀM SẠCH ===")
print("==================================================")
print(f"Kích thước file sạch: {df_clean.shape[0]} dòng, {df_clean.shape[1]} cột")
print(f"Tổng số giá trị khuyết còn lại: {df_clean.isnull().sum().sum()}")
print(f"Tổng số dòng trùng lặp còn lại: {df_clean.duplicated().sum()}")
print(f"-> Đã xuất tệp dữ liệu sạch thành công tại '{"media_all_channels_cleaned.csv"}'!")