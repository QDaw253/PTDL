import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

# 1. ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ
df = pd.read_csv('media_all_channels_cleaned.csv')
df['profit'] = df['revenue'] - df['cost']

# ==============================================================================
# PHẦN 1: PHÂN TÍCH TỔNG QUAN TOÀN BỘ TỆP DỮ LIỆU
# ==============================================================================
total_cost = df['cost'].sum()
total_revenue = df['revenue'].sum()
total_profit = df['profit'].sum()
total_impressions = df['impressions'].sum()
total_clicks = df['clicks'].sum()
total_conversions = df['conversions'].sum()

overall_roas = total_revenue / total_cost
overall_roi = total_profit / total_cost
overall_cpc = total_cost / total_clicks
overall_cpa = total_cost / total_conversions
overall_ctr = total_clicks / total_impressions
overall_cr = total_conversions / total_clicks

print("=== 1. TỔNG QUAN TOÀN BỘ TỆP DỮ LIỆU ===")
print(f"- Tổng Chi phí (Cost):       {total_cost:,.2f}")
print(f"- Tổng Doanh thu (Revenue):  {total_revenue:,.2f}")
print(f"- Tổng Lợi nhuận (Profit):   {total_profit:,.2f}")
print(f"- ROAS tổng thể: {overall_roas:.4f} | ROI tổng thể: {overall_roi:.4f}")
print(f"- CPC: {overall_cpc:.4f} | CPA: {overall_cpa:.4f} | CTR: {overall_ctr:.4f} | CR: {overall_cr:.4f}\n")

# ==============================================================================
# PHẦN 2: PHÂN TÍCH SO SÁNH THEO TỪNG KÊNH (CHANNEL)
# ==============================================================================
channel_df = df.groupby('channel').agg({
    'cost': 'sum',
    'revenue': 'sum',
    'profit': 'sum',
    'impressions': 'sum',
    'clicks': 'sum',
    'conversions': 'sum'
}).reset_index()

channel_df['roas'] = channel_df['revenue'] / channel_df['cost']
channel_df['roi'] = channel_df['profit'] / channel_df['cost']
channel_df['cpc'] = channel_df['cost'] / channel_df['clicks']
channel_df['cpa'] = channel_df['cost'] / channel_df['conversions']
channel_df['ctr'] = channel_df['clicks'] / channel_df['impressions']
channel_df['cr'] = channel_df['conversions'] / channel_df['clicks']
channel_df = channel_df.sort_values(by='roas', ascending=False)

print("=== 2. BẢNG SO SÁNH HIỆU QUẢ THEO KÊNH ===")
print(channel_df[['channel', 'cost', 'revenue', 'profit', 'roas', 'roi', 'cpa', 'cpc']].to_string(index=False))
print("\n")

# ==============================================================================
# PHẦN 3: PHÂN TÍCH CHI TIẾT KÊNH + CHIẾN DỊCH & GÁN NHÃN ROAS
# ==============================================================================
cc_df = df.groupby(['channel', 'campaign']).agg({
    'cost': 'sum',
    'revenue': 'sum',
    'profit': 'sum',
    'impressions': 'sum',
    'clicks': 'sum',
    'conversions': 'sum'
}).reset_index()

cc_df['roas'] = cc_df['revenue'] / cc_df['cost']
cc_df['roi'] = cc_df['profit'] / cc_df['cost']
cc_df['cpc'] = cc_df['cost'] / cc_df['clicks']
cc_df['cpa'] = cc_df['cost'] / cc_df['conversions']
cc_df['ctr'] = cc_df['clicks'] / cc_df['impressions']
cc_df['conversion_rate'] = cc_df['conversions'] / cc_df['clicks']

# Gán nhãn dựa trên Trung vị (Median) của ROAS
roas_median = cc_df['roas'].median()
cc_df['is_optimized'] = (cc_df['roas'] > roas_median).astype(int)
cc_df['status'] = cc_df['is_optimized'].map({1: 'Tối ưu', 0: 'Chưa tối ưu'})

print(f"=== 3. CHI TIẾT KÊNH + CHIẾN DỊCH (Ngưỡng Median ROAS = {roas_median:.4f}) ===")
print(cc_df[['channel', 'campaign', 'cost', 'revenue', 'roas', 'cpa', 'status']].sort_values(by='roas', ascending=False).to_string(index=False))
print("\n")

# ==============================================================================
# PHẦN 4: HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING (CHU KỲ TUẦN & CHỐNG DATA LEAKAGE)
# ==============================================================================
# Gom dữ liệu theo chu kỳ Tuần để triệt tiêu nhiễu độ trễ chuyển đổi theo ngày
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.to_period('W').astype(str)

weekly_df = df.groupby(['week', 'channel', 'campaign']).agg({
    'cost': 'sum',
    'impressions': 'sum',
    'clicks': 'sum',
    'revenue': 'sum'
}).reset_index()

weekly_df['roas'] = weekly_df['revenue'] / weekly_df['cost']
weekly_df['cpc'] = weekly_df['cost'] / weekly_df['clicks']
weekly_df['ctr'] = weekly_df['clicks'] / weekly_df['impressions']
weekly_df['cpm'] = (weekly_df['cost'] / weekly_df['impressions']) * 1000
weekly_df['is_optimized'] = (weekly_df['roas'] > weekly_df['roas'].median()).astype(int)

# Chỉ dùng các biến đầu vào & tương tác (loại bỏ conversions, cpa, conversion_rate để tránh rò rỉ dữ liệu)
num_features = ['cost', 'impressions', 'clicks', 'cpc', 'ctr', 'cpm']
cat_features = ['channel', 'campaign']
features = cat_features + num_features

X = weekly_df[features]
y = weekly_df['is_optimized']

# Chia tập Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tiền xử lý: Chuẩn hóa biến số + One-Hot Encoding biến phân loại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

# Xây dựng Pipeline với Random Forest Classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== 4. ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH (DỮ LIỆU TUẦN) ===")
print(f"- Accuracy trên tập Test: {accuracy_score(y_test, y_pred):.4f}")
print(f"- Chỉ số ROC-AUC:        {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}\n")

# Tính trung bình theo tuần của 42 cặp Kênh + Chiến dịch để đưa vào mô hình dự đoán xác suất tối ưu
cc_weekly_avg = weekly_df.groupby(['channel', 'campaign'])[num_features].mean().reset_index()
cc_df['opt_prob'] = model.predict_proba(cc_weekly_avg[features])[:, 1]

# Quy tắc đề xuất tăng/giảm ngân sách kết hợp giữa Nhãn thực tế & Xác suất mô hình
def recommend_budget(row):
    if row['opt_prob'] >= 0.70 and row['is_optimized'] == 1:
        return 'Tăng ngân sách (+30%)'
    elif row['opt_prob'] < 0.40 and row['is_optimized'] == 0:
        return 'Giảm ngân sách (-30%)'
    else:
        return 'Giữ nguyên / Tối ưu nội dung'

cc_df['recommendation'] = cc_df.apply(recommend_budget, axis=1)

print("=== 5. BẢNG ĐỀ XUẤT PHÂN BỔ NGÂN SÁCH TỪ MÔ HÌNH ===")
print(cc_df[['channel', 'campaign', 'roas', 'opt_prob', 'status', 'recommendation']].sort_values(by='opt_prob', ascending=False).to_string(index=False))

# ==============================================================================
# PHẦN 5: TRỰC QUAN HÓA KẾT QUẢ PHÂN TÍCH
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: So sánh Chi phí và Doanh thu theo Kênh
x_pos = np.arange(len(channel_df))
width = 0.35
axes[0].bar(x_pos - width/2, channel_df['cost']/1e6, width, label='Chi phí (Triệu USD)', color='#e74c3c')
axes[0].bar(x_pos + width/2, channel_df['revenue']/1e6, width, label='Doanh thu (Triệu USD)', color='#2ecc71')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(channel_df['channel'], rotation=25)
axes[0].set_title('So sánh Chi phí và Doanh thu theo Kênh')
axes[0].set_ylabel('Triệu USD')
axes[0].legend()

# Biểu đồ 2: Heatmap ROAS theo Kênh và Chiến dịch
pivot_roas = cc_df.pivot(index='channel', columns='campaign', values='roas')
sns.heatmap(pivot_roas, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[1])
axes[1].set_title('Ma trận ROAS theo Kênh và Chiến dịch')
axes[1].set_xlabel('Chiến dịch')
axes[1].set_ylabel('Kênh')

plt.tight_layout()
plt.savefig('phan_tich_quang_cao2.png', dpi=150)
plt.show()