import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. Đọc dữ liệu và tạo đặc trưng
df = pd.read_csv('media_all_channels_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['cpm'] = (df['cost'] / df['impressions']) * 1000

# 2. Gán nhãn tối ưu hóa (vùng đệm 40% - 60%) và chia Train/Test 80/20
q_low, q_high = df['roas'].quantile(0.40), df['roas'].quantile(0.60)
df_filtered = df[(df['roas'] <= q_low) | (df['roas'] >= q_high)].copy()
df_filtered['is_optimized'] = (df_filtered['roas'] >= q_high).astype(int)

num_cols = ['cost', 'impressions', 'clicks', 'conversions', 'cpc', 'cpa', 'ctr', 'conversion_rate', 'cpm', 'day_of_week', 'month', 'is_weekend']
cat_cols = ['channel', 'campaign']

X = df_filtered[num_cols + cat_cols]
y = df_filtered['is_optimized']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

# Huấn luyện trên tập Train và đánh giá trên tập Test
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (80/20) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Chưa tối ưu (0)', 'Tối ưu (1)']))

# 3. Tổng hợp 42 cặp Kênh + Chiến dịch & dự đoán xác suất tối ưu
cc_df = df.groupby(['channel', 'campaign']).agg({
    'cost': 'sum',
    'revenue': 'sum'
}).reset_index()
cc_df['roas'] = cc_df['revenue'] / cc_df['cost']

cc_daily = df.groupby(['channel', 'campaign'])[num_cols].mean().reset_index()
cc_df['opt_prob'] = model.predict_proba(cc_daily[num_cols + cat_cols])[:, 1]

# 4. Mô phỏng Phân bổ Ngân sách GIỮ NGUYÊN TỔNG VỐN BAN ĐẦU
total_initial_cost = cc_df['cost'].sum()
total_initial_rev = cc_df['revenue'].sum()
total_initial_profit = total_initial_rev - total_initial_cost

def get_weight_multiplier(prob):
    if prob >= 0.70:
        return 1.30  # Tăng 30% trọng số vốn
    elif prob < 0.40:
        return 0.70  # Giảm 30% trọng số vốn
    return 1.00

cc_df['raw_new_cost'] = cc_df['cost'] * cc_df['opt_prob'].apply(get_weight_multiplier)
cc_df['new_cost'] = cc_df['raw_new_cost'] * (total_initial_cost / cc_df['raw_new_cost'].sum())
cc_df['new_revenue'] = cc_df['new_cost'] * cc_df['roas']

total_new_rev = cc_df['new_revenue'].sum()
total_new_profit = total_new_rev - total_initial_cost
rev_gain = total_new_rev - total_initial_rev
rev_gain_pct = (rev_gain / total_initial_rev) * 100
profit_gain_pct = (rev_gain / total_initial_profit) * 100

print("=== KẾT QUẢ MÔ PHỎNG TỐI ƯU HÓA DOANH THU & LỢI NHUẬN ===")
print(f"Tổng Ngân sách Ban đầu:    {total_initial_cost:,.2f}")
print(f"Tổng Ngân sách Mới:        {cc_df['new_cost'].sum():,.2f}")
print(f"Doanh thu Ban đầu:         {total_initial_rev:,.2f}")
print(f"Doanh thu Dự kiến Mới:     {total_new_rev:,.2f} (+{rev_gain:,.2f} | +{rev_gain_pct:.2f}%)")
print(f"Lợi nhuận Dự kiến Mới:     {total_new_profit:,.2f} (+{rev_gain:,.2f} | +{profit_gain_pct:.2f}%)")

# 5. Vẽ biểu đồ trực quan Before vs After
ch_comp = cc_df.groupby('channel')[['cost', 'new_cost', 'revenue', 'new_revenue']].sum().reset_index()

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(ch_comp))
w = 0.35
axes[0].bar(x - w/2, ch_comp['cost']/1e6, w, label='Ngân sách Ban đầu', color='#95a5a6')
axes[0].bar(x + w/2, ch_comp['new_cost']/1e6, w, label='Ngân sách Tối ưu Mới', color='#3498db')
axes[0].set_xticks(x)
axes[0].set_xticklabels(ch_comp['channel'], rotation=25)
axes[0].set_title('Dịch chuyển Ngân sách theo Kênh (Cùng Tổng Vốn)')
axes[0].set_ylabel('Triệu USD')
axes[0].legend()

axes[1].bar(['Trước Tối Ưu', 'Sau Phân Bổ Ngân Sách'], [total_initial_rev/1e6, total_new_rev/1e6], color=['#e74c3c', '#2ecc71'], width=0.5)
axes[1].set_title(f'Tổng Doanh Thu: Tăng thêm +{rev_gain/1e6:.2f} Triệu USD (+{rev_gain_pct:.2f}%)')
axes[1].set_ylabel('Triệu USD')
for i, v in enumerate([total_initial_rev/1e6, total_new_rev/1e6]):
    axes[1].text(i, v + 10, f"{v:,.2f}M", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('budget_reallocation_simulation2.png', dpi=150)
plt.show()