import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Đọc dữ liệu và tổng hợp theo chu kỳ Tuần (đồng bộ với analyticalOverview.py)
df = pd.read_csv('media_all_channels_cleaned.csv')
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

# 2. Huấn luyện mô hình phân loại trên dữ liệu Tuần (Đồng bộ 100% với analyticalOverview.py)
num_cols = ['cost', 'impressions', 'clicks', 'cpc', 'ctr', 'cpm']
cat_cols = ['channel', 'campaign']
features = cat_cols + num_cols

X_train, X_test, y_train, y_test = train_test_split(
    weekly_df[features], weekly_df['is_optimized'], test_size=0.2, random_state=42, stratify=weekly_df['is_optimized']
)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42))
])
model.fit(X_train, y_train)

# 3. Tổng hợp 42 cặp Kênh + Chiến dịch & dự đoán xác suất tối ưu
cc_df = df.groupby(['channel', 'campaign']).agg({
    'cost': 'sum',
    'revenue': 'sum'
}).reset_index()
cc_df['roas'] = cc_df['revenue'] / cc_df['cost']
cc_df['is_optimized'] = (cc_df['roas'] > cc_df['roas'].median()).astype(int)

cc_weekly = weekly_df.groupby(['channel', 'campaign'])[num_cols].mean().reset_index()
cc_df['opt_prob'] = model.predict_proba(cc_weekly[features])[:, 1]

# 4. Mô phỏng Phân bổ Ngân sách GIỮ NGUYÊN TỔNG VỐN (Đồng bộ quy tắc kép & Hiệu suất biên giảm dần)
total_initial_cost = cc_df['cost'].sum()
total_initial_rev = cc_df['revenue'].sum()
total_initial_profit = total_initial_rev - total_initial_cost

def get_weight_multiplier(row):
    if row['opt_prob'] >= 0.70 and row['is_optimized'] == 1:
        return 1.30  # Tăng 30% trọng số vốn
    elif row['opt_prob'] < 0.40 and row['is_optimized'] == 0:
        return 0.70  # Giảm 30% trọng số vốn
    return 1.00

cc_df['raw_new_cost'] = cc_df['cost'] * cc_df.apply(get_weight_multiplier, axis=1)

# Chuẩn hóa lại để Tổng Ngân sách Mới == Tổng Ngân sách Ban đầu
cc_df['new_cost'] = cc_df['raw_new_cost'] * (total_initial_cost / cc_df['raw_new_cost'].sum())

# Điều chỉnh ROAS mới theo quy luật hiệu suất biên giảm dần (Diminishing Returns, alpha = 0.15)
alpha = 0.15
cc_df['adjusted_roas'] = cc_df['roas'] * ((cc_df['cost'] / cc_df['new_cost']) ** alpha)

# Tính Doanh thu dự kiến mới dựa trên ROAS đã điều chỉnh
cc_df['new_revenue'] = cc_df['new_cost'] * cc_df['adjusted_roas']

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
plt.savefig('budget_reallocation_simulation.png', dpi=150)
plt.show()