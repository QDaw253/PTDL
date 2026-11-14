import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid")

# 1. Đọc và tổng hợp dữ liệu theo chu kỳ Tuần
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
weekly_df['label_name'] = weekly_df['is_optimized'].map({0: 'Chưa tối ưu (0)', 1: 'Tối ưu (1)'})

# ==============================================================================
# VẼ PHẦN 1: BOXPLOT SO SÁNH PHÂN PHỐI CPC, CPM, COST GIỮA 2 NHÓM NHÃN
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [
    ('cpc', 'Chi phí mỗi lượt nhấp - CPC (USD)', False),
    ('cpm', 'Chi phí 1.000 hiển thị - CPM (USD)', False),
    ('cost', 'Ngân sách chi tiêu tuần - Cost (Nghìn USD)', True)
]

for ax, (col, title, div_k) in zip(axes, metrics):
    plot_data = weekly_df.copy()
    if div_k:
        plot_data[col] = plot_data[col] / 1000.0
    sns.boxplot(
        data=plot_data, x='label_name', y=col,
        order=['Chưa tối ưu (0)', 'Tối ưu (1)'],
        palette=['#e74c3c', '#2ecc71'], width=0.45, ax=ax
    )
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.suptitle('Phần 1: So sánh Phân phối Đặc trưng Chi phí giữa 2 Nhóm Nhãn', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_phan1_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# VẼ PHẦN 3: TỶ LỆ TỐI ƯU THEO CHIẾN DỊCH (ĐỘC LẬP) VS TƯƠNG TÁC KÊNH × CHIẾN DỊCH
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

# Biểu đồ trái: Tỷ lệ tuần đạt Tối ưu theo từng Chiến dịch khi xét độc lập
camp_opt = weekly_df.groupby('campaign')['is_optimized'].mean().sort_values(ascending=True) * 100
bars = axes[0].barh(camp_opt.index, camp_opt.values, color='#3498db', height=0.55)
axes[0].axvline(50, color='#e74c3c', linestyle='--', linewidth=1.5, label='Ngưỡng cân bằng (50%)')
axes[0].set_xlim(0, 75)
axes[0].set_title('Tỷ lệ tuần Tối ưu theo Chiến dịch (Xét độc lập)', fontweight='bold')
axes[0].set_xlabel('Tỷ lệ đạt Tối ưu (%)')
axes[0].legend()
for bar in bars:
    w = bar.get_width()
    axes[0].text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontweight='bold')

# Biểu đồ phải: Heatmap chứng minh hiệu ứng tương tác giữa Kênh và Chiến dịch
pivot_opt = weekly_df.pivot_table(index='channel', columns='campaign', values='is_optimized', aggfunc='mean') * 100
sns.heatmap(
    pivot_opt, annot=True, fmt='.0f', cmap='RdYlGn',
    vmin=0, vmax=100, cbar_kws={'label': 'Tỷ lệ Tối ưu (%)'}, ax=axes[1]
)
axes[1].set_title('Hiệu ứng tương tác: Tỷ lệ tuần Tối ưu (%) theo Kênh × Chiến dịch', fontweight='bold')
axes[1].set_xlabel('Chiến dịch')
axes[1].set_ylabel('Kênh')

plt.suptitle('Phần 3: Tác động Độc lập của Chiến dịch và Hiệu ứng Tương tác với Kênh', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_phan3_campaign_interaction.png', dpi=150, bbox_inches='tight')
plt.show()