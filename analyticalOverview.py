import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # chia dữ liệu thành tập train/test
from sklearn.preprocessing import OneHotEncoder, StandardScaler# tiền xử lý trước khi đưa vào model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid") # nền trắng, lưới
plt.rcParams['figure.figsize'] = (15, 6) # kích thước mặc định của biểu đồ ngang 15, dọc 6

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

overall_roas = total_revenue / total_cost # bỏ ra 1 đồng quảng cáo, thu về bao nhiêu đồng doanh thu
overall_roi = total_profit / total_cost # bỏ ra 1 đồng quảng cáo, thu về bao nhiêu đồng lợi nhuận
overall_cpc = total_cost / total_clicks # chi phí trung bình cho mỗi lượt nhấp (Cost Per Click)
overall_cpa = total_cost / total_conversions # chi phí trung bình cho mỗi lượt chuyển đổi (Cost Per Acquisition)
overall_ctr = total_clicks / total_impressions # tỷ lệ nhấp chuột
overall_cr = total_conversions / total_clicks # tỷ lệ chuyển đổi (Conversion Rate)

print("=== 1. TỔNG QUAN TOÀN BỘ TỆP DỮ LIỆU ===")
print(f"- Tổng Chi phí (Cost):       {total_cost:,.2f}") # .2f là làm tròn hiển thị 2 chữ số thập phân, , là hiển thị dấu phẩy phân cách hàng nghìn
print(f"- Tổng Doanh thu (Revenue):  {total_revenue:,.2f}")
print(f"- Tổng Lợi nhuận (Profit):   {total_profit:,.2f}")
print(f"- ROAS tổng thể: {overall_roas:.4f} | ROI tổng thể: {overall_roi:.4f}")
print(f"- CPC: {overall_cpc:.4f} | CPA: {overall_cpa:.4f} | CTR: {overall_ctr:.4f} | CR: {overall_cr:.4f}\n")

# ==============================================================================
# PHẦN 2: PHÂN TÍCH SO SÁNH THEO TỪNG KÊNH (CHANNEL)
# ==============================================================================
channel_df = df.groupby('channel').agg({ # gom nhóm theo channel và tính tổng các chỉ số
    'cost': 'sum',
    'revenue': 'sum',
    'profit': 'sum',
    'impressions': 'sum',
    'clicks': 'sum',
    'conversions': 'sum'
}).reset_index() # đưa index về dạng cột bình thường

channel_df['roas'] = channel_df['revenue'] / channel_df['cost']
channel_df['roi'] = channel_df['profit'] / channel_df['cost']
channel_df['cpc'] = channel_df['cost'] / channel_df['clicks']
channel_df['cpa'] = channel_df['cost'] / channel_df['conversions']
channel_df['ctr'] = channel_df['clicks'] / channel_df['impressions']
channel_df['cr'] = channel_df['conversions'] / channel_df['clicks']
channel_df = channel_df.sort_values(by='roas', ascending=False) # sắp xếp theo ROAS giảm dần để thấy kênh nào hiệu quả nhất
# ascending=False là sắp xếp giảm dần, ascending=True là sắp xếp tăng dần

print("=== 2. BẢNG SO SÁNH HIỆU QUẢ THEO KÊNH ===")
print(channel_df[['channel', 'cost', 'revenue', 'profit', 'roas', 'roi', 'cpa', 'cpc']].to_string(index=False))
# in ra bảng so sánh các chỉ số theo từng kênh, chỉ lấy các cột cần thiết và không in index
print("\n")

# ==============================================================================
# PHẦN 3: PHÂN TÍCH CHI TIẾT KÊNH + CHIẾN DỊCH & GÁN NHÃN ROAS
# ==============================================================================
#.agg() tức là tổng hợp cho phép gom tất cả các phép tính toán (sum, mean, median...) vào trong một hàm duy nhất
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
cc_df['is_optimized'] = (cc_df['roas'] > roas_median).astype(int) # tạo cột is_optimized = 1 nếu ROAS > Median, ngược lại = 0
cc_df['status'] = cc_df['is_optimized'].map({1: 'Tối ưu', 0: 'Chưa tối ưu'}) # tạo cột status = 'Tối ưu' nếu is_optimized = 1, ngược lại = 'Chưa tối ưu'

print(f"=== 3. CHI TIẾT KÊNH + CHIẾN DỊCH (Ngưỡng Median ROAS = {roas_median:.4f}) ===")
print(cc_df[['channel', 'campaign', 'cost', 'revenue', 'roas', 'cpa', 'status']].sort_values(by='roas', ascending=False).to_string(index=False))
print("\n")

# ==============================================================================
# PHẦN 4: HUẤN LUYỆN & SO SÁNH 3 MÔ HÌNH MACHINE LEARNING (CHU KỲ TUẦN)
# ==============================================================================
df['date'] = pd.to_datetime(df['date']) # chuyển cột date về dạng datetime để có thể thao tác với ngày tháng
df['week'] = df['date'].dt.to_period('W').astype(str) # tạo cột week = tuần của ngày trong cột date, định dạng chu kỳ tuần (W) và chuyển về chuỗi để dễ hiển thị

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
num_features = ['cost', 'impressions', 'clicks', 'cpc', 'ctr', 'cpm'] # không đưa revenue, conversions và cpa vào vì chúng tính trực tiếp ra roas gây ra rò rỉ dữ liệu
cat_features = ['channel', 'campaign']
features = cat_features + num_features

X = weekly_df[features]
y = weekly_df['is_optimized']

# Chia tập Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# phần này được hiểu như sau x_train lấy 80% dữ liệu để huấn luyện mô hình, x_test lấy 20% dữ liệu còn lại để kiểm tra mô hình
# y_train là đáp án tương ứng của 80% dữ liệu huấn luyện, y_test là đáp án tương ứng của 20% dữ liệu kiểm tra
# random_state=42 là để đảm bảo kết quả chia dữ liệu giống nhau mỗi lần chạy, stratify=y là để đảm bảo tỷ lệ nhãn trong tập train/test giống nhau

preprocessor = ColumnTransformer(
    transformers=[# transformers là một danh sách các bộ biến đổi, mỗi bộ gồm tên, bộ biến đổi và danh sách cột áp dụng
        ('num', StandardScaler(), num_features),# standardScaler chuẩn hóa các biến số về trung bình 0 và độ lệch chuẩn 1 num_features là danh sách các cột số cần chuẩn hóa
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        # oneHotEncoder mã hóa các biến phân loại thành các cột nhị phân 0/1, drop='first' bỏ đi cột nhị phân đầu tiên
        # ví dụ có 3 cột fb, ggads, ins thì chỉ cần tạo 2 cột ggads và ins nếu cả 2 đều = 0 thì máy tự hiểu dòng còn lại là fb giúp giảm đa cộng tuyến
        # sparse_output=False trả về kết quả dạng mảng số thông thường thay vì ma trận thưa sparse, giúp dễ dàng thao tác với dữ liệu
    ]
)

# Khởi tạo 3 mô hình phân loại để đối chiếu
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    # mô hình này tạo ra 1 rừng cây quyết định (decision tree) với 150 cây, độ sâu tối đa của mỗi cây là 8, random_state=42 để đảm bảo kết quả giống nhau mỗi lần chạy
    # ưu Rất mạnh mẽ, ít bị học vẹt (Overfitting), tự động nắm bắt tốt các mối quan hệ phức tạp giữa chi phí, lượt nhấp và kênh quảng cáo
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    # mô hình này tạo một vòng phân loại tuyến tính nó tính tổng trọng số các đặc trưng sau đó ép kết quả về dạng 0 và 1 
    # mô hình hồi quy logistic, max_iter=1000 là số vòng lặp tối đa để hội tụ, random_state=42 để đảm bảo kết quả giống nhau mỗi lần chạy
    # ưu Tốc độ huấn luyện cực nhanh, dễ giải thích xem yếu tố nào (ví dụ ctr tăng hay cpc giảm) tác động mạnh nhất đến khả năng tối ưu
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    # mô hình này tạo ra một chuỗi các cây quyết định, mỗi cây mới học từ lỗi của cây trước đó, n_estimators=150 là số lượng cây, learning_rate=0.05 là tốc độ học, max_depth=4 là độ sâu tối đa của mỗi cây
    # ưu Thường đạt độ chính xác cao nhất trong các bài toán dữ liệu dạng bảng (Tabular Data) như dữ liệu Marketing/Tài chính, nhưng thời gian huấn luyện sẽ lâu hơn Logistic Regression.
}

print("=== 4. SO SÁNH HIỆU NĂNG 3 MÔ HÌNH TRÊN TẬP TEST (DỮ LIỆU TUẦN) ===")
results = []
trained_models = {}

for name, clf in classifiers.items():# duyệt qua từng mô hình trong classifiers, name là tên mô hình, clf là đối tượng mô hình
    pipe = Pipeline(steps=[# tạo một pipeline gồm 2 bước: tiền xử lý dữ liệu và huấn luyện mô hình
        ('preprocessor', preprocessor),# trạm 1 nhận dữ liệu X tự chuẩn hóa bằng standardScaler và mã hóa OneHotEncoder chữ thành số
        ('classifier', clf)# trạm 2 nhận dữ liệu đã chuẩn hóa và huấn luyện mô hình phân loại clf (Random Forest, Logistic Regression hoặc Gradient Boosting)
    ])
    pipe.fit(X_train, y_train)# kích hoạt học trên 80% dữ liệu huấn luyện 
    # ở trạm 1 nó tính toán rồi ghi nhớ gtri trung bình và độ lệch chuẩn của các cột số, đồng thời ghi nhớ các giá trị phân loại của các cột chữ để mã hóa OneHotEncoder
    # ở trạm 2 liên hệ giữa dữ liệu x_train đã biến đổi và đáp án y_train để tìm ra các trọng số tối ưu cho mô hình phân loại
    y_pred = pipe.predict(X_test)
    # đưa dữ liệu 20% tập kiểm tra X_test đã chuẩn hóa vào mô hình để dự đoán nhãn tối ưu (0 hoặc 1) và lưu kết quả vào y_pred
    # tự áp dụng thông số đã train để biến đổi x_test không tính lại  sau đó đưa ra dự đoán nhãn tối ưu (0 hoặc 1) cho từng dòng dữ liệu trong X_test
    y_prob = pipe.predict_proba(X_test)[:, 1]
    # predict_proba() trả về bảng gồm 2 cột xác suất 0 và 1 tổng xs 2 cột trên 1 hàng luôn bằng 1.0 
    # [:, 1] lấy tất cả các hàng nhưng chỉ lấy cột thứ 2 (index =1) tức chỉ giữ lại các số xác suất đạt trạng thái tối ưu
    
    acc = accuracy_score(y_test, y_pred)
    # so sánh trực tiếp đáp án thực tế y_test với dự đoán y_pred ( 0 hoặc 1) để tính tỷ lệ dự đoán đúng trên tổng số dòng dữ liệu
    # acc = số tuần đúng / tổng số tuần
    pr_auc = average_precision_score(y_test, y_prob)
    # so sánh đáp án thật y_test với điểm xác suất y_prob để tính diện tích dưới đường cong Precision-Recall (PR-AUC)
    # PR-AUC = 1 là dự đoán hoàn hảo, PR-AUC = 0.5 là dự đoán ngẫu nhiên, PR-AUC < 0.5 là dự đoán sai lệch
    roc_auc = roc_auc_score(y_test, y_prob)
    # so sánh đáp án thật y_test với điểm xác suất y_prob để tính diện tích dưới đường cong ROC (ROC-AUC)
    # ROC-AUC = 1 là dự đoán hoàn hảo, ROC-AUC = 0.5 là dự đoán ngẫu nhiên, ROC-AUC < 0.5 là dự đoán sai lệch
    
    results.append({'Mô hình': name, 'Accuracy': acc, 'PR-AUC': pr_auc, 'ROC-AUC': roc_auc})
    trained_models[name] = pipe

results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
print(results_df.to_string(index=False))

# Chọn mô hình tốt nhất dựa trên ROC-AUC (hoặc chỉ định 'Random Forest' nếu muốn cố định)
best_model_name = results_df.iloc[0]['Mô hình']
best_model = trained_models[best_model_name]
print(f"\n-> Mô hình tốt nhất được chọn để dự đoán xác suất tối ưu: {best_model_name}\n")

# Tính trung bình theo tuần của 42 cặp Kênh + Chiến dịch để dự đoán xác suất tối ưu
cc_weekly_avg = weekly_df.groupby(['channel', 'campaign'])[num_features].mean().reset_index()
cc_df['opt_prob'] = best_model.predict_proba(cc_weekly_avg[features])[:, 1]

# Quy tắc đề xuất tăng/giảm ngân sách kết hợp giữa Nhãn thực tế & Xác suất mô hình
def recommend_budget(row):
    if row['opt_prob'] >= 0.70 and row['is_optimized'] == 1:
        return 'Tăng ngân sách (+30%)'
    elif row['opt_prob'] < 0.40 and row['is_optimized'] == 0:
        return 'Giảm ngân sách (-30%)'
    else:
        return 'Giữ nguyên / Tối ưu nội dung'

cc_df['recommendation'] = cc_df.apply(recommend_budget, axis=1)

print(f"=== 5. BẢNG ĐỀ XUẤT PHÂN BỔ NGÂN SÁCH TỪ MÔ HÌNH ({best_model_name}) ===")
print(cc_df[['channel', 'campaign', 'roas', 'opt_prob', 'status', 'recommendation']].sort_values(by='opt_prob', ascending=False).to_string(index=False))

# ==============================================================================
# PHẦN 5: TRỰC QUAN HÓA KẾT QUẢ PHÂN TÍCH
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x_pos = np.arange(len(channel_df))
width = 0.35
axes[0].bar(x_pos - width/2, channel_df['cost']/1e6, width, label='Chi phí (Triệu USD)', color='#e74c3c')
axes[0].bar(x_pos + width/2, channel_df['revenue']/1e6, width, label='Doanh thu (Triệu USD)', color='#2ecc71')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(channel_df['channel'], rotation=25)
axes[0].set_title('So sánh Chi phí và Doanh thu theo Kênh')
axes[0].set_ylabel('Triệu USD')
axes[0].legend()

pivot_roas = cc_df.pivot(index='channel', columns='campaign', values='roas')
sns.heatmap(pivot_roas, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[1])
axes[1].set_title('Ma trận ROAS theo Kênh và Chiến dịch')
axes[1].set_xlabel('Chiến dịch')
axes[1].set_ylabel('Kênh')

plt.tight_layout()
plt.savefig('phan_tich_quang_cao2.png', dpi=150)
plt.show()