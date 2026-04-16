import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 1. VERİ YÜKLEME ──────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/VRP.csv')
print(f"Veri seti: {df.shape[0]:,} VRP örneği, {df.shape[1]} özellik")

# ── 2. ÖZELLİK MÜHENDİSLİĞİ ─────────────────────────────────────────
df['distance_spread']   = df['max_distance_depot'] - df['min_distance_depot']
df['demand_spread']     = df['max_demand'] - df['min_demand']
df['avg_route_load']    = df['num_customers'] * df['average_demand'] / df['vehicle_capacity']
df['capacity_util']     = (df['num_customers'] * df['average_demand']) / df['vehicle_capacity']
df['distance_per_cust'] = df['average_distance_nondepot'] / df['num_customers']
df['obj_per_customer']  = df['best_objective_value'] / df['num_customers']

# Araç sayısı tahmini (ceil(toplam talep / kapasite))
df['est_vehicles'] = np.ceil(df['num_customers'] * df['average_demand'] / df['vehicle_capacity'])

# ── 3. KEŞİFSEL ANALİZ ───────────────────────────────────────────────
print("\nSüreç Büyüklüğü Dağılımı:")
print(df['num_customers'].describe())
print(f"\nOrtalama en iyi çözüm: {df['best_objective_value'].mean():,.0f}")
print(f"Ortalama hesaplama süresi: {df['computational_time'].mean():.1f} sn")

# ── 4. TAHMİN MODELİ: Optimal Çözüm Değeri ───────────────────────────
features = ['min_distance_depot','average_distance_depot','max_distance_depot',
            'average_distance_nondepot','average_demand','num_customers',
            'vehicle_capacity','distance_spread','demand_spread',
            'avg_route_load','capacity_util','distance_per_cust','est_vehicles']

X = df[features]
y = df['best_objective_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model':  model,
        'y_pred': y_pred,
        'r2':     r2_score(y_test, y_pred),
        'mae':    mean_absolute_error(y_test, y_pred),
        'rmse':   np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    print(f"{name}: R²={results[name]['r2']:.3f} | MAE={results[name]['mae']:,.0f} | RMSE={results[name]['rmse']:,.0f}")

best_model = results['Random Forest']
rf = best_model['model']

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# ── 5. VRP SİMÜLASYONU: Nearest Neighbor vs Greedy ──────────────────
np.random.seed(42)
N = 20  # müşteri sayısı
depot = np.array([500, 500])
customers = np.random.randint(50, 950, size=(N, 2))
demands   = np.random.randint(20, 80, size=N)
capacity  = 200

all_points = np.vstack([depot, customers])
labels = ['Depo'] + [f'M{i+1}' for i in range(N)]

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def nearest_neighbor(depot, customers, demands, capacity):
    """Nearest Neighbor heuristik"""
    unvisited = list(range(len(customers)))
    routes = []
    total_dist = 0
    while unvisited:
        route = []
        load = 0
        current = depot.copy()
        while unvisited:
            # En yakın ziyaret edilmemiş müşteriyi bul
            best_idx = None
            best_d   = float('inf')
            for i in unvisited:
                if load + demands[i] <= capacity:
                    d = dist(current, customers[i])
                    if d < best_d:
                        best_d   = d
                        best_idx = i
            if best_idx is None:
                break
            route.append(best_idx)
            load    += demands[best_idx]
            total_dist += dist(current, customers[best_idx])
            current  = customers[best_idx]
            unvisited.remove(best_idx)
        total_dist += dist(current, depot)
        routes.append(route)
    return routes, total_dist

def two_opt(route, customers, depot):
    """2-opt lokal arama iyileştirmesi"""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-1):
            for j in range(i+1, len(best)):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new_route, customers, depot) < route_distance(best, customers, depot):
                    best = new_route
                    improved = True
    return best

def route_distance(route, customers, depot):
    if not route:
        return 0
    d = dist(depot, customers[route[0]])
    for i in range(len(route)-1):
        d += dist(customers[route[i]], customers[route[i+1]])
    d += dist(customers[route[-1]], depot)
    return d

# Nearest Neighbor çözümü
nn_routes, nn_total = nearest_neighbor(depot, customers, demands, capacity)

# 2-opt iyileştirme
opt_routes   = [two_opt(r, customers, depot) for r in nn_routes]
opt_total    = sum(route_distance(r, customers, depot) for r in opt_routes)
improvement  = (nn_total - opt_total) / nn_total * 100

print(f"\nVRP Simülasyonu ({N} müşteri, kapasite={capacity}):")
print(f"  Nearest Neighbor toplam mesafe: {nn_total:,.0f}")
print(f"  2-opt sonrası toplam mesafe:    {opt_total:,.0f}")
print(f"  İyileşme: %{improvement:.1f}")
print(f"  Kullanılan araç sayısı: {len(opt_routes)}")

# ── 6. GÖRSELLEŞTİRME ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Araç Rota Optimizasyonu (VRP) — Kaggle Dataset Analizi + Simülasyon\n'
             f'4,550 VRP Örneği | Random Forest R²=0.99 | 2-opt İyileştirme %{improvement:.0f}',
             fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
route_colors = ['#E24B4A','#2E75B6','#1F9E75','#F2A623','#7030A0','#C55A11']

# ─ 1. Nearest Neighbor rotası
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(customers[:,0], customers[:,1], c='#2E75B6', s=60, zorder=5)
ax1.scatter(*depot, c='#E24B4A', s=200, marker='*', zorder=6, label='Depo')
for i, (r, c) in enumerate(zip(nn_routes, route_colors)):
    pts = [depot] + [customers[j] for j in r] + [depot]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    ax1.plot(xs, ys, c=c, linewidth=1.5, alpha=0.7, label=f'Araç {i+1}')
for i, (x,y) in enumerate(customers):
    ax1.annotate(f'M{i+1}', (x,y), fontsize=6, ha='center', va='bottom')
ax1.set_title(f'Nearest Neighbor\nToplam: {nn_total:,.0f}', fontweight='bold', fontsize=10)
ax1.legend(fontsize=7, loc='upper right')
ax1.set_xlim(0,1000); ax1.set_ylim(0,1000)

# ─ 2. 2-opt Optimized rotası
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(customers[:,0], customers[:,1], c='#2E75B6', s=60, zorder=5)
ax2.scatter(*depot, c='#E24B4A', s=200, marker='*', zorder=6, label='Depo')
for i, (r, c) in enumerate(zip(opt_routes, route_colors)):
    pts = [depot] + [customers[j] for j in r] + [depot]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    ax2.plot(xs, ys, c=c, linewidth=1.5, alpha=0.7, label=f'Araç {i+1}')
for i, (x,y) in enumerate(customers):
    ax2.annotate(f'M{i+1}', (x,y), fontsize=6, ha='center', va='bottom')
ax2.set_title(f'2-opt Optimizasyon\nToplam: {opt_total:,.0f} (↓%{improvement:.1f})',
              fontweight='bold', fontsize=10, color='#1F9E75')
ax2.legend(fontsize=7, loc='upper right')
ax2.set_xlim(0,1000); ax2.set_ylim(0,1000)

# ─ 3. Araç yük durumu
ax3 = fig.add_subplot(gs[0, 2])
route_loads  = [sum(demands[j] for j in r) for r in opt_routes]
route_labels = [f'Araç {i+1}' for i in range(len(opt_routes))]
bars = ax3.bar(route_labels, route_loads,
               color=route_colors[:len(opt_routes)], alpha=0.85)
ax3.axhline(y=capacity, color='red', linestyle='--', linewidth=2, label=f'Kapasite ({capacity})')
ax3.set_title('Araç Yük Durumu\n(2-opt Sonrası)', fontweight='bold', fontsize=10)
ax3.set_ylabel('Toplam Yük')
ax3.legend(fontsize=9)
for bar, val in zip(bars, route_loads):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ─ 4. Model karşılaştırması
ax4 = fig.add_subplot(gs[1, 0])
model_names_short = ['Lin. Reg.','Random\nForest','Gradient\nBoosting']
r2_vals = [results[m]['r2'] for m in models]
mae_vals = [results[m]['mae']/1000 for m in models]
x = np.arange(len(model_names_short))
w = 0.35
b1 = ax4.bar(x-w/2, r2_vals, w, label='R² Skoru', color='#2E75B6', alpha=0.85)
ax4_twin = ax4.twinx()
b2 = ax4_twin.bar(x+w/2, mae_vals, w, label='MAE (K)', color='#E24B4A', alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(model_names_short, fontsize=9)
ax4.set_ylabel('R² Skoru', color='#2E75B6')
ax4_twin.set_ylabel('MAE (Bin)', color='#E24B4A')
ax4.set_ylim(0, 1.2)
ax4.set_title('Model Karşılaştırması\n(Çözüm Değeri Tahmini)', fontweight='bold', fontsize=10)
for bar, val in zip(b1, r2_vals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E75B6')

# ─ 5. Feature importance
ax5 = fig.add_subplot(gs[1, 1])
top_feats = feat_imp.head(8)
feat_tr = {
    'num_customers':           'Müşteri Sayısı',
    'average_distance_depot':  'Ort. Depo Mesafesi',
    'est_vehicles':            'Tahmini Araç Sayısı',
    'avg_route_load':          'Ort. Rota Yükü',
    'capacity_util':           'Kapasite Kullanımı',
    'distance_per_cust':       'Müşteri Başı Mesafe',
    'average_distance_nondepot':'Müşteriler Arası Mes.',
    'max_distance_depot':      'Maks. Depo Mesafesi',
    'distance_spread':         'Mesafe Yayılımı',
    'average_demand':          'Ort. Talep',
}
colors_fi = ['#E24B4A' if v > top_feats.mean() else '#2E75B6' for v in top_feats.values]
bars5 = ax5.barh(range(len(top_feats)), top_feats.values, color=colors_fi)
ax5.set_yticks(range(len(top_feats)))
ax5.set_yticklabels([feat_tr.get(f,f) for f in top_feats.index], fontsize=9)
ax5.set_title('Özellik Önemi\n(Random Forest)', fontweight='bold', fontsize=10)
ax5.set_xlabel('Önem Skoru')
for bar, val in zip(bars5, top_feats.values):
    ax5.text(val+0.001, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8)

# ─ 6. Tahmin vs Gerçek
ax6 = fig.add_subplot(gs[1, 2])
y_pred_rf = best_model['y_pred']
ax6.scatter(y_test/1000, y_pred_rf/1000, alpha=0.3, s=5, color='#2E75B6')
lim = max(y_test.max(), y_pred_rf.max())/1000
ax6.plot([0,lim],[0,lim], 'r--', linewidth=2, label='Mükemmel Tahmin')
ax6.set_xlabel('Gerçek Değer (K)')
ax6.set_ylabel('Tahmin (K)')
ax6.set_title(f'Tahmin vs Gerçek\nR²={best_model["r2"]:.3f}', fontweight='bold', fontsize=10)
ax6.legend(fontsize=9)

# ─ 7. Müşteri sayısı vs çözüm değeri
ax7 = fig.add_subplot(gs[2, 0])
for cap, color in [(300,'#2E75B6'),(400,'#F2A623'),(500,'#E24B4A')]:
    sub = df[df['vehicle_capacity']==cap]
    ax7.scatter(sub['num_customers'], sub['best_objective_value']/1000,
                alpha=0.3, s=5, color=color, label=f'Kapasite={cap}')
ax7.set_xlabel('Müşteri Sayısı')
ax7.set_ylabel('En İyi Çözüm (K)')
ax7.set_title('Müşteri Sayısı vs\nOptimal Çözüm Değeri', fontweight='bold', fontsize=10)
ax7.legend(fontsize=9)

# ─ 8. Hesaplama süresi dağılımı
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(df[df['computational_time']<100]['computational_time'],
         bins=50, color='#2E75B6', alpha=0.7, edgecolor='white')
ax8.axvline(df['computational_time'].median(), color='red', linestyle='--',
            label=f"Medyan: {df['computational_time'].median():.1f} sn")
ax8.set_title('Hesaplama Süresi Dağılımı\n(<100 sn)', fontweight='bold', fontsize=10)
ax8.set_xlabel('Süre (saniye)')
ax8.set_ylabel('Frekans')
ax8.legend(fontsize=9)

# ─ 9. Gantt chart — araç çizelgesi
ax9 = fig.add_subplot(gs[2, 2])
speed = 50  # km/h simüle
service_time = 15  # dk/müşteri
time_cursor = [0] * len(opt_routes)
for vi, (route, color) in enumerate(zip(opt_routes, route_colors)):
    t = 0
    pts = [depot] + [customers[j] for j in route] + [depot]
    for k in range(len(pts)-1):
        travel = dist(pts[k], pts[k+1]) / speed * 60  # dakikaya çevir
        ax9.barh(vi, travel, left=t, color=color, alpha=0.7, height=0.5)
        t += travel
        if k < len(pts)-2:
            ax9.barh(vi, service_time, left=t, color='gray', alpha=0.4, height=0.5)
            t += service_time

ax9.set_yticks(range(len(opt_routes)))
ax9.set_yticklabels([f'Araç {i+1}' for i in range(len(opt_routes))], fontsize=9)
ax9.set_xlabel('Zaman (dakika)')
ax9.set_title('Araç Çizelgesi (Gantt)\n2-opt Rotası', fontweight='bold', fontsize=10)
travel_patch  = mpatches.Patch(color='#2E75B6', alpha=0.7, label='Seyahat')
service_patch = mpatches.Patch(color='gray',    alpha=0.4, label='Hizmet')
ax9.legend(handles=[travel_patch, service_patch], fontsize=8)

plt.savefig('/home/claude/vrp_analiz.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ── 7. ÖZET RAPOR ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("ARAÇ ROTA OPTİMİZASYONU — SONUÇ RAPORU")
print("="*60)
print(f"\nKaggle VRP Dataset: {len(df):,} problem örneği")
print(f"\nTahmin Modeli (Optimal Çözüm Değeri):")
for name, res in results.items():
    print(f"  {name:<25}: R²={res['r2']:.3f}, MAE={res['mae']:,.0f}")
print(f"\nSimülasyon ({N} müşteri):")
print(f"  Nearest Neighbor: {nn_total:,.0f}")
print(f"  2-opt Optimizasyon: {opt_total:,.0f} (↓%{improvement:.1f})")
print(f"  Araç sayısı: {len(opt_routes)}")
print(f"\n✓ Grafik kaydedildi: vrp_analiz.png")
