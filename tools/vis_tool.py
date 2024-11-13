import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 读取过滤后的数据
with open('/home/bingxing2/ailab/scxlab0056/CODE/DS_CMA_HRRR/meta_info_weather5k_hrrr.json', 'r', encoding='utf-8') as f:
    filtered_data = json.load(f)

# 提取经纬度
lons = [info['longitude'] for info in filtered_data.values()]
lats = [info['latitude'] for info in filtered_data.values()]

# 创建地图
plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置可视化区域
ax.set_extent([239, 286, 25, 47], crs=ccrs.PlateCarree())

# 添加海岸线和边界
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 绘制点
ax.scatter(lons, lats, color='red', s=50, label='Locations', transform=ccrs.PlateCarree())

# 添加标题和图例
plt.title('Filtered Locations Visualization')
plt.legend()

# 显示图形
plt.show()