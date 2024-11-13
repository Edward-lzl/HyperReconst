import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.feature as cf
import io, os
import numpy as np
from PIL import Image

T_MEAN = 288.98371189102784
T_STD = 288.98371189102784

class VisField():
    def __init__(self, lons: list, lats: list):
        super().__init__()
        self.lon = [self.adjust_longitude(lon) for lon in lons]
        self.lat = lats
    
        self.projection = ccrs.PlateCarree()
        self.crs = ccrs.PlateCarree()
    def adjust_longitude(self, lon):
        """调整经度到-180到180范围内"""
        return (lon + 180) % 360 - 180
    
    def forward_single_image_w_stn_gt(self, result_grid, result_stn, interp_stn, label_grid, label_stn, coord_stn, var_name, result_file_name):
        
        #result_grid = result_grid.permute(1,0).reshape(1,320,320).cpu().detach().numpy()#*T_STD+T_MEAN
        #result_stn = result_stn.cpu().detach().numpy()#*T_STD+T_MEAN
        #interp_stn = interp_stn.cpu().detach().numpy()#*T_STD+T_MEAN
        label_grid = label_grid.permute(1,0).reshape(1,320,320).cpu().detach().numpy()#*T_STD+T_MEAN
        label_stn = label_stn.cpu().detach().numpy()#*T_STD+T_MEAN
        coord_stn = coord_stn.cpu().detach().numpy()
        
        data_min = np.min(label_grid)
        data_max = np.max(label_grid)
        if var_name == 't2m':
            _cmap = 'RdYlBu_r'
            bar_name = '[K]'
            vmax = 0.05
            vmin = 0.
        elif var_name == 'tp1h':
            _cmap = 'Blues'
            bar_name = '[mm]'
            vmax = 0.08
            vmin = 0.
            data_min = data_min-1.e-3
        elif var_name == 'sp':
            _cmap = 'jet'
            bar_name = '[Pa]'
            vmax = 0.25
            vmin = 0.
        elif var_name == 'u10' or var_name == 'v10':
            bar_name = '[m/s]'
            _cmap = 'seismic'
            vmax = 0.12
            vmin = 0.
        
        p_lon = self.adjust_longitude(coord_stn[:-320*320,0])
        p_lat = coord_stn[:-320*320,1]
        
        plt.figure(dpi=150)
        fig, axes = plt.subplots(1, 1, subplot_kw={'projection':self.crs},figsize=(5,5))
        
        axes.set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        # axes[1].set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        # axes[2].set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        
        if data_min<0 and data_max>0:
            data_min = -max(np.abs(data_min), np.abs(data_max))
            data_max = max(np.abs(data_min), np.abs(data_max))
        # ax.set_title(f"Input Field of {var_name}")
        im1 = axes.imshow(label_grid[0][::-1,:],cmap=_cmap, extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        #img_bar1 = plt.colorbar(im1, shrink=0.66)
        #img_bar1.set_label(bar_name)
        #loss_min, loss_max = np.percentile(loss_list, [5,95])
        # norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
        # 添加海岸线
        axes.coastlines()

        # 添加网格
        axes.gridlines(draw_labels=True)
        axes.set_title("Ground")
        scatter1 = axes.scatter(p_lon.tolist(), p_lat.tolist(), marker='o', s=50, c=label_stn[:,0].tolist(), cmap=_cmap, vmin=data_min, vmax=data_max, edgecolors='k', transform=self.crs)
        #sc_bar = plt.colorbar(scatter, shrink=0.66, orientation='horizontal')
        #sc_bar.set_label("Norm-MSE")
        
        # im2 = axes[1].imshow(label_grid[0][::-1,:],cmap=_cmap, extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        # axes[1].coastlines()
        # axes[1].set_title("Interp")
        
        # # 添加网格
        # axes[1].gridlines(draw_labels=True)
        # scatter2 = axes[1].scatter(p_lon.tolist(), p_lat.tolist(), marker='o', s=50, c=interp_stn[0,:].tolist(), cmap=_cmap, vmin=data_min, vmax=data_max, edgecolors='k', transform=self.crs)
        
        # im3 = axes[2].imshow(result_grid[0][::-1,:],cmap=_cmap, extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        # axes[2].coastlines()
        # axes[2].set_title("Pred")

        # # 添加网格
        # axes[2].gridlines(draw_labels=True)
        # #img_bar2 = plt.colorbar(im1, shrink=0.66)
        # #img_bar2.set_label(bar_name)
        # #loss_min, loss_max = np.percentile(loss_list, [5,95])
        # # norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
        # scatter3 = axes[2].scatter(p_lon.tolist(), p_lat.tolist(), marker='o', s=50, c=result_stn[:,0].tolist(), cmap=_cmap, vmin=data_min, vmax=data_max, edgecolors='k', transform=self.crs)
        plt.tight_layout()
        plt.savefig(result_file_name)
        plt.close()
    
    def forward_single_image_w_stn(self, result_grid, result_stn, interp_stn, label_grid, label_stn, coord_stn, var_name, result_file_name):
        
        result_grid = result_grid.permute(1,0).reshape(1,320,320).cpu().detach().numpy()*T_STD+T_MEAN
        result_stn = result_stn.cpu().detach().numpy()*T_STD+T_MEAN
        interp_stn = interp_stn.cpu().detach().numpy()*T_STD+T_MEAN
        label_grid = label_grid.permute(1,0).reshape(1,320,320).cpu().detach().numpy()*T_STD+T_MEAN
        label_stn = label_stn.cpu().detach().numpy()*T_STD+T_MEAN
        coord_stn = coord_stn.cpu().detach().numpy()
        
        data_min = np.min(label_grid)
        data_max = np.max(label_grid)
        if var_name == 't2m':
            _cmap = 'RdYlBu_r'
            bar_name = '[K]'
            vmax = 0.05
            vmin = 0.
        elif var_name == 'tp1h':
            _cmap = 'Blues'
            bar_name = '[mm]'
            vmax = 0.08
            vmin = 0.
            data_min = data_min-1.e-3
        elif var_name == 'sp':
            _cmap = 'jet'
            bar_name = '[Pa]'
            vmax = 0.25
            vmin = 0.
        elif var_name == 'u10' or var_name == 'v10':
            bar_name = '[m/s]'
            _cmap = 'seismic'
            vmax = 0.12
            vmin = 0.
        
        p_lon = self.adjust_longitude(coord_stn[:-320*320,0])*360
        p_lat = coord_stn[:-320*320,1]*90
        
        plt.figure(dpi=150)
        fig, axes = plt.subplots(1, 3, subplot_kw={'projection':self.crs},figsize=(15,5))
        
        axes[0].set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        axes[1].set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        axes[2].set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        
        if data_min<0 and data_max>0:
            data_min = -max(np.abs(data_min), np.abs(data_max))
            data_max = max(np.abs(data_min), np.abs(data_max))
        # ax.set_title(f"Input Field of {var_name}")
        im1 = axes[0].imshow(label_grid[0][::-1,:],cmap=_cmap, extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        #img_bar1 = plt.colorbar(im1, shrink=0.66)
        #img_bar1.set_label(bar_name)
        #loss_min, loss_max = np.percentile(loss_list, [5,95])
        # norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
        # 添加海岸线
        axes[0].coastlines()

        # 添加网格
        axes[0].gridlines(draw_labels=True)
        axes[0].set_title("Ground")
        scatter1 = axes[0].scatter(p_lon.tolist(), p_lat.tolist(), marker='o', s=50, c=label_stn[:,0].tolist(), cmap=_cmap, vmin=data_min, vmax=data_max, edgecolors='k', transform=self.crs)
        #sc_bar = plt.colorbar(scatter, shrink=0.66, orientation='horizontal')
        #sc_bar.set_label("Norm-MSE")
        
        im2 = axes[1].imshow(label_grid[0][::-1,:],cmap=_cmap, extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        axes[1].coastlines()
        axes[1].set_title("Interp")
        
        # 添加网格
        axes[1].gridlines(draw_labels=True)
        scatter2 = axes[1].scatter(p_lon.tolist(), p_lat.tolist(), marker='o', s=50, c=interp_stn[0,:].tolist(), cmap=_cmap, vmin=data_min, vmax=data_max, edgecolors='k', transform=self.crs)
        
        im3 = axes[2].imshow(result_grid[0][::-1,:],cmap=_cmap, extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        axes[2].coastlines()
        axes[2].set_title("Pred")

        # 添加网格
        axes[2].gridlines(draw_labels=True)
        #img_bar2 = plt.colorbar(im1, shrink=0.66)
        #img_bar2.set_label(bar_name)
        #loss_min, loss_max = np.percentile(loss_list, [5,95])
        # norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
        scatter3 = axes[2].scatter(p_lon.tolist(), p_lat.tolist(), marker='o', s=50, c=result_stn[:,0].tolist(), cmap=_cmap, vmin=data_min, vmax=data_max, edgecolors='k', transform=self.crs)
        plt.tight_layout()
        plt.savefig(result_file_name)
        plt.close()


    def forward_single_image(self, data, result_file_name):
        # if var_name == 't2m':
        #     _cmap = 'RdYlBu'
        # elif var_name == 'tp1h':
        #     _cmap = 'Blues'
        # elif var_name == 'sp':
        #     _cmap = 'jet'
        # elif var_name == 'u10' or var_name == 'v10':
        #     _cmap = 'seismic'
        _cmap = 'RdYlBu_r'

        plt.figure(dpi=150)
        ax = plt.axes(projection = self.projection, frameon=True)
        # gl = ax.gridlines(crs=self.crs, draw_labels=True,
        #                         linewidth=.6, color='gray',
        #                         alpha=0.5, linestyle='-.')
        # gl.xlabel_style = {"size": 7}
        # gl.ylabel_style = {"size": 7}
        def adjust_longitude(lon):
            return (lon + 180) % 360 - 180
        self.lon[1] = adjust_longitude(self.lon[1])
        ax.set_extent([min(self.lon), max(self.lon[1]), min(self.lat), max(self.lat)], crs=self.crs)
        data_min = np.min(data)
        data_max = np.max(data)
        if data_min<0 and data_max>0:
            data_min = -max(np.abs(data_min), np.abs(data_max))
            data_max = max(np.abs(data_min), np.abs(data_max))
        # ax.set_title(f"Input Field of {var_name}")
        im1 = ax.imshow(data,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        # plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.savefig(result_file_name,bbox_inches='tight')
        plt.close()
    
        
    def forward(self, data_input, data_pred, data_baseline, data_gt, var_name, result_file_name):
        # generate basemap
        # plt.figure(dpi=150)

        if var_name == 't2m':
            _cmap = 'RdYlBu_r'
        elif var_name == 'tp1h':
            _cmap = 'Blues'
        elif var_name == 'sp':
            _cmap = 'jet'
        elif var_name == 'u10' or var_name == 'v10':
            _cmap = 'seismic'
        fig, axes = plt.subplots(2, 3, subplot_kw={'projection':self.projection},figsize=(30,10))
        for i in range(2):
            for j in range(3):
                gl = axes[i,j].gridlines(crs=self.crs, draw_labels=True,
                                linewidth=.6, color='gray',
                                alpha=0.5, linestyle='-.')
                gl.xlabel_style = {"size": 7}
                gl.ylabel_style = {"size": 7}
                axes[i,j].set_extent([min(self.lon), max(self.lon), min(self.lat), max(self.lat)], crs=self.crs)
        
        # ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
        # ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

        data_min = min(np.min(data_pred), np.min(data_gt))
        data_max = max(np.max(data_pred), np.max(data_gt))
        if data_min<0 and data_max>0:
            data_min = -max(np.abs(data_min), np.abs(data_max))
            data_max = max(np.abs(data_min), np.abs(data_max))
        
        
        axes[0,0].set_title(f"Input Field of {var_name}")
        im1 = axes[0,0].imshow(data_input,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im1, ax=axes[0,0], shrink=0.5,orientation='horizontal')
        axes[1,0].set_title(f"Target Field of {var_name}")
        im2 = axes[1,0].imshow(data_gt,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im2, ax=axes[1,0], shrink=0.5)
        axes[0,1].set_title(f"Pred Field of {var_name}")
        im3 = axes[0,1].imshow(data_pred,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im3, ax=axes[0,1], shrink=0.5)
        axes[1,1].set_title(f"Baseline Field of {var_name}")
        im4 = axes[1,1].imshow(data_baseline,cmap=_cmap,extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=data_min, vmax=data_max, transform=self.crs)
        plt.colorbar(im4, ax=axes[1,1], shrink=0.5)
        
        ape1 = np.abs((data_gt-data_pred)/(np.abs(data_gt)+1e-4))
        axes[0,2].set_title(f"Pred Error Field of {var_name}")
        im5 = axes[0,2].imshow(ape1, cmap='OrRd',extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=np.percentile(ape1,0.01), vmax=np.percentile(ape1,99.99), transform=self.crs)
        plt.colorbar(im5, ax=axes[0,2], shrink=0.5)

        ape2 = np.abs((data_gt-data_baseline)/(np.abs(data_gt)+1e-4))
        axes[1,2].set_title(f"Baseline Error Field of {var_name}")
        im5 = axes[1,2].imshow(ape2, cmap='OrRd',extent=[min(self.lon), max(self.lon), min(self.lat), max(self.lat)], vmin=np.percentile(ape1,0.01), vmax=np.percentile(ape1,99.99), transform=self.crs)
        plt.colorbar(im5, ax=axes[1,2], shrink=0.5)
        plt.savefig(result_file_name,bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    lon_range = [72., 136.]
    lat_range = [18., 54.]
    client = Client(conf_path="/mnt/petrelfs/liuzili/petreloss.conf")
    data_path = 'cluster1:s3://pretrained_models/TIGGE/NCEP_1d/2021/GFS_2021-01-01-00-00-00_f000_rio.tiff'
    with io.BytesIO(client.get(data_path)) as f:
        # data = np.load(f)[144:288, 320:544]
        data = np.array(Image.open(f))
    # data_path = '/mnt/petrelfs/liuzili/data/PINNs_draw/results/GFS_2021-01-01-00-00-00_f000_v10.tiff'
        
        # data = np.array(Image.open(data_path))
    result_path = '/mnt/petrelfs/liuzili/data/PINNs_draw/vis'
    

    VisUtil = VisField(lon_range, lat_range)
    # data_input = data[::4,::4]
    data = np.flipud(data)
    filename = os.path.join(result_path, 'Pred_GFS_rio.png')
    VisUtil.forward_single_image(data,'v10', filename)