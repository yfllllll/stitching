import cv2 as cv    
import numpy as np    
import matplotlib.pyplot as plt    
from matplotlib.patches import Circle    
from collections import OrderedDict    
from types import SimpleNamespace    
import os  
  
# 复用 stitching_detailed.py 中的配置    
FEATURES_FIND_CHOICES = OrderedDict()    
FEATURES_FIND_CHOICES["orb"] = cv.ORB.create    
try:    
    FEATURES_FIND_CHOICES["sift"] = cv.SIFT_create  # 修正SIFT创建方法  
except AttributeError:    
    try:  
        FEATURES_FIND_CHOICES["sift"] = cv.xfeatures2d_SIFT.create  
    except AttributeError:  
        pass    
    
ESTIMATOR_CHOICES = OrderedDict()    
ESTIMATOR_CHOICES["homography"] = cv.detail_HomographyBasedEstimator    
ESTIMATOR_CHOICES["affine"] = cv.detail_AffineBasedEstimator    
    
BA_COST_CHOICES = OrderedDict()    
BA_COST_CHOICES["ray"] = cv.detail_BundleAdjusterRay    
BA_COST_CHOICES["reproj"] = cv.detail_BundleAdjusterReproj    
BA_COST_CHOICES["affine"] = cv.detail_BundleAdjusterAffinePartial    
    
WAVE_CORRECT_CHOICES = OrderedDict()    
WAVE_CORRECT_CHOICES["horiz"] = cv.detail.WAVE_CORRECT_HORIZ    
WAVE_CORRECT_CHOICES["no"] = None    
    
class PanoramaPointMapper:    
    def __init__(self, exclude_top_percent=0.15, exclude_bottom_percent=0.15):    
        self.cameras = None    
        self.warped_image_scale = None    
        self.corners = None    
        self.original_images = None    
        self.panorama = None    
        self.warper = None  
        self.exclude_top_percent = exclude_top_percent  
        self.exclude_bottom_percent = exclude_bottom_percent  
        self.feature_masks = []  # 存储创建的掩码  
        self.pano_ax = None  # 用于交互式显示  
            
    def create_watermark_mask(self, image_path):  
        """创建排除水印区域的掩码"""  
        img = cv.imread(image_path)  
        if img is None:  
            raise ValueError(f"无法读取图像: {image_path}")  
          
        height, width = img.shape[:2]  
          
        # 计算要排除的像素数  
        exclude_top_pixels = int(height * self.exclude_top_percent)  
        exclude_bottom_pixels = int(height * self.exclude_bottom_percent)  
          
        # 创建掩码（白色=检测特征，黑色=忽略）  
        mask = np.zeros((height, width), dtype=np.uint8)  
        mask[exclude_top_pixels:height-exclude_bottom_pixels, :] = 255  
          
        return mask  
      
    def stitch_and_extract_params(self, image_paths, use_masks=True):    
        """基于 stitching_detailed.py 的拼接流程，提取变换参数，支持水印排除"""    
        # 设置默认参数    
        args = SimpleNamespace(    
            work_megapix=0.6,    
            seam_megapix=0.1,    
            compose_megapix=-1,    
            features="sift",  # 改为SIFT以获得更好的特征检测  
            matcher="homography",    
            estimator="homography",    
            match_conf=None,    
            conf_thresh=0.2,  # 降低置信度阈值  
            ba="ray",    
            ba_refine_mask="xxxxx",    
            wave_correct="horiz",    
            warp="spherical",    
            blend="multiband",    
            blend_strength=5,    
            save_graph=None,    
            timelapse=None,    
            rangewidth=-1,    
            try_cuda=False,    
            expos_comp="gain",    
            expos_comp_nr_feeds=1,    
            expos_comp_block_size=32,    
            seam="gc_color"    
        )    
            
        # 保存原始图像    
        self.original_images = []    
        for path in image_paths:    
            img = cv.imread(path)    
            if img is None:    
                raise ValueError(f"无法读取图像: {path}")    
            self.original_images.append(img)    
          
        # 创建特征掩码  
        if use_masks:  
            print(f"创建特征掩码，排除上端{self.exclude_top_percent*100}%和下端{self.exclude_bottom_percent*100}%区域...")  
            self.feature_masks = []  
            for i, path in enumerate(image_paths):  
                mask = self.create_watermark_mask(path)  
                self.feature_masks.append(mask)  
                print(f"已创建掩码 {i+1}/{len(image_paths)}")  
            
        # 执行拼接并提取参数    
        self.panorama, self.cameras, self.warped_image_scale, self.corners = self._detailed_stitch(    
            image_paths, args, use_masks    
        )    
            
        return self.panorama    
        
    def _detailed_stitch(self, img_names, args, use_masks=True):    
        """修改后的 stitching_detailed.py 主函数，支持特征掩码"""    
        work_megapix = args.work_megapix    
        seam_megapix = args.seam_megapix    
        compose_megapix = args.compose_megapix    
        conf_thresh = args.conf_thresh    
        ba_refine_mask = args.ba_refine_mask    
        wave_correct = WAVE_CORRECT_CHOICES[args.wave_correct]    
        warp_type = args.warp    
        blend_type = args.blend    
        blend_strength = args.blend_strength    
            
        finder = FEATURES_FIND_CHOICES[args.features]()    
        seam_work_aspect = 1    
        full_img_sizes = []    
        features = []    
        images = []    
        is_work_scale_set = False    
        is_seam_scale_set = False    
        is_compose_scale_set = False    
            
        # 特征检测阶段 - 基于 stitching_detailed.py，添加掩码支持  
        for idx, name in enumerate(img_names):    
            full_img = cv.imread(name)    
            if full_img is None:    
                raise ValueError(f"无法读取图像: {name}")    
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))    
                
            if work_megapix < 0:    
                img = full_img    
                work_scale = 1    
                is_work_scale_set = True    
            else:    
                if is_work_scale_set is False:    
                    work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))    
                    is_work_scale_set = True    
                img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)    
                
            if is_seam_scale_set is False:    
                seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))    
                seam_work_aspect = seam_scale / work_scale    
                is_seam_scale_set = True    
              
            # 使用掩码进行特征检测  
            if use_masks and idx < len(self.feature_masks):  
                # 缩放掩码以匹配工作分辨率  
                mask = cv.resize(self.feature_masks[idx],   
                               (img.shape[1], img.shape[0]),   
                               interpolation=cv.INTER_NEAREST)  
                img_feat = cv.detail.computeImageFeatures2(finder, img, mask)  
                print(f"使用掩码检测图像 {idx+1} 的特征点: {len(img_feat.keypoints)} 个")  
            else:  
                img_feat = cv.detail.computeImageFeatures2(finder, img)  
                print(f"检测图像 {idx+1} 的特征点: {len(img_feat.keypoints)} 个")  
              
            features.append(img_feat)    
            img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)    
            images.append(img)    
    
        # 特征匹配  
        matcher = self._get_matcher(args)    
        p = matcher.apply2(features)    
        matcher.collectGarbage()    
    
        # 图像子集选择  
        indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)    
        img_subset = []    
        img_names_subset = []    
        full_img_sizes_subset = []    
        for i in range(len(indices)):    
            img_names_subset.append(img_names[indices[i]])    
            img_subset.append(images[indices[i]])    
            full_img_sizes_subset.append(full_img_sizes[indices[i]])    
        images = img_subset    
        img_names = img_names_subset    
        full_img_sizes = full_img_sizes_subset    
        num_images = len(img_names)    
            
        if num_images < 2:    
            raise ValueError("需要至少2张图像进行拼接")    
          
        print(f"成功匹配 {num_images} 张图像")  
    
        # 相机参数估计  
        estimator = ESTIMATOR_CHOICES[args.estimator]()    
        b, cameras = estimator.apply(features, p, None)    
        if not b:    
            raise ValueError("相机参数估计失败")    
        for cam in cameras:    
            cam.R = cam.R.astype(np.float32)    
    
        # 束调整  
        adjuster = BA_COST_CHOICES[args.ba]()    
        adjuster.setConfThresh(1)    
        refine_mask = np.zeros((3, 3), np.uint8)    
        if ba_refine_mask[0] == "x":    
            refine_mask[0, 0] = 1    
        if ba_refine_mask[1] == "x":    
            refine_mask[0, 1] = 1    
        if ba_refine_mask[2] == "x":    
            refine_mask[0, 2] = 1    
        if ba_refine_mask[3] == "x":    
            refine_mask[1, 1] = 1    
        if ba_refine_mask[4] == "x":    
            refine_mask[1, 2] = 1    
        adjuster.setRefinementMask(refine_mask)    
        b, cameras = adjuster.apply(features, p, cameras)    
        if not b:    
            raise ValueError("相机参数调整失败")    
    
        # 计算变形尺度  
        focals = []    
        for cam in cameras:    
            focals.append(cam.focal)    
        focals.sort()    
        if len(focals) % 2 == 1:    
            warped_image_scale = focals[len(focals) // 2]    
        else:    
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2    
    
        # 波形校正  
        if wave_correct is not None:    
            rmats = []    
            for cam in cameras:    
                rmats.append(np.copy(cam.R))    
            rmats = cv.detail.waveCorrect(rmats, wave_correct)    
            for idx, cam in enumerate(cameras):    
                cam.R = rmats[idx]    
    
        # 创建 warper 并进行变形  
        warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)    
        corners = []    
        for idx in range(0, num_images):    
            K = cameras[idx].K().astype(np.float32)    
            swa = seam_work_aspect    
            K[0, 0] *= swa    
            K[0, 2] *= swa    
            K[1, 1] *= swa    
            K[1, 2] *= swa    
            corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)    
            corners.append(corner)    
    
        # 执行完整的拼接流程获得最终全景图  
        panorama = self._complete_stitching(img_names, args)    
            
        return panorama, cameras, warped_image_scale, corners    
        
    def _get_matcher(self, args):    
        """复制 stitching_detailed.py 的 get_matcher 函数"""    
        try_cuda = args.try_cuda    
        matcher_type = args.matcher    
        if args.match_conf is None:    
            if args.features == "orb":    
                match_conf = 0.3    
            else:    
                match_conf = 0.65    
        else:    
            match_conf = args.match_conf    
        range_width = args.rangewidth    
            
        if matcher_type == "affine":    
            matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)    
        elif range_width == -1:    
            matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)    
        else:    
            matcher = cv.detail.BestOf2NearestRangeMatcher_create(range_width, try_cuda, match_conf)    
        return matcher
    def _complete_stitching(self, img_names, args):    
        """简化的拼接流程，避免依赖外部文件"""  
        print("执行简化的拼接流程...")  
            
        # 使用OpenStitching库进行简化拼接  
        try:  
            from stitching import Stitcher  
                
            # 创建拼接器配置  
            stitcher_config = {  
                "detector": args.features,  
                "confidence_threshold": args.conf_thresh,  
                "crop": False,  
                "warper_type": args.warp,  
                "blender_type": args.blend,  
                "blend_strength": args.blend_strength  
            }  
                
            stitcher = Stitcher(**stitcher_config)  
                
            # 如果有掩码，使用掩码进行拼接  
            if hasattr(self, 'feature_masks') and self.feature_masks:  
                # 保存掩码到临时文件  
                mask_paths = []  
                for i, mask in enumerate(self.feature_masks):  
                    mask_path = f"temp_mask_{i}.png"  
                    cv.imwrite(mask_path, mask)  
                    mask_paths.append(mask_path)  
                    
                panorama = stitcher.stitch(img_names, feature_masks=mask_paths)  
                    
                # 清理临时文件  
                for mask_path in mask_paths:  
                    if os.path.exists(mask_path):  
                        os.remove(mask_path)  
            else:  
                panorama = stitcher.stitch(img_names)  
                
            return panorama  
                
        except ImportError:  
            # 如果无法导入stitching库，返回第一张图像作为占位符  
            print("警告：无法导入stitching库，返回第一张图像")  
            return cv.imread(img_names[0])  
      
    def map_point_to_panorama(self, image_index, point_x, point_y):    
        """将原始图像中的点映射到全景图像"""    
        if self.cameras is None:    
            raise ValueError("请先调用 stitch_and_extract_params 方法")    
            
        # 获取相机参数    
        camera = self.cameras[image_index]    
        K = camera.K().astype(np.float32)    
        R = camera.R.astype(np.float32)    
            
        # 创建 warper（与拼接时使用相同的参数）    
        warper = cv.PyRotationWarper("spherical", self.warped_image_scale)    
            
        # 将点转换为齐次坐标    
        point_2d = np.array([[point_x, point_y]], dtype=np.float32)    
            
        # 使用 warper 进行坐标变换    
        mapped_point = warper.warpPoint(point_2d, K, R)    
            
        # 添加角点偏移    
        panorama_x = int(mapped_point[0][0] + self.corners[image_index][0])    
        panorama_y = int(mapped_point[0][1] + self.corners[image_index][1])    
            
        return panorama_x, panorama_y    
        
    def create_interactive_viewer(self):    
        """创建交互式查看器"""    
        if self.panorama is None:    
            raise ValueError("请先进行图像拼接")    
              
        # 计算布局    
        num_images = len(self.original_images)    
        cols = min(3, num_images)  # 最多3列    
        rows = (num_images + cols - 1) // cols + 1  # 原始图像行数 + 1行全景图    
              
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))    
        if rows == 1:    
            axes = [axes] if num_images == 1 else axes    
        elif cols == 1:    
            axes = [[ax] for ax in axes]    
              
        # 显示原始图像    
        for i, img in enumerate(self.original_images):    
            row = i // cols    
            col = i % cols    
            ax = axes[row][col] if rows > 1 else axes[col]    
                  
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))    
            ax.set_title(f'原始图像 {i+1} (点击选择点)')    
            ax.axis('off')    
                  
            # 添加点击事件    
            def make_click_handler(img_idx):    
                def on_click(event):    
                    if event.inaxes == ax and event.xdata and event.ydata:    
                        x, y = int(event.xdata), int(event.ydata)    
                        try:    
                            pano_x, pano_y = self.map_point_to_panorama(img_idx, x, y)    
                            self.show_point_on_panorama(pano_x, pano_y, img_idx)    
                            print(f"原始图像{img_idx+1}点({x},{y}) -> 全景图像点({pano_x},{pano_y})")    
                        except Exception as e:    
                            print(f"坐标变换失败: {e}")    
                return on_click    
                  
            fig.canvas.mpl_connect('button_press_event', make_click_handler(i))    
              
        # 隐藏多余的子图    
        for i in range(num_images, rows * cols - 1):    
            row = i // cols    
            col = i % cols    
            if rows > 1:    
                axes[row][col].axis('off')    
              
        # 显示全景图像（占据最后一行）    
        if rows > 1:    
            # 合并最后一行的所有子图来显示全景图    
            for col in range(cols):    
                axes[-1][col].axis('off')    
                  
            # 在最后一行创建一个大的子图    
            pano_ax = fig.add_subplot(rows, 1, rows)    
        else:    
            pano_ax = axes[-1] if num_images > 1 else axes[0]    
              
        pano_ax.imshow(cv.cvtColor(self.panorama, cv.COLOR_BGR2RGB))    
        pano_ax.set_title('全景图像 (点击映射结果显示在这里)')    
        pano_ax.axis('off')    
              
        self.pano_ax = pano_ax    
        plt.tight_layout()    
        plt.show()    
            
    def show_point_on_panorama(self, x, y, source_img_idx):    
        """在全景图像上显示对应点"""    
        # 清除之前的点    
        for patch in list(self.pano_ax.patches):    
            patch.remove()    
              
        # 添加新的点    
        circle = Circle((x, y), radius=15, color='red', fill=False, linewidth=3)    
        self.pano_ax.add_patch(circle)    
              
        # 添加文本标签    
        self.pano_ax.text(x + 20, y, f'来自图像{source_img_idx+1}',     
                            color='red', fontsize=12, fontweight='bold',    
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))    
              
        self.pano_ax.figure.canvas.draw()  
  
# 使用示例    
def main():    
    # 创建映射器，设置水印排除参数  
    mapper = PanoramaPointMapper(  
        exclude_top_percent=0.2,    # 排除顶部20%  
        exclude_bottom_percent=0.2  # 排除底部20%  
    )    
        
    # 图像路径    
    imgdir = 'imgs/722点位采集'  
    if not os.path.exists(imgdir):  
        print(f"目录 {imgdir} 不存在，请检查路径")  
        return  
          
    image_paths = [os.path.join(imgdir, p) for p in os.listdir(imgdir) if p.endswith('.bmp')]  
      
    if len(image_paths) < 2:  
        print(f"目录中至少需要2张图片，当前找到 {len(image_paths)} 张")  
        return  
  
    print(f"找到 {len(image_paths)} 张图片")  
      
    # 拼接图像并保存参数    
    print("开始图像拼接...")    
    try:  
        panorama = mapper.stitch_and_extract_params(image_paths, use_masks=True)    
        print("拼接完成！")    
          
        # 保存全景图  
        output_path = os.path.join(imgdir, "panorama_with_masks.jpg")  
        cv.imwrite(output_path, panorama)  
        print(f"全景图已保存到: {output_path}")  
              
        # 创建交互式查看器    
        print("创建交互式查看器...")    
        mapper.create_interactive_viewer()  
          
    except Exception as e:  
        print(f"拼接失败: {e}")  
        print("请检查图片质量和重叠度")  
    
if __name__ == "__main__":    
    main()    