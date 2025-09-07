import os    
import cv2 as cv    
import numpy as np    
from datetime import datetime    
from pathlib import Path    
from typing import List, Optional, Tuple    
from stitching import Stitcher  
from stitching.feature_detector import FeatureDetector  
from stitching.feature_matcher import FeatureMatcher  
    
class ImageStitchingOptimizer:    
    """优化的图像拼接器，支持水印排除和性能优化，现已支持 SuperPoint + LightGlue"""    
        
    def __init__(self,     
                 detector: str = "sift",    
                 matcher_type: str = "homography",  
                 confidence_threshold: float = 0.2,    
                 exclude_top_percent: float = 0.15,    
                 exclude_bottom_percent: float = 0.15,  
                 superpoint_model_path: str = "superpoint.onnx",  
                 lightglue_model_path: str = "lightglue.onnx"):    
        self.detector = detector    
        self.matcher_type = matcher_type  
        self.confidence_threshold = confidence_threshold    
        self.exclude_top_percent = exclude_top_percent    
        self.exclude_bottom_percent = exclude_bottom_percent  
        self.superpoint_model_path = superpoint_model_path  
        self.lightglue_model_path = lightglue_model_path  
            
        # 基于测试框架的最佳配置    
        self.stitcher_config = {    
                "detector": detector,  
                "matcher_type": matcher_type,    
                "confidence_threshold": confidence_threshold,    
                "crop": False,    
                "nfeatures": 2000,  # 增加特征点    
                "medium_megapix": 0.6,    
                "low_megapix": 0.1,    
                "final_megapix": -1,    
                "matches_graph_dot_file": None  # 可选：启用匹配图分析    
            }  
        
    def find_images(self, folder_path: str) -> List[str]:    
        """查找文件夹中的图像文件，使用 Images.resolve_wildcards 的逻辑"""    
        folder = Path(folder_path)    
        if not folder.exists():    
            raise ValueError(f"文件夹不存在: {folder_path}")    
            
        # 支持的图像格式    
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']    
        images = []    
            
        for ext in extensions:    
            pattern = folder / ext    
            images.extend(folder.glob(ext.lower()))    
            images.extend(folder.glob(ext.upper()))    
            
        # 排序确保一致性    
        images = sorted([str(img) for img in images])    
            
        if len(images) < 2:    
            raise ValueError(f"文件夹 {folder_path} 中至少需要2张图片进行拼接，当前找到 {len(images)} 张")    
            
        return images    
        
    def create_optimized_mask(self, image_path: str) -> np.ndarray:    
        """创建优化的掩码，基于 FeatureDetector.detect_with_masks 的要求"""    
        img = cv.imread(image_path)    
        if img is None:    
            raise ValueError(f"无法读取图像: {image_path}")    
            
        height, width = img.shape[:2]    
            
        # 计算排除区域    
        exclude_top_pixels = int(height * self.exclude_top_percent)    
        exclude_bottom_pixels = int(height * self.exclude_bottom_percent)    
            
        # 创建二值掩码 (白色=255表示检测区域，黑色=0表示忽略区域)    
        mask = np.zeros((height, width), dtype=np.uint8)    
        mask[exclude_top_pixels:height-exclude_bottom_pixels, :] = 255    
            
        return mask    
        
    def validate_masks(self, images: List[str], masks: List[str]) -> None:    
        """验证掩码与图像的兼容性，基于 FeatureDetector.detect_with_masks 的验证逻辑"""    
        if len(images) != len(masks):    
            raise ValueError("图像和掩码列表长度必须相同")    
            
        for idx, (img_path, mask_path) in enumerate(zip(images, masks)):    
            img = cv.imread(img_path)    
            mask = cv.imread(mask_path, 0)  # 灰度模式读取    
                
            if img is None:    
                raise ValueError(f"无法读取图像 {idx + 1}: {img_path}")    
            if mask is None:    
                raise ValueError(f"无法读取掩码 {idx + 1}: {mask_path}")    
                
            if not np.array_equal(img.shape[:2], mask.shape):    
                raise ValueError(    
                    f"掩码 {idx + 1} 的分辨率 {mask.shape} 与图像 {idx + 1} 的分辨率 {img.shape[:2]} 不匹配"    
                )  
  
    def create_custom_stitcher(self) -> Stitcher:  
        """创建自定义拼接器，支持 SuperPoint 和 LightGlue"""  
        # 验证模型文件（如果使用深度学习方法）  
        if self.detector == "superpoint":  
            if not os.path.exists(self.superpoint_model_path):  
                raise FileNotFoundError(f"SuperPoint 模型文件不存在: {self.superpoint_model_path}")  
          
        if self.matcher_type == "lightglue":  
            if not os.path.exists(self.lightglue_model_path):  
                raise FileNotFoundError(f"LightGlue 模型文件不存在: {self.lightglue_model_path}")  
          
        # 创建基础拼接器配置（排除检测器和匹配器参数）  
        base_config = {k: v for k, v in self.stitcher_config.items()   
                      if k not in ["detector", "matcher_type"]}  
          
        # 创建拼接器  
        stitcher = Stitcher(**base_config)  
          
        # 手动创建并设置检测器  
        if self.detector == "superpoint":  
            print(f"使用 SuperPoint 检测器，模型路径: {self.superpoint_model_path}")  
            stitcher.detector = FeatureDetector("superpoint", model_path=self.superpoint_model_path)  
        else:  
            # 传统检测器  
            if self.detector in ("orb", "sift"):  
                stitcher.detector = FeatureDetector(self.detector, nfeatures=self.stitcher_config.get("nfeatures", 2000))  
            else:  
                stitcher.detector = FeatureDetector(self.detector)  
          
        # 手动创建并设置匹配器  
        if self.matcher_type == "lightglue":  
            print(f"使用 LightGlue 匹配器，模型路径: {self.lightglue_model_path}")  
            stitcher.matcher = FeatureMatcher("lightglue", model_path=self.lightglue_model_path)  
        else:  
            # 传统匹配器  
            match_conf = FeatureMatcher.get_match_conf(None, self.detector)  
            stitcher.matcher = FeatureMatcher(  
                self.matcher_type,  
                range_width=FeatureMatcher.DEFAULT_RANGE_WIDTH,  
                match_conf=match_conf  
            )  
          
        return stitcher  
        
    def stitch_with_performance_monitoring(self,     
                                         images: List[str],     
                                         masks: Optional[List[str]] = None,    
                                         verbose: bool = False,    
                                         output_dir: Optional[str] = None) -> Tuple[np.ndarray, dict]:    
        """执行拼接并监控性能，基于 TestPerformance 的监控方法"""    
        import time    
        import tracemalloc    
            
        # 开始性能监控    
        start_time = time.time()    
        tracemalloc.start()    
            
        try:    
            # 创建自定义拼接器（支持 SuperPoint + LightGlue）  
            stitcher = self.create_custom_stitcher()  
                
            # 执行拼接    
            if verbose and output_dir:    
                os.makedirs(output_dir, exist_ok=True)    
                panorama = stitcher.stitch_verbose(images, masks or [], output_dir)    
            else:    
                panorama = stitcher.stitch(images, masks or [])    
                
            # 获取性能指标    
            _, peak_memory = tracemalloc.get_traced_memory()    
            tracemalloc.stop()    
            end_time = time.time()    
                
            performance_stats = {    
                "execution_time": end_time - start_time,    
                "peak_memory_mb": peak_memory / (1024 * 1024),    
                "output_shape": panorama.shape,    
                "num_images": len(images)    
            }    
                
            return panorama, performance_stats    
                
        except Exception as e:    
            tracemalloc.stop()    
            raise e    
        
    def process_folder(self,     
                      folder_path: str,     
                      verbose: bool = False,    
                      save_masks: bool = True) -> dict:    
        """处理整个文件夹的图像拼接"""    
        print(f"开始处理文件夹: {folder_path}")  
        print(f"使用检测器: {self.detector}")  
        print(f"使用匹配器: {self.matcher_type}")  
            
        # 查找图像    
        images = self.find_images(folder_path)    
        print(f"找到 {len(images)} 张图片")    
        print(f"将排除上端 {self.exclude_top_percent*100}% 和下端 {self.exclude_bottom_percent*100}% 的区域")    
            
        # 创建输出目录    
        output_dir = Path(folder_path) / "stitching_results"    
        output_dir.mkdir(exist_ok=True)    
            
        # 创建掩码    
        masks = []    
        if save_masks:    
            mask_dir = output_dir / "masks"    
            mask_dir.mkdir(exist_ok=True)    
                
            for i, img_path in enumerate(images):    
                mask = self.create_optimized_mask(img_path)    
                mask_path = mask_dir / f"mask_{i+1:03d}.png"    
                cv.imwrite(str(mask_path), mask)    
                masks.append(str(mask_path))    
                print(f"已创建掩码 {i+1}/{len(images)}: {mask_path.name}")    
            
        # 验证掩码    
        if masks:    
            self.validate_masks(images, masks)    
            print("掩码验证通过")    
            
        # 设置详细输出目录    
        verbose_dir = None    
        if verbose:    
            verbose_dir = str(output_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}")    
            
        # 执行拼接    
        print("开始拼接...")    
        try:    
            panorama, stats = self.stitch_with_performance_monitoring(    
                images, masks, verbose, verbose_dir    
            )    
                
            # 保存结果（根据使用的方法命名）  
            method_suffix = f"{self.detector}_{self.matcher_type}"  
            output_path = output_dir / f"panorama_{method_suffix}.jpg"  
            cv.imwrite(str(output_path), panorama)    
                
            # 输出结果信息    
            print(f"\n拼接成功！")    
            print(f"结果保存在: {output_path}")    
            print(f"图像尺寸: {stats['output_shape']}")    
            print(f"执行时间: {stats['execution_time']:.2f} 秒")    
            print(f"峰值内存: {stats['peak_memory_mb']:.1f} MB")    
                
            if verbose_dir:    
                print(f"调试信息保存在: {verbose_dir}")    
                
            return {    
                "success": True,    
                "output_path": str(output_path),    
                "performance": stats,    
                "masks_created": len(masks)    
            }    
                
        except Exception as e:    
            print(f"拼接失败: {e}")    
            return {    
                "success": False,    
                "error": str(e),    
                "masks_created": len(masks)    
            }    
    
def main():    
    """主函数"""    
    print("OpenStitching 优化拼接工具 (支持 SuperPoint + LightGlue)")    
    print("=" * 50)    
  
    # 获取用户输入  
    folder_path = input("请输入图像文件夹路径 [默认: imgs]: ").strip() or 'imgs'  
  
    # 配置选项  
    print("\n配置选项:")  
    print("支持的检测器: sift, orb, superpoint, akaze, brisk")  
    detector = input("特征检测器 [默认: superpoint]: ").strip() or "superpoint"  
      
    print("支持的匹配器: homography, affine, lightglue")  
    matcher_type = input("特征匹配器 [默认: lightglue]: ").strip() or "homography"  
      
    confidence = float(input("置信度阈值 (0.1-1.0) [默认: 0.1]: ").strip() or "0.1")    
    exclude_top = float(input("排除顶部百分比 (0.0-0.5) [默认: 0.15]: ").strip() or "0.15") 
    exclude_bottom = float(input("排除底部百分比 (0.0-0.5) [默认: 0.15]: ").strip() or "0.15")    
    verbose = input("启用详细输出? (y/n) [默认: n]: ").strip().lower() == 'y'    
    save_masks = input("保存掩码? (y/n) [默认: y]: ").strip().lower() == 'y'  
      
    # 模型路径配置（仅在使用深度学习方法时需要）  
    superpoint_model_path = "/root/weights/superpoint.onnx"  
    lightglue_model_path = "/root/weights/superpoint_lightglue.onnx"  
      
    if detector == "superpoint":  
        superpoint_model_path = input(f"请输入 SuperPoint 模型路径 [默认: {superpoint_model_path}]: ").strip() or superpoint_model_path  
          
    if matcher_type == "lightglue":  
        lightglue_model_path = input(f"请输入 LightGlue 模型路径 [默认: {lightglue_model_path}]: ").strip() or lightglue_model_path  
      
    # 创建优化器并处理    
    optimizer = ImageStitchingOptimizer(    
        detector=detector,  
        matcher_type=matcher_type,  
        confidence_threshold=confidence,    
        exclude_top_percent=exclude_top,    
        exclude_bottom_percent=exclude_bottom,  
        superpoint_model_path=superpoint_model_path,  
        lightglue_model_path=lightglue_model_path  
    )    
        
    result = optimizer.process_folder(folder_path, verbose=verbose, save_masks=save_masks)    
        
    if result["success"]:    
        print(f"\n处理完成！共创建 {result['masks_created']} 个掩码")    
        print(f"使用方法: {detector} + {matcher_type}")  
    else:    
        print(f"\n处理失败: {result['error']}")    
    
if __name__ == "__main__":    
    main()
