import os  
import cv2 as cv  
import numpy as np  
from datetime import datetime  
from pathlib import Path  
from typing import List, Optional, Tuple  
from stitching import Stitcher  
  
class ImageStitchingOptimizer:  
    """优化的图像拼接器，支持水印排除和性能优化"""  
      
    def __init__(self,   
                 detector: str = "sift",  
                 confidence_threshold: float = 0.2,  
                 exclude_top_percent: float = 0.15,  
                 exclude_bottom_percent: float = 0.15):  
        self.detector = detector  
        self.confidence_threshold = confidence_threshold  
        self.exclude_top_percent = exclude_top_percent  
        self.exclude_bottom_percent = exclude_bottom_percent  
          
        # 基于测试框架的最佳配置  
        self.stitcher_config = {  
                "detector": detector,  
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
            # 创建拼接器  
            stitcher = Stitcher(**self.stitcher_config)  
              
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
              
            # 保存结果  
            output_path = output_dir / "panorama_optimized.jpg"  
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
    print("OpenStitching 优化拼接工具")  
    print("=" * 40)  

    # 获取用户输入
    folder_path = 'imgs/722点位采集'

    # 配置选项
    print("\n配置选项:")
    detector = input("特征检测器 (sift/orb) [默认: orb]: ").strip() or "orb"
    confidence = float(input("置信度阈值 (0.1-1.0) [默认: 0.1]: ").strip() or "0.1")  
    exclude_top = float(input("排除顶部百分比 (0.0-0.5) [默认: 0.15]: ").strip() or "0.15")  
    exclude_bottom = float(input("排除底部百分比 (0.0-0.5) [默认: 0.15]: ").strip() or "0.15")  
    verbose = input("启用详细输出? (y/n) [默认: n]: ").strip().lower() == 'y'  
    save_masks = input("保存掩码? (y/n) [默认: y]: ").strip().lower() == 'y'
    
    # 创建优化器并处理  
    optimizer = ImageStitchingOptimizer(  
        detector=detector,  
        confidence_threshold=confidence,  
        exclude_top_percent=exclude_top,  
        exclude_bottom_percent=exclude_bottom  
    )  
      
    result = optimizer.process_folder(folder_path, verbose=verbose)  
      
    if result["success"]:  
        print(f"\n处理完成！共创建 {result['masks_created']} 个掩码")  
    else:  
        print(f"\n处理失败: {result['error']}")  
  
if __name__ == "__main__":  
    main()