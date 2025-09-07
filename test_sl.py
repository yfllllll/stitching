import os  
import cv2 as cv  
from pathlib import Path  
import time  
import tracemalloc  
from stitching import Stitcher  
from stitching.feature_detector import FeatureDetector  
from stitching.feature_matcher import FeatureMatcher  
  
def test_superpoint_lightglue_stitching(image_folder, superpoint_model_path, lightglue_model_path, output_folder=None):  
    """  
    测试 SuperPoint + LightGlue 的图像拼接功能  
      
    Args:  
        image_folder: 输入图像文件夹路径  
        superpoint_model_path: SuperPoint ONNX 模型文件路径  
        lightglue_model_path: LightGlue ONNX 模型文件路径  
        output_folder: 输出文件夹路径，默认为输入文件夹下的 results 子文件夹  
    """  
    # 设置输出文件夹  
    if output_folder is None:  
        output_folder = Path(image_folder) / "results"  
    output_folder = Path(output_folder)  
    output_folder.mkdir(exist_ok=True)  
      
    # 查找图像文件  
    image_folder = Path(image_folder)  
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']  
    image_files = []  
      
    for ext in image_extensions:  
        image_files.extend(image_folder.glob(ext))  
        image_files.extend(image_folder.glob(ext.upper()))  
      
    image_files = sorted([str(img) for img in image_files])  
      
    if len(image_files) < 2:  
        raise ValueError(f"需要至少2张图片进行拼接，当前找到 {len(image_files)} 张")  
      
    print(f"找到 {len(image_files)} 张图片:")  
    for i, img_path in enumerate(image_files, 1):  
        print(f"  {i}. {Path(img_path).name}")  
      
    # 验证模型文件存在  
    if not os.path.exists(superpoint_model_path):  
        raise FileNotFoundError(f"SuperPoint 模型文件不存在: {superpoint_model_path}")  
      
    if not os.path.exists(lightglue_model_path):  
        raise FileNotFoundError(f"LightGlue 模型文件不存在: {lightglue_model_path}")  
      
    print(f"\n使用模型:")  
    print(f"  SuperPoint: {superpoint_model_path}")  
    print(f"  LightGlue: {lightglue_model_path}")  
      

    # 手动创建检测器和匹配器以指定模型路径  
    print("\n创建 SuperPoint 特征检测器...")  
    detector = FeatureDetector("superpoint", model_path=superpoint_model_path)  
        
    print("创建 LightGlue 特征匹配器...")  
    matcher = FeatureMatcher("lightglue", model_path=lightglue_model_path)  
        
    # 创建拼接器配置（不包含检测器和匹配器相关参数）  
    stitcher_config = {  
        "confidence_threshold": 0.2,  
        "crop": False,  
        "medium_megapix": 0.6,  
        "low_megapix": 0.1,  
        "final_megapix": -1  
    }  
        
    # 创建拼接器  
    print("创建拼接器...")  
    stitcher = Stitcher(**stitcher_config)  
        
    # 替换默认的检测器和匹配器  
    stitcher.detector = detector  
    stitcher.matcher = matcher  
        
    # 开始性能监控  
    start_time = time.time()  
    tracemalloc.start()  
        
    # 执行拼接  
    print("\n开始拼接...")  
    panorama = stitcher.stitch(image_files)  
        
    # 获取性能指标  
    _, peak_memory = tracemalloc.get_traced_memory()  
    tracemalloc.stop()  
    end_time = time.time()  
        
    # 保存结果  
    output_path = output_folder / "panorama_superpoint_lightglue.jpg"  
    cv.imwrite(str(output_path), panorama)  
        
    # 输出结果信息  
    print(f"\n拼接成功！")  
    print(f"全景图保存在: {output_path}")  
    print(f"全景图尺寸: {panorama.shape}")  
    print(f"执行时间: {end_time - start_time:.2f} 秒")  
    print(f"峰值内存: {peak_memory / (1024 * 1024):.1f} MB")  
        
    return {  
        "success": True,  
        "output_path": str(output_path),  
        "shape": panorama.shape,  
        "execution_time": end_time - start_time,  
        "peak_memory_mb": peak_memory / (1024 * 1024)  
    }  
        

  
def test_with_verbose_mode(image_folder, superpoint_model_path, lightglue_model_path, output_folder=None):  
    """  
    使用详细模式测试，生成中间结果用于调试  
    """  
    if output_folder is None:  
        output_folder = Path(image_folder) / "verbose_results"  
    output_folder = Path(output_folder)  
    output_folder.mkdir(exist_ok=True)  
      
    # 查找图像文件  
    image_folder = Path(image_folder)  
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']  
    image_files = []  
      
    for ext in image_extensions:  
        image_files.extend(image_folder.glob(ext))  
        image_files.extend(image_folder.glob(ext.upper()))  
      
    image_files = sorted([str(img) for img in image_files])  
      
    if len(image_files) < 2:  
        raise ValueError(f"需要至少2张图片进行拼接，当前找到 {len(image_files)} 张")  
      
   
    # 创建检测器和匹配器  
    detector = FeatureDetector("superpoint", model_path=superpoint_model_path)  
    matcher = FeatureMatcher("lightglue", model_path=lightglue_model_path)  
        
    # 配置拼接器  
    stitcher_config = {  
        "confidence_threshold": 0.2,  
        "crop": False  
    }  
        
    stitcher = Stitcher(**stitcher_config)  
    stitcher.detector = detector  
    stitcher.matcher = matcher  
        
    print(f"使用详细模式拼接，结果保存在: {output_folder}")  
    panorama = stitcher.stitch_verbose(image_files, verbose_dir=str(output_folder))  
        
    print(f"详细模式拼接完成！")  
    print(f"检查 {output_folder} 文件夹查看中间结果")  
        
    return panorama  
          
 
  
def compare_with_traditional_methods(image_folder, superpoint_model_path, lightglue_model_path):  
    """  
    比较 SuperPoint+LightGlue 与传统方法的性能  
    """  
    print("\n=== 性能对比测试 ===")  
      
    # 查找图像文件  
    image_folder = Path(image_folder)  
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']  
    image_files = []  
      
    for ext in image_extensions:  
        image_files.extend(image_folder.glob(ext))  
        image_files.extend(image_folder.glob(ext.upper()))  
      
    image_files = sorted([str(img) for img in image_files])  
      
    results = {}  
      
    # 测试传统 ORB + Homography  
    print("\n1. 测试 ORB + Homography...")  
   
    start_time = time.time()  
    tracemalloc.start()  
        
    stitcher_orb = Stitcher(detector="orb", matcher_type="homography", confidence_threshold=0.3)  
    panorama_orb = stitcher_orb.stitch(image_files)  
        
    _, peak_memory = tracemalloc.get_traced_memory()  
    tracemalloc.stop()  
    end_time = time.time()  
        
    results["orb_homography"] = {  
        "success": True,  
        "execution_time": end_time - start_time,  
        "peak_memory_mb": peak_memory / (1024 * 1024),  
        "shape": panorama_orb.shape  
    }  
        
    # 保存结果  
    output_path = Path(image_folder) / "results" / "panorama_orb_homography.jpg"  
    output_path.parent.mkdir(exist_ok=True)  
    cv.imwrite(str(output_path), panorama_orb)  
        
    print(f"  ORB + Homography 完成: {end_time - start_time:.2f}s, {peak_memory / (1024 * 1024):.1f}MB")  
        

      
    # 测试 SIFT + Homography  
    print("\n2. 测试 SIFT + Homography...")  
    
    start_time = time.time()  
    tracemalloc.start()  
        
    stitcher_sift = Stitcher(detector="sift", matcher_type="homography", confidence_threshold=0.65)  
    panorama_sift = stitcher_sift.stitch(image_files)  
        
    _, peak_memory = tracemalloc.get_traced_memory()  
    tracemalloc.stop()  
    end_time = time.time()  
        
    results["sift_homography"] = {  
        "success": True,  
        "execution_time": end_time - start_time,  
        "peak_memory_mb": peak_memory / (1024 * 1024),  
        "shape": panorama_sift.shape  
    }  
        
    # 保存结果  
    output_path = Path(image_folder) / "results" / "panorama_sift_homography.jpg"  
    cv.imwrite(str(output_path), panorama_sift)  
        
    print(f"  SIFT + Homography 完成: {end_time - start_time:.2f}s, {peak_memory / (1024 * 1024):.1f}MB")  
        

      
    # 测试 SuperPoint + LightGlue  
    print("\n3. 测试 SuperPoint + LightGlue...")  
    result_sp_lg = test_superpoint_lightglue_stitching(  
        image_folder, superpoint_model_path, lightglue_model_path  
    )  
    results["superpoint_lightglue"] = result_sp_lg  
      
    if result_sp_lg["success"]:  
        print(f"  SuperPoint + LightGlue 完成: {result_sp_lg['execution_time']:.2f}s, {result_sp_lg['peak_memory_mb']:.1f}MB")  
    else:  
        print(f"  SuperPoint + LightGlue 失败: {result_sp_lg['error']}")  
      
    # 输出对比结果  
    print("\n=== 性能对比结果 ===")  
    for method, result in results.items():  
        if result["success"]:  
            print(f"{method:20}: {result['execution_time']:6.2f}s, {result['peak_memory_mb']:6.1f}MB, {result['shape']}")  
        else:  
            print(f"{method:20}: 失败 - {result['error']}")  
      
    return results  
  
def main():  
    """主函数"""  
    print("SuperPoint + LightGlue 图像拼接测试")  
    print("=" * 50)  
      
    # 获取用户输入  
    image_folder = "/root/myopencv-superpoint-main/test_imgs"  
    if not os.path.exists(image_folder):  
        print(f"文件夹不存在: {image_folder}")  
        return

    superpoint_model = "/root/weights/superpoint.onnx"
    if not os.path.exists(superpoint_model):
        print(f"SuperPoint 模型文件不存在: {superpoint_model}")
        return

    lightglue_model = "/root/weights/superpoint_lightglue.onnx"
    if not os.path.exists(lightglue_model):  
        print(f"LightGlue 模型文件不存在: {lightglue_model}")  
        return  
      
    # 选择测试模式  
    print("\n请选择测试模式:")  
    print("1. 基本拼接测试")  
    print("2. 详细模式测试（生成中间结果）")  
    print("3. 性能对比测试（与传统方法对比）")  
    print("4. 全部测试")  
      
    choice = input("请输入选择 (1-4): ").strip()  
      
 
    if choice in ["1", "4"]:  
        print("\n=== 基本拼接测试 ===")  
        result = test_superpoint_lightglue_stitching(  
            image_folder, superpoint_model, lightglue_model  
        )  
            
        if result["success"]:  
            print("基本测试完成！")  
        else:  
            print("基本测试失败！")  
        
    if choice in ["2", "4"]:  
        print("\n=== 详细模式测试 ===")  
        test_with_verbose_mode(  
            image_folder, superpoint_model, lightglue_model  
        )
    if choice in ["3", "4"]:  
        print("\n=== 性能对比测试 ===")  
        compare_with_traditional_methods(  
            image_folder, superpoint_model, lightglue_model  
        )  
        
    print("\n所有测试完成!")  
          

  
if __name__ == "__main__":  
    main()