from stitching.feature_detector import FeatureDetector  
from stitching.feature_matcher import RoMAFeatureMatcher  
from stitching import Stitcher  
from matching import get_matcher
import cv2 as cv

device = 'cuda'  # 'cpu'
roma_matcher = get_matcher('minima-roma', device=device)

detector = FeatureDetector()
matcher = RoMAFeatureMatcher(
    roma_matcher,
    detector,
    match_conf=0.5,  
    # range_width=1  # 只匹配相邻图像  
)  
  
                 # 可选：启用匹配图分析    
stitcher = Stitcher(confidence_threshold=0.2, matches_graph_dot_file="matches.txt", detector="sift")
stitcher.detector = detector  
stitcher.matcher = matcher  
output_path = "output_roma.jpg"
panorama = stitcher.stitch(["imgs/灯杆2一倍_frame_000012.jpg", "imgs/灯杆2一倍_frame_000013.jpg", "imgs/灯杆2一倍_frame_000014.jpg"])
print(f"stitching result saved to {output_path}")   
cv.imwrite(str(output_path), panorama)