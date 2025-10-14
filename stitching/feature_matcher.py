import math

import cv2 as cv
import numpy as np


class FeatureMatcher:
    """https://docs.opencv.org/4.x/da/d87/classcv_1_1detail_1_1FeaturesMatcher.html"""

    MATCHER_CHOICES = ("homography", "affine", "lightglue")
    DEFAULT_MATCHER = "homography"
    DEFAULT_RANGE_WIDTH = -1

    def __init__(
        self,
        matcher_type=DEFAULT_MATCHER,
        range_width=DEFAULT_RANGE_WIDTH,
        model_path="lightglue.onnx",
        mode=cv.Stitcher_PANORAMA, # its value only can be cv.Stitcher_PANORAMA or cv.Stitcher_SCANS
        match_thresh=0.7,
        **kwargs,
    ):
        if matcher_type == "affine":
            self.matcher = cv.detail_AffineBestOf2NearestMatcher(**kwargs)
        elif matcher_type == "lightglue":
            self.matcher = cv.detail.LightGlue.create(model_path, mode, match_thresh, range_width)
        elif range_width == -1:
            self.matcher = cv.detail_BestOf2NearestMatcher(**kwargs)
        else:
            self.matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, **kwargs)

    def match_features(self, features, *args, **kwargs):
        pairwise_matches = self.matcher.apply2(features, *args, **kwargs)
        self.matcher.collectGarbage()
        return pairwise_matches

    @staticmethod
    def draw_matches_matrix(
        imgs, features, matches, conf_thresh=1, inliers=False, **kwargs
    ):
        matches_matrix = FeatureMatcher.get_matches_matrix(matches)
        for idx1, idx2 in FeatureMatcher.get_all_img_combinations(len(imgs)):
            match = matches_matrix[idx1, idx2]
            if match.confidence < conf_thresh or len(match.matches) == 0:
                continue
            if inliers:
                kwargs["matchesMask"] = match.getInliers()
            yield idx1, idx2, FeatureMatcher.draw_matches(
                imgs[idx1], features[idx1], imgs[idx2], features[idx2], match, **kwargs
            )

    @staticmethod
    def draw_matches(img1, features1, img2, features2, match1to2, **kwargs):
        kwargs.setdefault("flags", cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        keypoints1 = features1.getKeypoints()
        keypoints2 = features2.getKeypoints()
        matches = match1to2.getMatches()

        return cv.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None, **kwargs
        )

    @staticmethod
    def get_matches_matrix(pairwise_matches):
        return FeatureMatcher.array_in_square_matrix(pairwise_matches)

    @staticmethod
    def get_confidence_matrix(pairwise_matches):
        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        match_confs = [[m.confidence for m in row] for row in matches_matrix]
        match_conf_matrix = np.array(match_confs)
        return match_conf_matrix

    @staticmethod
    def array_in_square_matrix(array):
        matrix_dimension = int(math.sqrt(len(array)))
        rows = []
        for i in range(0, len(array), matrix_dimension):
            rows.append(array[i : i + matrix_dimension])
        return np.array(rows)

    def get_all_img_combinations(number_imgs):
        ii, jj = np.triu_indices(number_imgs, k=1)
        for i, j in zip(ii, jj):
            yield i, j

    @staticmethod
    def get_match_conf(match_conf, feature_detector_type):
        if match_conf is None:
            match_conf = FeatureMatcher.get_default_match_conf(feature_detector_type)
        return match_conf

    @staticmethod
    def get_default_match_conf(feature_detector_type):
        if feature_detector_type == "orb":
            return 0.3
        return 0.65


class RoMAFeatureMatcher:  
    def __init__(self, roma_matcher, detector, match_conf=0.5, img_size=1024, range_width=-1):  
        self.roma_matcher = roma_matcher  
        self.detector = detector  
        self.match_conf = match_conf  
        self.img_size = img_size  
        self.range_width = range_width  
          
    def match_features(self, features):  
        """使用 RoMA 进行特征匹配,支持 range_width"""  
        imgs = self.detector.images  
        num_images = len(imgs)  
        pairwise_matches = []  
          
        # 先创建一个字典存储已计算的匹配  
        computed_matches = {}  
          
        for i in range(num_images):  
            for j in range(num_images):  
                match_info = cv.detail.MatchesInfo()  
                  
                if i == j:  
                    # 同一张图像  
                    match_info.confidence = 0  
                    match_info.num_inliers = 0  
                    match_info.src_img_idx = i  
                    match_info.dst_img_idx = j  
                elif i < j:  
                    # 检查是否在 range_width 范围内  
                    if self.range_width != -1 and abs(i - j) > self.range_width:  
                        match_info.confidence = 0  
                        match_info.num_inliers = 0  
                    else:  
                        # 计算匹配  
                        result = self._compute_matches_roma(imgs[i], imgs[j])  
                          
                        if result and 'inlier_kpts0' in result and len(result['inlier_kpts0']) > 0:  
                            kpts0 = result['inlier_kpts0']  
                            kpts1 = result['inlier_kpts1']  
                              
                            matches = []  
                            for idx in range(len(kpts0)):  
                                dmatch = cv.DMatch()  
                                dmatch.queryIdx = idx  
                                dmatch.trainIdx = idx  
                                dmatch.distance = 0  
                                matches.append(dmatch)  
                              
                            match_info.matches = matches  
                            match_info.num_inliers = len(matches)  
                            match_info.confidence = min(len(matches) / 50.0, 1.0)  
                              
                            if 'H' in result and result['H'] is not None:  
                                match_info.H = result['H']  
                              
                            # 更新 features  
                            if not hasattr(features[i], 'keypoints') or len(features[i].keypoints) == 0:  
                                features[i].keypoints = [cv.KeyPoint(x=pt[0], y=pt[1], size=1)   
                                                        for pt in kpts0]  
                            if not hasattr(features[j], 'keypoints') or len(features[j].keypoints) == 0:  
                                features[j].keypoints = [cv.KeyPoint(x=pt[0], y=pt[1], size=1)   
                                                        for pt in kpts1]  
                        else:  
                            match_info.confidence = 0  
                            match_info.num_inliers = 0  
                      
                    match_info.src_img_idx = i  
                    match_info.dst_img_idx = j  
                    # 保存这个匹配,供对称位置使用  
                    computed_matches[(i, j)] = match_info  
                else:  
                    # i > j: 创建对称匹配  
                    if (j, i) in computed_matches:  
                        original = computed_matches[(j, i)]  
                          
                        # 复制匹配信息  
                        match_info.confidence = original.confidence  
                        match_info.num_inliers = original.num_inliers  
                        match_info.src_img_idx = i  
                        match_info.dst_img_idx = j  
                          
                        # 交换 queryIdx 和 trainIdx  
                        matches = []   
                        for m in original.matches:  
                            dmatch = cv.DMatch()  
                            dmatch.queryIdx = m.trainIdx  
                            dmatch.trainIdx = m.queryIdx  
                            dmatch.distance = m.distance  
                            matches.append(dmatch) 
                        match_info.matches = matches  
                        # 求逆单应性矩阵  
                        if original.H is not None:  
                            match_info.H = np.linalg.inv(original.H)  
                    else:  
                        match_info.confidence = 0  
                        match_info.num_inliers = 0  
                        match_info.src_img_idx = i  
                        match_info.dst_img_idx = j  
                  
                pairwise_matches.append(match_info)  
          
        return pairwise_matches
      
    def _compute_matches_roma(self, img_a, img_b):  
        """使用 RoMA 进行匹配,并处理坐标中心化"""  
        # 保存原始尺寸  
        orig_h_a, orig_w_a = img_a.shape[:2]  
        orig_h_b, orig_w_b = img_b.shape[:2]  
        
        scale_a = max(orig_h_a, orig_w_a) / self.img_size  
        scale_b = max(orig_h_b, orig_w_b) / self.img_size  
        
        img_a_resized = cv.resize(img_a, None, fx=1/scale_a, fy=1/scale_a,   
                                interpolation=cv.INTER_LINEAR)  
        img_b_resized = cv.resize(img_b, None, fx=1/scale_b, fy=1/scale_b,   
                                interpolation=cv.INTER_LINEAR)  
        
        # 执行匹配  
        result = self.roma_matcher(  
            self.roma_matcher.load_image(img_a_resized),   
            self.roma_matcher.load_image(img_b_resized)  
        )  
        
        # 缩放关键点坐标到原始分辨率  
        if 'inlier_kpts0' in result:  
            result['inlier_kpts0'] = self.roma_matcher.rescale_coords(  
                result['inlier_kpts0'], orig_h_a, orig_w_a, self.img_size, self.img_size  
            )  
        if 'inlier_kpts1' in result:  
            result['inlier_kpts1'] = self.roma_matcher.rescale_coords(  
                result['inlier_kpts1'], orig_h_b, orig_w_b, self.img_size, self.img_size  
            )  
        
        # 如果 RoMA 提供了 H,需要重新估计以匹配 OpenCV 的中心化坐标系统  
        if 'inlier_kpts0' in result and 'inlier_kpts1' in result and len(result['inlier_kpts0']) >= 4:  
            kpts0 = result['inlier_kpts0']  
            kpts1 = result['inlier_kpts1']  
            
            # 中心化坐标  
            src_points = np.array([[pt[0] - orig_w_a * 0.5, pt[1] - orig_h_a * 0.5] for pt in kpts0], dtype=np.float32)  
            dst_points = np.array([[pt[0] - orig_w_b * 0.5, pt[1] - orig_h_b * 0.5] for pt in kpts1], dtype=np.float32)  
            
            # 使用中心化的坐标重新估计单应性矩阵  
            H, _ = cv.findHomography(src_points, dst_points, cv.RANSAC)  
            if H is not None:  
                result['H'] = H  
        
        return result