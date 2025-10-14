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
        imgs = self.detector.images  
        num_images = len(imgs)  
        pairwise_matches = []  
        
        for i in range(num_images):  
            for j in range(num_images):  
                match_info = cv.detail.MatchesInfo()  
                
                if i == j:  
                    match_info.confidence = 0  
                    match_info.num_inliers = 0  
                elif i < j and (self.range_width == -1 or abs(i-j) <= self.range_width):  
                    # 使用 RoMA 计算单应性矩阵  
                    result = self._compute_matches_roma(imgs[i], imgs[j])  
                    
                    if result and 'H' in result and result['H'] is not None:  
                        H = result['H']  
                        
                        # 获取原始特征点  
                        kpts_i = features[i].getKeypoints()  
                        kpts_j = features[j].getKeypoints()  
                        
                        # 使用 set 确保一对一匹配  
                        matched_pairs = set()  
                        
                        # 正向匹配: i -> j  
                        pts_i = np.float32([kp.pt for kp in kpts_i]).reshape(-1, 1, 2)  
                        pts_j_projected = cv.perspectiveTransform(pts_i, H)  
                        pts_j = np.float32([kp.pt for kp in kpts_j])  
                        
                        reproj_threshold = 5.0  
                        
                        for idx_i, pt_proj in enumerate(pts_j_projected.reshape(-1, 2)):  
                            distances = np.linalg.norm(pts_j - pt_proj, axis=1)  
                            min_idx = np.argmin(distances)  
                            
                            if distances[min_idx] < reproj_threshold:  
                                matched_pairs.add((idx_i, min_idx))  
                        
                        # 反向匹配: j -> i (可选,用于双向验证)  
                        if result['H'] is not None:  
                            H_inv = np.linalg.inv(H)  
                            pts_j_reshaped = pts_j.reshape(-1, 1, 2)  
                            pts_i_projected = cv.perspectiveTransform(pts_j_reshaped, H_inv)  
                            pts_i_orig = np.float32([kp.pt for kp in kpts_i])  
                            
                            reverse_matches = set()  
                            for idx_j, pt_proj in enumerate(pts_i_projected.reshape(-1, 2)):  
                                distances = np.linalg.norm(pts_i_orig - pt_proj, axis=1)  
                                min_idx = np.argmin(distances)  
                                
                                if distances[min_idx] < reproj_threshold:  
                                    reverse_matches.add((min_idx, idx_j))  
                            
                            # 只保留双向一致的匹配  
                            matched_pairs = matched_pairs.intersection(reverse_matches)  
                        
                        # 构建 DMatch 对象  
                        matches = []  
                        for idx_i, idx_j in matched_pairs:  
                            dmatch = cv.DMatch()  
                            dmatch.queryIdx = idx_i  
                            dmatch.trainIdx = idx_j  
                            dmatch.distance = 0  # 或使用重投影误差  
                            matches.append(dmatch)  
                        
                        match_info.matches = matches  
                        match_info.num_inliers = len(matches)  
                        match_info.confidence = min(len(matches) / 50.0, 1.0)  
                        match_info.H = H  
                    else:  
                        match_info.confidence = 0  
                        match_info.num_inliers = 0  
                else:  
                    match_info.confidence = 0  
                    match_info.num_inliers = 0  
                
                match_info.src_img_idx = i  
                match_info.dst_img_idx = j  
                pairwise_matches.append(match_info)  
        
        return pairwise_matches
      
    def _compute_matches_roma(self, img_a, img_b):  
        """使用 RoMA 进行匹配,并处理坐标中心化"""  
        # # 保存原始尺寸  
        # orig_h_a, orig_w_a = img_a.shape[:2]  
        # orig_h_b, orig_w_b = img_b.shape[:2]  
        
        # scale_a = max(orig_h_a, orig_w_a) / self.img_size  
        # scale_b = max(orig_h_b, orig_w_b) / self.img_size  
        
        # img_a_resized = cv.resize(img_a, None, fx=1/scale_a, fy=1/scale_a,   
        #                         interpolation=cv.INTER_LINEAR)  
        # img_b_resized = cv.resize(img_b, None, fx=1/scale_b, fy=1/scale_b,   
        #                         interpolation=cv.INTER_LINEAR)  
        
        # 执行匹配  
        result = self.roma_matcher(  
            self.roma_matcher.load_image(img_a),   
            self.roma_matcher.load_image(img_b)  
        )  
        
        # # 缩放关键点坐标到原始分辨率  
        # if 'inlier_kpts0' in result:  
        #     result['inlier_kpts0'] = self.roma_matcher.rescale_coords(  
        #         result['inlier_kpts0'], orig_h_a, orig_w_a, self.img_size, self.img_size  
        #     )  
        # if 'inlier_kpts1' in result:  
        #     result['inlier_kpts1'] = self.roma_matcher.rescale_coords(  
        #         result['inlier_kpts1'], orig_h_b, orig_w_b, self.img_size, self.img_size  
        #     )  
        

        
        return result