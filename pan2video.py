import cv2
import numpy as np
import os

def crop_to_max_inner_rect(image_path):
    """
    自动检测图像内容，并裁剪出其中最大的内接矩形，彻底去除黑边。

    参数:
    image_path (str): 输入图像的文件路径。

    返回:
    numpy.ndarray: 裁剪后的图像。
    """
    print(f"正在加载图像: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"错误: 无法读取图像文件 {image_path}")
        return None

    # --- 1. 创建二值掩码 ---
    # 与之前方法相同，先得到内容区域的二值图 (内容为白色，黑边为黑色)
    if img.shape[2] == 4:
        bgr = img[:, :, 0:3]
    else:
        bgr = img
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8) # 确保掩码是8位整型

    # --- 2. 寻找最大内接矩形 ---
    # 这是一个更高级的算法，它能保证找到的矩形完全不包含黑色像素
    print("正在计算最大内接矩形以彻底去除黑边...")
    best_rect = (0, 0, 0, 0) # x, y, w, h
    max_area = 0

    # 我们逐行扫描，计算一个高度图（直方图）
    # 然后在每一行的直方图中寻找最大矩形面积
    hist = np.zeros(mask.shape[1], dtype=int)
    for i in range(mask.shape[0]):
        # 更新直方图
        for j in range(mask.shape[1]):
            hist[j] = hist[j] + 1 if mask[i, j] else 0

        # 在当前直方图中寻找最大矩形
        stack = [-1]
        for k, h in enumerate(hist):
            while stack[-1] != -1 and hist[stack[-1]] >= h:
                height = hist[stack.pop()]
                width = k - stack[-1] - 1
                area = height * width
                if area > max_area:
                    max_area = area
                    best_rect = (stack[-1] + 1, i - height + 1, width, height)
            stack.append(k)

        while stack[-1] != -1:
            height = hist[stack.pop()]
            width = len(hist) - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                best_rect = (stack[-1] + 1, i - height + 1, width, height)

    if max_area == 0:
        print("警告: 未找到有效内容区域，返回原始图像。")
        return img

    x, y, w, h = best_rect
    cropped_image = img[y:y+h, x:x+w]

    print(f"裁剪完成。新尺寸: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    return cropped_image

def create_panning_video(image, output_path='panorama_video.mp4', slice_width=1280, fps=30, step=10):
    """
    从一张宽幅图像创建平移扫描效果的视频。(此函数与之前版本相同)
    """
    if image is None:
        print("输入图像为空，无法创建视频。")
        return

    print("开始创建视频...")
    img_height, img_width = image.shape[:2]

    if slice_width > img_width:
        print(f"警告: 切片宽度 ({slice_width}) 大于图像宽度 ({img_width})。将使用图像宽度作为切片宽度。")
        slice_width = img_width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (slice_width, img_height))

    if not video_writer.isOpened():
        print(f"错误: 无法打开视频文件进行写入: {output_path}")
        return

    num_frames = 0
    for x in range(0, img_width - slice_width + 1, step):
        frame = image[:, x:x+slice_width]
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        video_writer.write(frame)
        num_frames += 1

    video_writer.release()
    print(f"视频创建成功！共写入 {num_frames} 帧。")
    print(f"视频已保存至: {os.path.abspath(output_path)}")


# --- 主程序 ---
if __name__ == "__main__":
    input_image_file = 'imgs/stitching_results/panorama_superpoint_homography.jpg' # <-- 您的全景图文件名
    output_video_file = 'panorama_scan_clean.mp4'
    output_image_file = 'panorama_cropped.png'
    # 1. 使用新的、更精确的裁剪函数
    cropped_img = crop_to_max_inner_rect(input_image_file)

    if cropped_img is not None:
        cv2.imwrite(output_image_file, cropped_img)
        print(f"裁剪后的图像已保存至: {output_image_file}")
        # 2. 创建扫描视频 (参数可自定义)
        create_panning_video(
            image=cropped_img,
            output_path=output_video_file,
            slice_width=1920,
            fps=30,
            step=5
        )