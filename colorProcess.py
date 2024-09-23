#transfer rpg to hsv
import cv2
import numpy as np
import matplotlib.pyplot as plt

def transfer_rpg_to_hsv(rpg_image):
    return cv2.cvtColor(rpg_image, cv2.COLOR_RGB2HSV)

def create_mask(hsv_image):
    # 调整HSV阈值
    lower_bound = np.array([90, 50, 50])
    upper_bound = np.array([100, 255, 170])
    
    # 创建初始掩码
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # # 应用形态学操作
    # kernel = np.ones((5,5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def define_roi(image):
    # 定义ROI，排除镜子区域
    symmetry_x = 92  # X坐标 (水平)
    symmetry_y = 175
    # select left down corner
    roi_mask = np.zeros(image.shape[:2], np.uint8)
    roi_mask[symmetry_y:, symmetry_x:] = 255
    # roi_mask = image[symmetry_y:, symmetry_x:]
    return roi_mask

def refine_mask(hsv_image, mask):
    # 结合饱和度通道进一步改进掩码
    s_channel = hsv_image[:,:,1]
    s_threshold = cv2.threshold(s_channel, 100, 255, cv2.THRESH_BINARY)[1]
    refined_mask = cv2.bitwise_and(mask, s_threshold)
    return refined_mask

def find_fish_features(mask):
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, None
    
    # 假设最大的轮廓是鱼
    fish_contour = max(contours, key=cv2.contourArea)
    
    # 计算最小外接矩形
    rect = cv2.minAreaRect(fish_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # 获取矩形的中心、宽度、高度和角度
    center, (width, height), angle = rect
    
    # 确保宽度始终小于高度
    if width > height:
        width, height = height, width
        angle += 90
    
    # 创建旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1)
    
    # 旋转mask
    rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # 计算旋转后的矩形坐标
    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)
    
    # 提取鱼的区域
    fish_region = rotated_mask[y:y+int(height), x:x+int(width)]
    
    # 计算上下两端的像素和
    top_sum = np.sum(fish_region[:int(height/4)])
    bottom_sum = np.sum(fish_region[-int(height/4):])
    
    # 确定鱼头位置
    if bottom_sum > top_sum:
        head_end = np.array([x + width/2, y + height])
    else:
        head_end = np.array([x + width/2, y])
    
    # 将鱼头坐标旋转回原始方向
    center_array = np.array(center)
    head_end = np.dot(M[:2, :2], head_end - center_array) + center_array
    head_end = tuple(map(int, head_end))
    
    return box, head_end, fish_contour, angle

def find_fish_mouth(contour, head, angle):
    # 将轮廓点转换为numpy数组
    contour_points = contour.squeeze()
    
    # 计算每个点到鱼头的距离
    distances = np.linalg.norm(contour_points - np.array(head), axis=1)
    
    # 选择距离鱼头较近的点（例如，最近的20%的点）
    near_head_indices = np.argsort(distances)[:int(len(distances) * 0.2)]
    near_head_points = contour_points[near_head_indices]
    
    # 根据鱼的方向，选择最前面的点作为鱼嘴
    if -45 <= angle <= 45 or angle >= 135 or angle <= -135:  # 鱼是竖直的
        mouth = near_head_points[np.argmin(near_head_points[:, 1])]
    else:  # 鱼是水平的
        mouth = near_head_points[np.argmin(near_head_points[:, 0])]
    
    return tuple(mouth)

def draw_fish_features(image, box, head, mouth):
    # 绘制最小外接矩形
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    
    # 绘制鱼头
    cv2.circle(image, head, 5, (0, 0, 255), -1)
    
    # 绘制鱼嘴
    cv2.circle(image, mouth, 3, (255, 0, 0), -1)
    
    return image

if __name__ == "__main__":
    # Load the RGB image
    rgb_image = cv2.imread('first_frame.jpg')
    
    # Transfer the RGB image to HSV color space
    hsv_image = transfer_rpg_to_hsv(rgb_image)
    
    # Create initial mask
    mask = create_mask(hsv_image)
    
    # Define ROI
    roi_mask = define_roi(rgb_image)
    
    # Apply ROI to mask
    mask = cv2.bitwise_and(mask, roi_mask)
    
    # Find fish features
    box, head, contour, angle = find_fish_features(mask)
    
    if box is not None:
        # Find fish mouth
        mouth = find_fish_mouth(contour, head, angle)
        
        # Draw fish features
        result = draw_fish_features(rgb_image.copy(), box, head, mouth)
        
        # Create a figure with subplots
        plt.figure(figsize=(20, 5))
        
        # Display the original RGB image
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        plt.title('Original RGB Image')
        plt.axis('off')
        
        # Display the mask
        plt.subplot(142)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        
        # Display the result with fish features
        plt.subplot(143)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Fish Features')
        plt.axis('off')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        print("No fish detected in the image.")