import cv2
import numpy as np
from skimage import color
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage import color, filters
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import label, find_objects
from scipy.ndimage import center_of_mass

def load_and_convert_image(image_path):
    # 加载图像
    image = Image.open(image_path)
    image_np = np.array(image)

    # 检查是否有Alpha通道，并分离RGB和Alpha通道
    alpha_channel = None
    if image_np.shape[2] == 4:
        rgb_image = image_np[:, :, :3]
        alpha_channel = image_np[:, :, 3]
    else:
        rgb_image = image_np

    # RGB到LAB转换
    lab_image = color.rgb2lab(rgb_image)
    return lab_image, alpha_channel, rgb_image


def create_edge_image(lab_image, alpha_channel):
    # 找到非透明像素，即物体的像素
    object_mask = alpha_channel > 0
    object_mask = ~ object_mask
    # 膨胀物体的掩码，以获取边缘的外围像素
    kernel4 = np.ones((4, 4), np.uint8)
    dilated_object_mask4 = cv2.dilate(object_mask.astype(np.uint8), kernel4, iterations=1)
    kernel1 = np.ones((1, 1), np.uint8)
    dilated_object_mask1 = cv2.dilate(object_mask.astype(np.uint8), kernel1, iterations=1)
    # 计算边缘的4排像素：膨胀后的掩码减去原始掩码
    edge_three_pixel_mask = dilated_object_mask4 - dilated_object_mask1
    # 提取L分量作为灰度图像
    L_channel = lab_image[:, :, 0]
    # 用黑色填充整个图像
    edge_image = np.zeros(L_channel.shape, dtype=np.uint8)
    # 在边缘的4排像素上应用L通道的灰度值
    edge_image[edge_three_pixel_mask > 0] = L_channel[edge_three_pixel_mask > 0]

    return edge_image




from scipy.ndimage import label, find_objects
import numpy as np

def find_brightest_regions(edge_image, alpha_channel, percentile=3.5, min_pixels=10):
    # 保留alpha通道非零的像素
    valid_pixels = edge_image[alpha_channel > 0]

    # 计算亮度直方图
    hist, bin_edges = np.histogram(valid_pixels, bins=256, range=(1, 255))
    threshold_pixels = valid_pixels[valid_pixels>0]
    # 计算亮度前15.87%的阈值
    brightness_threshold = np.percentile(threshold_pixels, 100 - percentile)
    if brightness_threshold > 90:
        brightness_threshold = 90
    # 在图像中找到所有亮度大于或等于阈值的像素
    region_mask = (edge_image >= brightness_threshold) & (alpha_channel > 0)

    # 标记连通区域
    labeled_array, num_features = label(region_mask)

    # 存储找到的亮区域的信息
    regions_with_area = []

    for region_index in range(1, num_features + 1):
        region = (labeled_array == region_index)
        region_pixel_count = np.sum(region)

        # 确保区域至少有min_pixels个像素
        if region_pixel_count >= min_pixels:
            slice_x, slice_y = find_objects(region)[0]
            regions_with_area.append((region_pixel_count, (region, slice_x, slice_y)))

    # 按区域大小排序并只取前三个
    regions_with_area.sort(key=lambda x: x[0], reverse=True)
    brightest_regions = [x[1] for x in regions_with_area[:2]]

    return brightest_regions


def calculate_weighted_normals(brightest_regions, edge_image, alpha_channel, s=5):
    normals = []
    object_mask = alpha_channel > 0

    for region, _, _ in brightest_regions:
        region_mask = region & object_mask
        y_coords, x_coords = np.where(region_mask)

        if len(x_coords) < 4:
            continue

        sorted_indices = np.argsort(x_coords)
        x_sorted = x_coords[sorted_indices]
        y_sorted = y_coords[sorted_indices]

        spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted) * s)

        center_y, center_x = center_of_mass(edge_image, labels=region_mask, index=1)

        if not (x_sorted[0] <= center_x <= x_sorted[-1]):
            continue


        slope = spline.derivative()(center_x)


        if np.isnan(slope):
            continue

        normal = np.array([-slope, 1])
        normal /= np.linalg.norm(normal)

        brightness = edge_image[int(center_y), int(center_x)]
        if brightness == 0:
            brightness = edge_image[int(center_y)+1, int(center_x)]

        weighted_normal = normal * brightness

        normals.append((center_x, center_y, weighted_normal))

    return normals




def normal_combine(normals):
    if len(normals) == 0:
        return np.array([0.0, 0.0]), 0.0

    normals = np.array(normals)  # Convert list of numpy arrays to a 2D numpy array

    # 第一步：舍弃与水平方向夹角超过 5 度且指向上方的向量
    filtered_normals = []
    for normal in normals:
        angle_with_horizontal = np.arcsin(np.clip(normal[1] / np.linalg.norm(normal), -1.0, 1.0))
        if not (normal[1] < 0 and abs(np.degrees(angle_with_horizontal)) > 5):
            filtered_normals.append(normal)

    if not filtered_normals:  # 如果没有剩余的向量，则返回零向量
        return np.array([0.0, 0.0]), 0.0

    # 第二步：计算剩余向量的平均值
    mean_normal = np.mean(filtered_normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)

    valid_normals = []
    for normal in filtered_normals:
        # 检查向量是否与平均向量的夹角小于等于 30 度
        angle_with_mean = np.arccos(np.clip(np.dot(normal, mean_normal) / (np.linalg.norm(normal) * np.linalg.norm(mean_normal)), -1.0, 1.0))

        if np.degrees(angle_with_mean) <= 30:
            valid_normals.append(normal)

    if valid_normals:
        combined_normal = np.mean(valid_normals, axis=0)
        combined_magnitude = np.linalg.norm(combined_normal)
        combined_normal /= combined_magnitude  # Normalize the combined vector
    else:
        combined_normal = np.array([0.0, 0.0])
        combined_magnitude = 0.0

    return combined_normal, combined_magnitude



def main():
    lab_image, alpha_channel, rgb_image = load_and_convert_image('D:/STUDY/1PhD/blender model/charizar/output_s/0120.png')
    if alpha_channel is not None:
        edge_image = create_edge_image(lab_image, alpha_channel)

        # 在这里找到亮度最高的区域
        brightest_regions = find_brightest_regions(edge_image, alpha_channel)

        highlighted_image = np.zeros(edge_image.shape, dtype=np.uint8)
        # 在图像上标记亮度最高的区域
        for region, slice_x, slice_y in brightest_regions:
            highlighted_image[region] = edge_image[region]

        weighted_normals = calculate_weighted_normals(brightest_regions, edge_image, alpha_channel)
        length = 2
        #length = 20
        highlighted_image_color = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
        for center_x, center_y, weighted_normal in weighted_normals:
            magnitude = np.linalg.norm(weighted_normal)
            if magnitude > 0:  # 确保法向量不是零向量
                end_x = int(center_x + (weighted_normal[0] / magnitude) * length)
                end_y = int(center_y + (weighted_normal[1] / magnitude) * length)
                cv2.arrowedLine(highlighted_image_color, (int(center_x), int(center_y)), (end_x, end_y), (0, 0, 255), 2)

            # 计算合并的法向量
        combined_normal, combined_magnitude = normal_combine([n[2] for n in weighted_normals])

        if combined_magnitude > 0:  # Check if the magnitude is greater than zero
            center_x, center_y = edge_image.shape[1] // 2, edge_image.shape[0] // 2
            end_x = int(center_x + combined_normal[0] * combined_magnitude)
            end_y = int(center_y + combined_normal[1] * combined_magnitude)
            cv2.arrowedLine(highlighted_image_color, (center_x, center_y), (end_x, end_y), (255, 0, 0),2)  # Draw in blue

        # 将RGB图像转换为BGR以便在OpenCV中正确显示
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # 显示原始图像和标记了亮度最高区域的图像
        cv2.imshow('Original Image', bgr_image)
        #cv2.imshow('Edge Image', edge_image)
        #cv2.imshow('Highlighted Image', highlighted_image)
        cv2.imshow('Edge Image with Normals', highlighted_image_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#D:/STUDY/1PhD/blender model/box/output_s/3D_WoodenBox0100.png
#D:/STUDY/1PhD/blender model/charizar/output_s/0020.png
#D:/STUDY/1PhD/blender model/telephone/output_s/0108.png
#D:/STUDY/1PhD/blender model/clock2/output_s/0120.png
#D:/STUDY/1PhD/blender model/clock/output_s/0110.png
if __name__ == "__main__":
    main()
