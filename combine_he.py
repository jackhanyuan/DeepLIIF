import os
from PIL import Image


def blend_images(base_image, overlay_image):
    base_data = base_image.load()
    overlay_data = overlay_image.load()
    width, height = base_image.size

    for y in range(height):
        for x in range(width):
            current_pixel = overlay_data[x, y]

            # 如果当前像素不是白色，则添加到基底图像
            if current_pixel != (255, 255, 255):  # If the pixel is not white
                base_data[x, y] = current_pixel

    return base_image


def blend_imallges_from_folder(folder_path, output_path, blend_he=False):
    # 获取文件夹中所有图像文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))]
    he_images = [f for f in image_files if 'he' in f.lower()]
    assert len(he_images) <= 1, "Multiple HE images found in the folder"
    ihc_images = [f for f in image_files if 'posseg' in f.lower()]
    assert len(ihc_images) != 0, "No IHC images found in the folder"
    print(f"IHC images: {ihc_images}")

    # 创建一个白色背景的新图像
    first_image = Image.open(os.path.join(folder_path, image_files[0])).convert("RGB")
    width, height = first_image.size
    combined_ihc = Image.new("RGB", (width, height), (255, 255, 255))

    # 迭代所有ihc图像并将它们合并
    for image_file in ihc_images:
        image_path = os.path.join(folder_path, image_file)
        current_image = Image.open(image_path).convert("RGB")
        combined_ihc = blend_images(combined_ihc, current_image)

    # 保存ihc融合后的图像
    ihc_output_path = os.path.join(output_path, 'combined_ihc.png')
    combined_ihc.save(ihc_output_path)
    combined_ihc.show()

    # 如果提供了blend_he，并且存在he图像，则将ihc图像融合到he图像中
    if blend_he and he_images:
        print(f"HE images: {he_images}")
        he_image_path = os.path.join(folder_path, he_images[0])
        he_image = Image.open(he_image_path).convert("RGB")
        combined_he = blend_images(he_image, combined_ihc)

        # 保存he和ihc融合后的图像
        he_output_path = os.path.join(output_path, 'combined_he.png')
        combined_he.save(he_output_path)
        combined_he.show()


# 读取color文件夹中的所有图片
# folder_path = './Datasets/Test/color/combine'
# output_path = './Datasets/Test/color/combine'
folder_path = './Datasets/Test'
output_path = './Datasets/Test'

# 运行函数，将参数blend_he设置为True
blend_images_from_folder(folder_path, output_path, blend_he=True)
