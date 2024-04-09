import dlib
import torch
import numpy as np
import os
import cv2
from PIL import Image


def tensor2PILImage(tensor):
    tmp = (
        tensor.detach()
        .clamp(min=-1, max=1)
        .add(1)
        .div(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )
    return Image.fromarray(tmp[0])


def tensor2PILImages(tensor):
    batch = tensor.shape[0]
    tmp = (
        tensor.detach()
        .clamp(min=-1, max=1)
        .add(1)
        .div(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )
    imgs = []
    for i in range(batch):
        imgs.append(Image.fromarray(tmp[i]))
    return imgs


def detect_crop_face(PIL_img, detector):
    # 返回PIL.Image格式的裁剪人脸图片
    img = cv2.cvtColor(
        np.asarray(PIL_img), cv2.COLOR_RGB2BGR
    )  # 将PIL.Image转OpenCV格式
    face = detector(img, 1)

    # print(f'人脸数：{len(face)}')

    try:
        d0 = face[0]
    except IndexError:
        return (0, 0), PIL_img
    else:
        height = d0.bottom() - d0.top()
        width = d0.right() - d0.left()
        start = (d0.left(), d0.top())

        img_blank = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                try:
                    img_blank[i, j, :] = img[d0.top() + i, d0.left() + j, :]
                except IndexError:
                    return (0, 0), PIL_img
        cropped_PIL_img = Image.fromarray(cv2.cvtColor(img_blank, cv2.COLOR_BGR2RGB))
        return start, cropped_PIL_img


def crop_face_pos(PIL_img, face_shape, img_size, detector):
    # 返回PIL.Image格式的裁剪人脸图片
    h, w = face_shape
    img = cv2.cvtColor(
        np.asarray(PIL_img), cv2.COLOR_RGB2BGR
    )  # 将PIL.Image转OpenCV格式
    face = detector(img, 1)

    if len(face) == 0:
        st_left = (img_size - w) // 2
        st_top = (img_size - h) // 2
        return st_left, st_top

    d0 = face[0]
    st_left = d0.left()
    st_top = d0.top()
    if st_left + w > img_size:
        st_left = img_size - w
    if st_top + h > img_size:
        st_top = img_size - h
    return st_left, st_top


def facenet_input_preprocessing(PIL_img, input_H_W):
    # 尺寸处理 + 归一化
    img = PIL_img.convert('RGB')
    org_w, org_h = img.size  # size为（宽，高），注意shape为H*W*C
    inp_h, inp_w = input_H_W
    scale = min(inp_h / org_h, inp_w / org_w)
    nw = int(org_w * scale)
    nh = int(org_h * scale)

    img = img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (inp_w, inp_h), (128, 128, 128))
    new_image.paste(img, ((inp_w - nw) // 2, (inp_h - nh) // 2))

    input_img = torch.from_numpy(
        np.expand_dims(
            np.transpose(np.array(new_image, np.float32) / 255, (2, 0, 1)), 0
        )
    ).type(torch.FloatTensor)
    return input_img


if __name__ == '__main__':
    img_path = '../1.png'
    save_path = '../cropped_img/'
    save_name = os.path.basename(img_path).split('.')[0]
    print(save_name)

    img = Image.open(img_path)
    start, cropped_img = detect_crop_face(img)
    print(type(start[0]))
    print(start[0])
    a = np.zeros((1000))
    print(a[start[0]])

    save_path = save_path + save_name + '.jpg'
    # cropped_img.save(save_path)
