import cv2


def put_text(
    img,
    text,
    font=cv2.FONT_HERSHEY_PLAIN,
    pos=(0, 0),
    font_scale=3,
    font_thickness=2,
    text_color=(255, 255, 255),
    text_color_bg=(0, 0, 0),
):
    # Source: https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size


def resize_and_pad(image, new_shape, padding_color=(0, 0, 0)):
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])

    if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
        ratio = float(min(new_shape)) / min(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])

    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image
