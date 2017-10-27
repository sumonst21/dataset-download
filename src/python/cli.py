import argparse
import os
import subprocess
from multiprocessing import Pool

import cv2
from PIL import Image
from tqdm import tqdm


class ImageGetter(object):
    class Downloader(object):
        def __init__(self, image_range, url):
            image_indices = "{0}-{1}".format(image_range[0], image_range[1])
            self.image_arg = "--images={0}".format(image_indices)
            self.url = url

        def __call__(self, tag):
            endpoint = "{0}{1}".format(self.url, tag)
            command = ["gallery-dl", self.image_arg, endpoint]
            subprocess.run(command, check=True)

    def __init__(self, url):
        self.url = url

    def execute(self, image_range, tags):
        with Pool(processes=4) as pool:
            pool.map(self.Downloader(image_range, self.url), tags)


class FaceCropper(object):
    def __init__(self, crop_size=(64, 64), only_color=True):
        if not os.path.exists("./faces"):
            os.makedirs("./faces")
        self.crop_size = crop_size
        self.only_color = only_color

    def execute(self):
        global_id = 0
        face_cascade = cv2.CascadeClassifier("./src/cascade/lbpcascade_animeface.xml")

        for root, sub_directories, files in tqdm(os.walk("./gallery-dl")):
            if files:
                label = os.path.basename(root)
                for index, file in enumerate(files):
                    image = cv2.imread(os.path.join(root, file))

                    if image is not None:

                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(90, 90))

                        if len(faces) == 1:
                            if self.only_color and (Image.fromarray(image).convert("RGB").getcolors() is not None):
                                continue

                            x, y, w, h = faces[0]
                            cropped_image = image[y: y + h, x: x + w, :]
                            resized_image = cv2.resize(cropped_image, self.crop_size)

                            output = "./faces/{0}_{1}.png".format(global_id, label)
                            cv2.imwrite(output, resized_image)
                            global_id += 1


def read_tags(tag_path):
    with open(tag_path) as tag_file:
        tags = tag_file.readlines()
    return [tag.strip() for tag in tags]


def main():
    parser = argparse.ArgumentParser(description="Anime Image Dataset Generator")

    parser.add_argument("--tags", "-t", help="Tags file", required=True, type=str)
    parser.add_argument("--download", "-d", help="Download Images (default: False)", default=False, action="store_true")
    parser.add_argument("--crop", "-c", help="Crop Faces out of Downloaded Images (default: False)", default=False, action="store_true")
    parser.add_argument("--range", "-r", help="Image range (default: 1-100)", default="1-100")
    parser.add_argument("--width", "-W", help="Images width (default: 64)", default=64, type=int)
    parser.add_argument("--height", "-H", help="Images height (default: 64)", default=64, type=int)
    parser.add_argument("--only-color", "-oc", help="Only colored images (default: False)", default=False, action="store_true")

    args = parser.parse_args()

    tags = read_tags(args.tags)
    image_indices = args.range.split("-")
    image_range = (int(image_indices[0]), int(image_indices[1]))
    if args.download:
        getter = ImageGetter("https://danbooru.donmai.us/posts?tags=")
        getter.execute(image_range=image_range, tags=tags)
    if args.crop:
        cropper = FaceCropper(crop_size=(args.width, args.height), only_color=args.only_color)
        cropper.execute()


if __name__ == "__main__":
    main()
