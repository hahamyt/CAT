import glob
import os
import imageio
from PIL import Image, ImageSequence

# filepaths
fp_in = "Bird1/*.jpg"
fp_out = "./Bird1.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=30, loop=0)

# Reduce frames
im = Image.open(fp_out)
index = 1
for frame in ImageSequence.all_frames(im):
    frame = frame.convert('RGB')
    frame.save(f"gif{index}.jpg", quality=100)
    index = index + 1

frame_limit = 60

n = int(index / frame_limit) + 1
images = []
for i in range(1, index):
    if i % n == 0:
        images.append(imageio.imread(f'gif{i}.jpg'))
imageio.mimsave(fp_out, images, duration=0.1)


for i in range(1, index):
    f = 'gif' + str(i) + '.jpg'
    if os.path.exists(f):
        os.remove(f)