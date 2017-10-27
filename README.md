# Data Set Downloader
[cascade filter](https://github.com/nagadomi/lbpcascade_animeface/blob/6bb3209d70aa9bb16689343547b29dc832e95ef7/lbpcascade_animeface.xml)
[face detection](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/6dc81725889d4ba782017873184355f5f5ad3188/anime_dataset_gen.py)

## Installation
Run this
```
pip install -r requirements.txt
```

## Usage
** IMPORTANT ** Images download are ** NOT SAFE FOR WORK (NSFW) **

Execute this
```
python src/python/cli.py -h
```

To get a list of flags and defaults.

TL;DR
```
python src/python/cli.py -t <tags.txt> -dc -oc
```
Gets the first 100 images of each `tag` in the `tags.txt`, a newline separated file of tags to search on danbooru.
These images are saved in the directory `./gallery-dl/danbooru/<tag_name>`. Then, run the face detection on each image.
This is denoted by the `-d` command.

Only images with one face detected are cropped and saved under the `./faces/<id>_<tag_name>.png`. This is denoted by the
`-c` command. The `-oc` command is used to only crop images that have color. All cropped images are by default 64 x 64.





