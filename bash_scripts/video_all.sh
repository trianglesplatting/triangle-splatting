# tandt
python create_video.py -m models/$1/truck
python create_video.py -m models/$1/train

# db
python create_video.py -m models/$1/drjohnson
python create_video.py -m models/$1/playroom

# m360 indoor
python create_video.py -m models/$1/room
python create_video.py -m models/$1/counter
python create_video.py -m models/$1/kitchen
python create_video.py -m models/$1/bonsai

# m360 outdoor
python create_video.py -m models/$1/bicycle
python create_video.py -m models/$1/flowers
python create_video.py -m models/$1/garden
python create_video.py -m models/$1/stump
python create_video.py -m models/$1/treehill