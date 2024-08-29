#rm -rf ./data/annotation
#mkdir "/media/zasim/69366298-8e7c-477a-a176-aba06b9848d94/reyad/cnn-lstm-master/data/annotation"
#rm -rf ./data/image_data
#mkdir "/media/zasim/69366298-8e7c-477a-a176-aba06b9848d94/reyad/cnn-lstm-master/data/image_data"

python video_jpg_ucf101_hmdb51.py
python n_frames_ucf101_hmdb51.py
python gen_anns_list.py
python ucf101_json.py