training:

python train.py --name train_default -b 8 --gpus 0 1 2 3 4 5 6 7

inference:

ffmpeg -i xxx.mp4 -strict -2 -vf scale=-1:360 test.mp4

cd src/video2npz
sh video2npz.sh ../../videos/test.mp4

python gen_midi_conditional.py -f "../inference/test.npz" -c "../exp/loss_8_params.pt" -n 5
