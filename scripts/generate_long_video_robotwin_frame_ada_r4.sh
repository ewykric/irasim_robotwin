export PYTHONPATH="/mnt/wyk/IRASim:$PYTHONPATH"
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 0 --thread 0 --thread-num 8 > bridge_frame_ada_0.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 1 --thread 1 --thread-num 8 > bridge_frame_ada_1.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 2 --thread 2 --thread-num 8 > bridge_frame_ada_2.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 3 --thread 3 --thread-num 8 > bridge_frame_ada_3.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 0 --thread 4 --thread-num 8 > bridge_frame_ada_4.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 1 --thread 5 --thread-num 8 > bridge_frame_ada_5.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 2 --thread 6 --thread-num 8 > bridge_frame_ada_6.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 3 --thread 7 --thread-num 8 > bridge_frame_ada_7.txt 2>&1 &
sleep 10