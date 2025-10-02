export PYTHONPATH="/mnt/wyk/IRASim:$PYTHONPATH"
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 0 --thread 0 --thread-num 16 > bridge_frame_ada_0.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 1 --thread 1 --thread-num 16 > bridge_frame_ada_1.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 2 --thread 2 --thread-num 16 > bridge_frame_ada_2.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 3 --thread 3 --thread-num 16 > bridge_frame_ada_3.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 4 --thread 4 --thread-num 16 > bridge_frame_ada_4.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 5 --thread 5 --thread-num 16 > bridge_frame_ada_5.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 6 --thread 6 --thread-num 16 > bridge_frame_ada_6.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 7 --thread 7 --thread-num 16 > bridge_frame_ada_7.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 0 --thread 8 --thread-num 16 > bridge_frame_ada_8.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 1 --thread 9 --thread-num 16 > bridge_frame_ada_9.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 2 --thread 10 --thread-num 16 > bridge_frame_ada_10.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 3 --thread 11 --thread-num 16 > bridge_frame_ada_11.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 4 --thread 12 --thread-num 16 > bridge_frame_ada_12.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 5 --thread 13 --thread-num 16 > bridge_frame_ada_13.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 6 --thread 14 --thread-num 16 > bridge_frame_ada_14.txt 2>&1 &
sleep 10
nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/robotwin/frame_ada.yaml' --data_config data_robotwin.yaml --rank 7 --thread 15 --thread-num 16 > bridge_frame_ada_15.txt 2>&1 &
sleep 10
