python3.6 generate_tracks.py --video test-0.mp4 --output-directory ../output
python3.6 create_dataset.py --videos test-0.mp4 --tracks ../output/test-0-tracks.npz --out-file ../dataset/test
python3.6 run_tda.py --dataset ../dataset/test --visualise
python3.6 run_tda.py --dataset ../dataset/test --tda
