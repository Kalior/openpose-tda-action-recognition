python3.6 generate_tracks.py --video test-0.mp4 --out-directory output
python3.6 create_dataset.py --videos test-0.mp4 --tracks output/test-0-tracks.npz --out-file dataset/test
python3.6 visualise_dataset.py --dataset dataset/test --point-clouds
python3.6 train_classifier.py --dataset dataset/test --tda
python3.6 live_prediction.py --classifier classifier.pkl --video test-0.mp4 --out-directory predictions
