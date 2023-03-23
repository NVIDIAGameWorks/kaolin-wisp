pytest -s \
  --log-cli-level=INFO \
  --log-cli-format="%(asctime)s|%(levelname)8s| %(message)s" \
  tests/apps/test_nerf.py \
  --dataroot data/ \
  --dataset-num-workers 16
