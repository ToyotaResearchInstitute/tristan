# Waymo Merge Filtering

This runs the `merge` filter over converted Waymo interactive protobufs and writes a JSON file with per-instance filter results. 

Set the input directory:

```bash
export WAYMO_INTERACTIVE_PB=/mnt/ssd/datasets/prediction/converted_protobufs/waymo/scenario/validation_interactive
```

Run from the repository root:

```bash
python intent/multiagents/create_resampling_data.py --input-dir "$WAYMO_INTERACTIVE_PB" --image-dir '' --cache-dir ~/intent/cache/waymo-filter --logs-dir ./filter-logs --output-file filter-output.merge.json --log-name-string tristan_test --filter-strategy merge --visualize true --visualize-failed-instances false --scene-image-mode none --agent-image-mode none --datasets-name-lists none --ignore-ego --use-waymo-dataset false --training-set-ratio 0.1 --past-timesteps 11 --future-timesteps 80 --past-timestep-size 0.1 --future-timestep-size 0.1 --interp-type none --max-agents 64 --num-workers 30 --batch-size 8 --map-points-max 20000 --map-points-subsample-ratio 2 --augment-trajectories false --report-agent-type-metrics true --report-waymo-metrics false
```

The output is written under:

```text
filter_result/tristan_test/filter-output.merge.json
filter_result/tristan_test/vis_merge/
```

For a quick smoke test, append:

```bash
--max-files-count 10 --num-workers 0 --output-file smoke-filter-output.merge.json
```
