#!/usr/bin/env bash
# Check renders for branch fix/visibility (worktree ~/code/physloc-fixes):
# real-time physics, multi fixes, pendulum post/axle. Every family of each job.
# 61 frames at 30 fps, 256 px / 32 spp for speed. One folder per job.
cd /home/ec2-user/code/physloc-fixes
OUT=/home/ec2-user/code/physloc/out/check_v2
run() {
  name=$1; job=$2
  conda run --no-capture-output -n physloc python -m physloc.cli generate \
    --config v0_release --outdir $OUT/$name --workdir /mnt/physloc/check_v2_work/$name \
    --only $job --resolution 256 --spp 32 --workers 1 > $OUT/$name.log 2>&1
  echo "$(date +%T) done $name rc=$?" >> $OUT/status.txt
}
run 01_barrier_pass_multi         L0_barrier_pass_20260832_v8 &
run 02_barrier_pass_camera_multi  L0_barrier_pass_20260833_v9 &
run 03_collision_camera_multi     L0_collision_20260833_v9 &
run 04_drop_multi                 L0_drop_20260832_v8 &
run 05_drop_camera_multi          L0_drop_20260833_v9 &
run 06_pyramid_impact_multi       L0_pyramid_impact_20260832_v8 &
run 07_resting_table_multi        L0_resting_table_20260832_v8 &
run 08_stack_topple_multi         L0_stack_topple_20260832_v8 &
run 09_stack_topple_camera_multi  L0_stack_topple_20260833_v9 &
run 10_toss_multi                 L0_toss_20260832_v8 &
run 11_drop_standard              L0_drop_20260824_v0 &
run 12_toss_camera_multi_L1       L1_toss_21260828_v4 &
run 13_pendulum_standard          L0_pendulum_swing_20260824_v0 &
run 14_pendulum_multi             L0_pendulum_swing_20260832_v8 &
run 15_pendulum_camera_multi      L0_pendulum_swing_20260833_v9 &
wait
echo "$(date +%T) ALL DONE" >> $OUT/status.txt
