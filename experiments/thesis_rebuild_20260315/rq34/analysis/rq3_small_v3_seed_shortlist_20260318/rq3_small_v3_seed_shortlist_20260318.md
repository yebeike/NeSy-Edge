# RQ3 Small V3 Seed Shortlist (2026-03-18)

## HDFS
- status: diagnostic_only
- blocked_reason: Only a pipeline-focused diagnostic slice currently survives; HDFS transfer/storage families still require a fresh pool.
- candidate_count: 122
- case_count_after_dedup: 25
- hard_flag_rows: 50
- approved_case_count: 4
- review_case_count: 0
- action_diversity: 1

### Approved
- hdfs_42 | HDFS_REBUILD_WRITE_PIPELINE | score=20.6 | diff=7.0 | bucket=received_block | flags=pipeline_keyword
  alert: 081111 085429 25314 INFO dfs.DataNode$PacketResponder: Received block blk_-6719337521981113643 of size 3538321 from /10.250.15.198
- hdfs_45 | HDFS_REBUILD_WRITE_PIPELINE | score=20.6 | diff=7.0 | bucket=received_block | flags=pipeline_keyword
  alert: 081111 083143 24261 INFO dfs.DataNode$PacketResponder: Received block blk_-5970165518655489569 of size 67108864 from /10.250.14.196
- hdfs_46 | HDFS_REBUILD_WRITE_PIPELINE | score=20.6 | diff=7.0 | bucket=received_block | flags=pipeline_keyword
  alert: 081111 071713 23240 INFO dfs.DataNode$PacketResponder: Received block blk_-1236077201457689973 of size 67108864 from /10.251.35.1
- hdfs_37 | HDFS_REBUILD_WRITE_PIPELINE | score=8.8 | diff=2.0 | bucket=receiving_block | flags=pipeline_keyword
  alert: 081111 081847 24224 INFO dfs.DataNode$DataXceiver: Receiving block blk_1512136249403454074 src: /10.250.7.244:47958 dest: /10.250.7.244:50010

## OpenStack
- status: diagnostic_only
- blocked_reason: OpenStack has a mixed diagnostic slice, but host-claim and direct cache-bookkeeping alerts stay review-only until local probes prove they are not shortcut-driven.
- candidate_count: 67
- case_count_after_dedup: 11
- hard_flag_rows: 4
- approved_case_count: 4
- review_case_count: 1
- action_diversity: 3

### Approved
- openstack_57 | OPENSTACK_REPAIR_BASE_IMAGE_CHAIN | score=29.2 | diff=9.9 | bucket=base_image_in_use | flags=none
  alert: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:14:30.678 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): in use: on this node 1 local, 0 on other nodes sharing this instance storage
- openstack_51 | OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD | score=16.8 | diff=5.5 | bucket=instance_spawned | flags=none
  alert: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:16.313 2931 INFO nova.virt.libvirt.driver [-] [instance: 127e769a-4fe6-4548-93b1-513ac51e0452] Instance spawned successfully.
- openstack_56 | OPENSTACK_VERIFY_TERMINATION_AND_RESTORE | score=15.8 | diff=4.0 | bucket=termination | flags=none
  alert: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:12:01.590 2931 INFO nova.compute.manager [req-121ecfae-3fb1-49cc-9a78-8b046fe73a77 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] [instance: d96a117b-0193-4549-bdcc-63b917273d1d] Terminating instance
- openstack_49 | OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD | score=13.8 | diff=3.0 | bucket=instance_spawned | flags=none
  alert: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:57.756 2931 INFO nova.virt.libvirt.driver [-] [instance: c62f4f25-982c-4ea2-b5e4-93000edfcfbf] Instance spawned successfully.

### Review
- openstack_win_20 | OPENSTACK_REBUILD_ON_COMPATIBLE_HOST | score=21.5 | diff=9.5 | bucket=host_claim | flags=direct_host_claim
  alert: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:50:06.115 2931 INFO nova.compute.claims [req-cf8e4d9c-928d-406f-bd8f-b0bcb1677a90 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] [instance: f14b2b2f-4cc2-450a-8314-ab3b6f6f30a0] Attempting claim: memory 2048 MB, disk 20 GB, vcpus 1 CPU

## Hadoop
- status: blocked
- blocked_reason: Dataset remains shortcut-dominated under current pool and should be reset before any new small selection.
- candidate_count: 44
- case_count_after_dedup: 10
- hard_flag_rows: 39
- approved_case_count: 0
- review_case_count: 0
- action_diversity: 0

