# RQ3 Small V3 Local Diagnostic Slice Spec (2026-03-18)

- benchmark_id: rq3_small_v3_local_diagnostic_slice_20260318
- formal_small_ready: false
- paid_api_allowed: false

## HDFS
- status: diagnostic_only
- slice_type: single_action_slice
- blocked_reason: Only a pipeline-focused diagnostic slice currently survives; HDFS transfer/storage families still require a fresh pool.
- approved_case_count: 4
- review_case_count: 0

### Approved
- hdfs_42 | HDFS_REBUILD_WRITE_PIPELINE | score=20.6 | bucket=received_block
- hdfs_45 | HDFS_REBUILD_WRITE_PIPELINE | score=20.6 | bucket=received_block
- hdfs_46 | HDFS_REBUILD_WRITE_PIPELINE | score=20.6 | bucket=received_block
- hdfs_37 | HDFS_REBUILD_WRITE_PIPELINE | score=8.8 | bucket=receiving_block

## OpenStack
- status: diagnostic_only
- slice_type: mixed_action_slice
- blocked_reason: OpenStack has a mixed diagnostic slice, but host-claim and direct cache-bookkeeping alerts stay review-only until local probes prove they are not shortcut-driven.
- approved_case_count: 4
- review_case_count: 1

### Approved
- openstack_57 | OPENSTACK_REPAIR_BASE_IMAGE_CHAIN | score=29.2 | bucket=base_image_in_use
- openstack_51 | OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD | score=16.8 | bucket=instance_spawned
- openstack_56 | OPENSTACK_VERIFY_TERMINATION_AND_RESTORE | score=15.8 | bucket=termination
- openstack_49 | OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD | score=13.8 | bucket=instance_spawned

### Review
- openstack_win_20 | OPENSTACK_REBUILD_ON_COMPATIBLE_HOST | score=21.5 | bucket=host_claim

## Hadoop
- status: blocked
- slice_type: blocked
- blocked_reason: Dataset remains shortcut-dominated under current pool and should be reset before any new small selection.
- approved_case_count: 0
- review_case_count: 0

