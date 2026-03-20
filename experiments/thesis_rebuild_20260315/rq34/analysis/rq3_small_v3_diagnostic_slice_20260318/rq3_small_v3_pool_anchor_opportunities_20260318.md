# RQ3 Small V3 Pool Anchor Opportunities (2026-03-18)

## HDFS
- total_cases: 122
- total_improved_cases: 68
- total_weak_opportunities: 59

- HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION | cases=10 | improved=1 | weak=3
- HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK | cases=7 | improved=1 | weak=0
- HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE | cases=4 | improved=0 | weak=2
- HDFS_REBUILD_WRITE_PIPELINE | cases=36 | improved=11 | weak=1
- HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE | cases=50 | improved=48 | weak=48
- HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK | cases=11 | improved=3 | weak=4
- HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE | cases=4 | improved=4 | weak=1

### HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION
- hdfs_19 | blacklisted=false | diff=0.0 | current=10.5 | best=0.0
  current: 081110 211541 18 INFO dfs.DataNode: 10.250.15.198:50010 Starting thread to transfer block blk_4292382298896622412 to 10.250.15.240:50010
  alt: 081110 211523 14688 INFO dfs.DataNode$DataXceiver: Receiving block blk_-852118157909078164 src: /10.251.30.101:57254 dest: /10.251.30.101:50010
- hdfs_20 | blacklisted=false | diff=0.0 | current=0.0 | best=0.0
  current: 081111 081055 24136 INFO dfs.DataNode$DataXceiver: Received block blk_1473949624670719319 src: /10.251.29.239:35617 dest: /10.251.29.239:50010 of size 67108864
  alt: 081111 081029 24268 INFO dfs.DataNode$PacketResponder: Received block blk_1600228140214754078 of size 67108864 from /10.251.90.81
- hdfs_21 | blacklisted=false | diff=0.0 | current=0.0 | best=0.0
  current: 081111 045834 19263 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-4411589101766563890 terminating
  alt: 081111 045804 20338 INFO dfs.DataNode$PacketResponder: Received block blk_-337972286882621518 of size 67108864 from /10.250.10.144
- hdfs_7 | blacklisted=false | diff=0.0 | current=8.0 | best=8.0
  current: 081111 000626 17362 INFO dfs.DataNode$DataXceiver: 10.251.91.84:50010 Served block blk_-8915491531889006304 to /10.251.91.84
  alt: 081111 000037 17163 INFO dfs.DataNode$DataXceiver: 10.251.195.70:50010 Served block blk_-3696162841836791939 to /10.251.195.70
- hdfs_8 | blacklisted=false | diff=0.0 | current=8.0 | best=8.0
  current: 081110 031019 6604 INFO dfs.DataNode$DataXceiver: 10.251.126.83:50010 Served block blk_-3860894070657427592 to /10.251.126.83
  alt: 081110 035357 6606 INFO dfs.DataNode$DataXceiver: 10.251.71.97:50010 Served block blk_5454332143498402824 to /10.250.15.67
- hdfs_19 | blacklisted=false | diff=0.0 | current=10.5 | best=na
  current: 081110 211541 18 INFO dfs.DataNode: 10.250.15.198:50010 Starting thread to transfer block blk_4292382298896622412 to 10.250.15.240:50010

### HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK
- hdfs_blk_blk_-3544583377289625738 | blacklisted=false | diff=0.0 | current=19.0 | best=3.5
  current: 081109 213838 19 WARN dfs.FSDataset: Unexpected error trying to delete block blk_-3544583377289625738. BlockInfo not found in volumeMap.
  alt: 081109 213809 32 INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: blk_-3544583377289625738 is added to invalidSet of 10.251.39.179:50010
- hdfs_3 | blacklisted=false | diff=0.0 | current=9.0 | best=9.0
  current: 081110 220853 19 INFO dfs.FSDataset: Deleting block blk_2353918919834315845 file /mnt/hadoop/dfs/data/current/subdir35/blk_2353918919834315845
  alt: 081110 220811 19 INFO dfs.FSDataset: Deleting block blk_-5718504625725408070 file /mnt/hadoop/dfs/data/current/blk_-5718504625725408070
- hdfs_39 | blacklisted=false | diff=0.0 | current=9.0 | best=9.0
  current: 081111 075838 19 INFO dfs.FSDataset: Deleting block blk_-3594701313167635091 file /mnt/hadoop/dfs/data/current/subdir54/blk_-3594701313167635091
  alt: 081111 075833 19 INFO dfs.FSDataset: Deleting block blk_-4380385204018751771 file /mnt/hadoop/dfs/data/current/subdir38/blk_-4380385204018751771
- hdfs_4 | blacklisted=false | diff=0.0 | current=9.0 | best=9.0
  current: 081110 210640 19 INFO dfs.FSDataset: Deleting block blk_7108974241330205218 file /mnt/hadoop/dfs/data/current/subdir22/blk_7108974241330205218
  alt: 081110 210620 19 INFO dfs.FSDataset: Deleting block blk_8598135237831029828 file /mnt/hadoop/dfs/data/current/subdir59/blk_8598135237831029828
- hdfs_43 | blacklisted=false | diff=0.0 | current=9.0 | best=9.0
  current: 081111 090138 19 INFO dfs.FSDataset: Deleting block blk_92946806844541836 file /mnt/hadoop/dfs/data/current/subdir59/blk_92946806844541836
  alt: 081111 090136 19 INFO dfs.FSDataset: Deleting block blk_-2722768686407017386 file /mnt/hadoop/dfs/data/current/subdir36/blk_-2722768686407017386
- hdfs_3 | blacklisted=false | diff=0.0 | current=9.0 | best=na
  current: 081110 220853 19 INFO dfs.FSDataset: Deleting block blk_2353918919834315845 file /mnt/hadoop/dfs/data/current/subdir35/blk_2353918919834315845

### HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE
- hdfs_17 | blacklisted=false | diff=0.0 | current=0.0 | best=0.0
  current: 081111 101356 26434 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-9001895211102825241 terminating
  alt: 081111 101243 26312 INFO dfs.DataNode$PacketResponder: Received block blk_-1440254020029439248 of size 67108864 from /10.251.30.85
- hdfs_18 | blacklisted=true | diff=0.0 | current=0.0 | best=0.0
  current: 081109 232639 3792 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-6837050820114491742 terminating
  alt: 081109 232346 3820 INFO dfs.DataNode$DataXceiver: Receiving block blk_8692428775973608797 src: /10.250.10.213:56574 dest: /10.250.10.213:50010
- hdfs_17 | blacklisted=false | diff=0.0 | current=0.0 | best=na
  current: 081111 101356 26434 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-9001895211102825241 terminating
- hdfs_18 | blacklisted=true | diff=0.0 | current=0.0 | best=na
  current: 081109 232639 3792 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-6837050820114491742 terminating

### HDFS_REBUILD_WRITE_PIPELINE
- hdfs_2 | blacklisted=false | diff=1.0 | current=12.5 | best=0.0
  current: 081110 112152 9621 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_6610244864386894422 terminating
  alt: 081110 112155 13 INFO dfs.DataBlockScanner: Verification succeeded for blk_774612454978154966
- hdfs_32 | blacklisted=false | diff=1.0 | current=12.5 | best=6.0
  current: 081111 072356 23300 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_707166530951154301 terminating
  alt: 081111 072356 30 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.126.22:50010 is added to blk_707166530951154301 size 67108864
- hdfs_42 | blacklisted=false | diff=7.0 | current=12.5 | best=11.5
  current: 081111 085429 25314 INFO dfs.DataNode$PacketResponder: Received block blk_-6719337521981113643 of size 3538321 from /10.250.15.198
  alt: 081111 085351 24958 INFO dfs.DataNode$DataXceiver: Receiving block blk_3808164640986377400 src: /10.251.75.79:36685 dest: /10.251.75.79:50010
- hdfs_45 | blacklisted=false | diff=7.0 | current=12.5 | best=11.5
  current: 081111 083143 24261 INFO dfs.DataNode$PacketResponder: Received block blk_-5970165518655489569 of size 67108864 from /10.250.14.196
  alt: 081111 082809 24442 INFO dfs.DataNode$DataXceiver: Receiving block blk_469126362137371425 src: /10.251.199.245:53897 dest: /10.251.199.245:50010
- hdfs_46 | blacklisted=false | diff=7.0 | current=12.5 | best=11.5
  current: 081111 071713 23240 INFO dfs.DataNode$PacketResponder: Received block blk_-1236077201457689973 of size 67108864 from /10.251.35.1
  alt: 081111 071904 22560 INFO dfs.DataNode$DataXceiver: Receiving block blk_-7646004878694144580 src: /10.251.202.181:47130 dest: /10.251.202.181:50010
- hdfs_0 | blacklisted=false | diff=1.0 | current=12.5 | best=11.5
  current: 081110 210837 14418 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_3940282600163903101 terminating
  alt: 081110 210832 14473 INFO dfs.DataNode$DataXceiver: Receiving block blk_-5808893254440551816 src: /10.250.11.194:53256 dest: /10.250.11.194:50010

### HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE
- hdfs_blk_blk_-2033526004731990730 | blacklisted=false | diff=0.0 | current=5.0 | best=0.0
  current: 081109 214222 2703 WARN dfs.DataNode$DataXceiver: 10.250.17.177:50010:Got exception while serving blk_-2033526004731990730 to /10.250.7.230:
  alt: 081109 213558 13 INFO dfs.DataBlockScanner: Verification succeeded for blk_-2033526004731990730
- hdfs_blk_blk_-2126554733521224025 | blacklisted=false | diff=0.0 | current=5.0 | best=0.0
  current: 081109 214022 2598 WARN dfs.DataNode$DataXceiver: 10.251.39.209:50010:Got exception while serving blk_-2126554733521224025 to /10.251.123.132:
  alt: 081109 211304 13 INFO dfs.DataBlockScanner: Verification succeeded for blk_-2126554733521224025
- hdfs_blk_blk_-3102267849859399193 | blacklisted=false | diff=0.0 | current=11.5 | best=0.0
  current: 081109 204731 148 INFO dfs.DataNode$DataXceiver: writeBlock blk_-3102267849859399193 received exception java.io.IOException: Connection reset by peer
  alt: 081109 203634 190 INFO dfs.DataNode$PacketResponder: PacketResponder blk_-3102267849859399193 2 Exception java.io.EOFException
- hdfs_blk_blk_-4560669296456646716 | blacklisted=false | diff=0.0 | current=5.0 | best=0.0
  current: 081109 214929 2693 WARN dfs.DataNode$DataXceiver: 10.251.106.10:50010:Got exception while serving blk_-4560669296456646716 to /10.250.7.32:
  alt: 081109 213706 13 INFO dfs.DataBlockScanner: Verification succeeded for blk_-4560669296456646716
- hdfs_blk_blk_-5337994755640289830 | blacklisted=false | diff=0.0 | current=5.0 | best=0.0
  current: 081109 214309 2624 WARN dfs.DataNode$DataXceiver: 10.251.202.181:50010:Got exception while serving blk_-5337994755640289830 to /10.251.39.64:
  alt: 081109 214149 13 INFO dfs.DataBlockScanner: Verification succeeded for blk_-5337994755640289830
- hdfs_blk_blk_-6689614308409336057 | blacklisted=false | diff=0.0 | current=5.0 | best=0.0
  current: 081109 215002 2854 WARN dfs.DataNode$DataXceiver: 10.250.7.96:50010:Got exception while serving blk_-6689614308409336057 to /10.250.14.38:
  alt: 081109 214657 13 INFO dfs.DataBlockScanner: Verification succeeded for blk_-6689614308409336057

### HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK
- hdfs_23 | blacklisted=false | diff=0.0 | current=1.0 | best=1.0
  current: 081110 222511 16129 INFO dfs.DataNode$DataXceiver: Receiving block blk_2147212041377545584 src: /10.251.75.49:34485 dest: /10.251.75.49:50010
  alt: 081110 222455 16220 INFO dfs.DataNode$PacketResponder: Received block blk_4294115716822011564 of size 67108864 from /10.251.43.210
- hdfs_24 | blacklisted=false | diff=0.0 | current=1.0 | best=1.0
  current: 081109 205749 997 INFO dfs.DataNode$DataXceiver: Receiving block blk_-28342503914935090 src: /10.251.123.132:57542 dest: /10.251.123.132:50010
  alt: 081109 205742 1001 INFO dfs.DataNode$PacketResponder: Received block blk_-5861636720645142679 of size 67108864 from /10.251.70.211
- hdfs_5 | blacklisted=false | diff=0.0 | current=1.0 | best=1.0
  current: 081110 105910 9510 INFO dfs.DataNode$PacketResponder: Received block blk_-5421365489363678155 of size 67108864 from /10.251.122.38
  alt: 081110 105553 9531 INFO dfs.DataNode$DataXceiver: Receiving block blk_5272731391668336674 src: /10.250.6.191:46509 dest: /10.250.6.191:50010
- hdfs_6 | blacklisted=false | diff=0.0 | current=1.0 | best=1.0
  current: 081110 013715 5706 INFO dfs.DataNode$PacketResponder: Received block blk_-3677956642639780952 of size 67108864 from /10.251.123.20
  alt: 081110 013559 5614 INFO dfs.DataNode$PacketResponder: Received block blk_7736147489540456118 of size 67108864 from /10.251.74.227
- hdfs_22 | blacklisted=false | diff=0.0 | current=16.5 | best=6.0
  current: 081111 082527 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand7/_temporary/_task_200811101024_0014_m_001049_0/part-01049. blk_-1533191386601391937
  alt: 081111 082039 28 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.66.3:50010 is added to blk_-4970213618915792510 size 67108864
- hdfs_36 | blacklisted=false | diff=0.0 | current=16.5 | best=6.0
  current: 081111 082527 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand7/_temporary/_task_200811101024_0014_m_001049_0/part-01049. blk_-1533191386601391937
  alt: 081111 082039 28 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.66.3:50010 is added to blk_-4970213618915792510 size 67108864

### HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE
- hdfs_38 | blacklisted=false | diff=0.0 | current=16.5 | best=1.0
  current: 081111 092436 29 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.6.223:50010 is added to blk_-6390296685077811803 size 3546314
  alt: 081111 091733 19 INFO dfs.FSNamesystem: BLOCK* ask 10.251.126.5:50010 to delete  blk_-9016567407076718172 blk_-8695715290502978219 blk_-7168328752988473716 blk_-4355192005224403537 blk_-3757501769775889193 blk_-154600013573668394 blk_167132135416677587 blk_2654596473569751784 blk_5202581916713319258
- hdfs_41 | blacklisted=false | diff=0.0 | current=16.5 | best=6.0
  current: 081111 100417 30 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.30.101:50010 is added to blk_6451403582950672007 size 67108864
  alt: 081111 101225 34 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/randtxt9/_temporary/_task_200811101024_0016_m_000347_0/part-00347. blk_-8426741581316629266
- hdfs_44 | blacklisted=false | diff=0.0 | current=16.5 | best=6.0
  current: 081111 072921 29 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.107.242:50010 is added to blk_621685330749869453 size 67108864
  alt: 081111 072818 28 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand6/_temporary/_task_200811101024_0013_m_001195_0/part-01195. blk_5708405953850477535
- hdfs_33 | blacklisted=true | diff=0.0 | current=16.5 | best=6.0
  current: 081111 072059 30 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.30.6:50010 is added to blk_-2345355483730155757 size 67108864
  alt: 081111 071611 32 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand6/_temporary/_task_200811101024_0013_m_000997_0/part-00997. blk_-500534246236005335

## OpenStack
- total_cases: 67
- total_improved_cases: 66
- total_weak_opportunities: 65

- OPENSTACK_REBUILD_ON_COMPATIBLE_HOST | cases=1 | improved=1 | weak=1
- OPENSTACK_REPAIR_BASE_IMAGE_CHAIN | cases=54 | improved=53 | weak=54
- OPENSTACK_RESYNC_INSTANCE_INVENTORY | cases=2 | improved=2 | weak=2
- OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD | cases=5 | improved=5 | weak=5
- OPENSTACK_SCALE_METADATA_SERVICE | cases=4 | improved=4 | weak=2
- OPENSTACK_VERIFY_TERMINATION_AND_RESTORE | cases=1 | improved=1 | weak=1

### OPENSTACK_REBUILD_ON_COMPATIBLE_HOST
- openstack_win_20 | blacklisted=false | diff=9.5 | current=4.5 | best=0.0
  current: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:50:06.115 2931 INFO nova.compute.claims [req-cf8e4d9c-928d-406f-bd8f-b0bcb1677a90 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] [instance: f14b2b2f-4cc2-450a-8314-ab3b6f6f30a0] Attempting claim: memory 2048 MB, disk 20 GB, vcpus 1 CPU
  alt: nova-api.log.2017-05-14_21:27:04 2017-05-14 19:50:06.010 25746 INFO nova.osapi_compute.wsgi.server [req-7daf8f0c-549b-49c7-95c8-2551b4d0c7fa 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1583 time: 0.1921122

### OPENSTACK_REPAIR_BASE_IMAGE_CHAIN
- openstack_57 | blacklisted=false | diff=9.9 | current=1.0 | best=0.0
  current: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:14:30.678 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): in use: on this node 1 local, 0 on other nodes sharing this instance storage
  alt: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:14:25.518 2931 INFO nova.compute.resource_tracker [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] Compute_service record updated for cp-1.slowvm1.tcloud-pg0.utah.cloudlab.us:cp-1.slowvm1.tcloud-pg0.utah.cloudlab.us
- openstack_win_22 | blacklisted=false | diff=9.5 | current=1.0 | best=0.0
  current: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:51:00.204 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): checking
  alt: nova-api.log.2017-05-14_21:27:04 2017-05-14 19:50:59.266 25746 INFO nova.osapi_compute.wsgi.server [req-69beb1db-6c02-42a1-8a06-b4857d72cf8e 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2466671
- openstack_win_23 | blacklisted=false | diff=9.5 | current=1.0 | best=0.0
  current: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:51:00.204 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): checking
  alt: nova-api.log.2017-05-14_21:27:04 2017-05-14 19:50:59.266 25746 INFO nova.osapi_compute.wsgi.server [req-69beb1db-6c02-42a1-8a06-b4857d72cf8e 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2466671
- openstack_win_1 | blacklisted=false | diff=8.0 | current=1.0 | best=0.0
  current: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:15.195 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): checking
  alt: nova-api.log.2017-05-14_21:27:04 2017-05-14 19:39:15.203 25746 INFO nova.osapi_compute.wsgi.server [req-3127022f-f330-4686-83b7-f6c8e087fc53 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2673762
- openstack_win_10 | blacklisted=false | diff=8.0 | current=1.0 | best=0.0
  current: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:44:47.079 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): checking
  alt: nova-api.log.2017-05-14_21:27:04 2017-05-14 19:44:46.263 25746 INFO nova.osapi_compute.wsgi.server [req-52931106-f23a-4593-9446-68c956c1d861 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2891469
- openstack_win_11 | blacklisted=false | diff=8.0 | current=1.0 | best=0.0
  current: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:44:47.079 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): checking
  alt: nova-api.log.2017-05-14_21:27:04 2017-05-14 19:44:46.263 25746 INFO nova.osapi_compute.wsgi.server [req-52931106-f23a-4593-9446-68c956c1d861 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2891469

### OPENSTACK_RESYNC_INSTANCE_INVENTORY
- openstack_48 | blacklisted=true | diff=8.5 | current=9.0 | best=0.0
  current: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:09:08.029 25743 INFO nova.api.openstack.compute.server_external_events [req-2f1a8f6f-b2fc-48af-92c3-8a37d40f54dc f7b8d1f1d4d44643b07fa10ca7d021fb e9746973ac574c6b8a9e8857f56a7608 - - -] Creating event network-vif-plugged:1bb8c7a3-effa-4568-8cf7-2b9a55011df9 for instance 70c1714b-c11b-4c88-b300-239afe1f5ff8
  alt: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:09:08.034 25743 INFO nova.osapi_compute.wsgi.server [req-2f1a8f6f-b2fc-48af-92c3-8a37d40f54dc f7b8d1f1d4d44643b07fa10ca7d021fb e9746973ac574c6b8a9e8857f56a7608 - - -] 10.11.10.1 "POST /v2/e9746973ac574c6b8a9e8857f56a7608/os-server-external-events HTTP/1.1" status: 200 len: 380 time: 0.0879810
- openstack_47 | blacklisted=true | diff=-1.5 | current=23.0 | best=1.0
  current: nova-scheduler.log.1.2017-05-16_13:53:08 2017-05-16 00:13:09.162 25998 INFO nova.scheduler.host_manager [req-c34edf86-25ac-4c79-ab19-9ea9caa4eddf - - - - -] The instance sync for host 'cp-1.slowvm1.tcloud-pg0.utah.cloudlab.us' did not match. Re-created its InstanceList.
  alt: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:09.118 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: 127e769a-4fe6-4548-93b1-513ac51e0452] VM Started (Lifecycle Event)

### OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD
- openstack_51 | blacklisted=false | diff=5.5 | current=9.0 | best=0.0
  current: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:16.313 2931 INFO nova.virt.libvirt.driver [-] [instance: 127e769a-4fe6-4548-93b1-513ac51e0452] Instance spawned successfully.
  alt: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:15.561 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] Active base files: /var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742
- openstack_49 | blacklisted=false | diff=3.0 | current=9.0 | best=0.0
  current: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:57.756 2931 INFO nova.virt.libvirt.driver [-] [instance: c62f4f25-982c-4ea2-b5e4-93000edfcfbf] Instance spawned successfully.
  alt: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:55.325 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] Active base files: /var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742
- openstack_win_0 | blacklisted=false | diff=-8.5 | current=13.5 | best=0.0
  current: nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:09.660 2931 WARNING nova.compute.manager [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] While synchronizing instance power states, found 1 instances in the database and 0 instances on the hypervisor.
  alt: nova-api.log.2017-05-14_21:27:04 2017-05-14 19:39:09.330 25746 INFO nova.osapi_compute.wsgi.server [req-c7d137bb-3573-427e-876c-c861aff95551 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2636831
- openstack_54 | blacklisted=false | diff=-11.5 | current=14.5 | best=0.0
  current: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:14:33.392 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: faf974ea-cba5-4e1b-93f4-3a3bc606006f] During sync_power_state the instance has a pending task (spawning). Skip.
  alt: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:14:35.138 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): checking
- openstack_50 | blacklisted=false | diff=-16.0 | current=14.5 | best=0.0
  current: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:50.867 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: c62f4f25-982c-4ea2-b5e4-93000edfcfbf] During sync_power_state the instance has a pending task (spawning). Skip.
  alt: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:13:55.140 2931 INFO nova.virt.libvirt.imagecache [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] image 0673dd71-34c5-4fbb-86c4-40623fbe45b4 at (/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742): checking

### OPENSTACK_SCALE_METADATA_SERVICE
- openstack_58 | blacklisted=false | diff=-13.5 | current=15.5 | best=0.0
  current: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:12:41.470 25777 INFO nova.metadata.wsgi.server [req-27525a89-b0a1-4d18-8263-f2bc3079f9dc - - - - -] 10.11.21.140,10.11.10.1 "GET /openstack/2013-10-17 HTTP/1.1" status: 200 len: 157 time: 0.2332909
  alt: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:12:35.663 25746 INFO nova.osapi_compute.wsgi.server [req-537bc103-6c6d-4c9d-a801-52aa85533601 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1910 time: 0.2428820
- openstack_64 | blacklisted=false | diff=-13.5 | current=15.5 | best=0.0
  current: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:13:22.695 25795 INFO nova.metadata.wsgi.server [-] 10.11.21.141,10.11.10.1 "GET /openstack/2013-10-17 HTTP/1.1" status: 200 len: 157 time: 0.0009151
  alt: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:13:17.448 25746 INFO nova.osapi_compute.wsgi.server [req-5b273dce-4719-47ba-acc2-276214bd6c87 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1910 time: 0.2577119
- openstack_52 | blacklisted=false | diff=-9.7 | current=15.5 | best=10.5
  current: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:14:04.140 25797 INFO nova.metadata.wsgi.server [-] 10.11.21.142,10.11.10.1 "GET /openstack/2013-10-17 HTTP/1.1" status: 200 len: 157 time: 0.0008900
  alt: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:14:04.129 25797 INFO nova.metadata.wsgi.server [req-3d4a0060-5493-49ef-bcfe-3661c420a8dc - - - - -] 10.11.21.142,10.11.10.1 "GET /openstack/2012-08-10/meta_data.json HTTP/1.1" status: 200 len: 264 time: 0.2294021
- openstack_59 | blacklisted=false | diff=-11.5 | current=15.5 | best=10.5
  current: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:14:45.462 25777 INFO nova.metadata.wsgi.server [-] 10.11.21.143,10.11.10.1 "GET /openstack/2013-10-17 HTTP/1.1" status: 200 len: 157 time: 0.0009041
  alt: nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:14:45.452 25777 INFO nova.metadata.wsgi.server [req-a9f11ae2-996a-4e7e-acf5-453f7a2ff027 - - - - -] 10.11.21.143,10.11.10.1 "GET /openstack/2012-08-10/meta_data.json HTTP/1.1" status: 200 len: 264 time: 0.2402911

### OPENSTACK_VERIFY_TERMINATION_AND_RESTORE
- openstack_56 | blacklisted=false | diff=4.0 | current=9.0 | best=1.0
  current: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:12:01.590 2931 INFO nova.compute.manager [req-121ecfae-3fb1-49cc-9a78-8b046fe73a77 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] [instance: d96a117b-0193-4549-bdcc-63b917273d1d] Terminating instance
  alt: nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:12:02.514 2931 INFO nova.virt.libvirt.driver [req-121ecfae-3fb1-49cc-9a78-8b046fe73a77 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] [instance: d96a117b-0193-4549-bdcc-63b917273d1d] Deleting instance files /var/lib/nova/instances/d96a117b-0193-4549-bdcc-63b917273d1d_del
