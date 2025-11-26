#!/bin/bash
# Monitor the size of the temp directory in real-time
TEMP_DIR="./temp"
echo "Monitoring size of $TEMP_DIR..."
PARTITION="/dev/nvme0n1p6"
watch -n 0.1 "echo '=== Partition ($PARTITION) ==='; df -h $PARTITION; echo ''; echo '=== Temp Directory ($TEMP_DIR) ==='; du -h -s $TEMP_DIR 2>/dev/null || echo 'Temp directory not created yet...'"
