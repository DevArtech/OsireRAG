# Wait for the DIGITS server to start
# echo "Waiting for RosieRAG server to open port ${port}..."
# echo "TIMING - Starting wait at: $(date)"
# if wait_until_port_used "${host}:${port}" 60; then
#   echo "Discovered RosieRAG server listening on port ${port}!"
#   echo "TIMING - Wait ended at: $(date)"
# else
#   echo "Timed out waiting for RosieRAG server to open port ${port}!"
#   echo "TIMING - Wait ended at: $(date)"
#   pkill -P ${SCRIPT_PID}
#   clean_up 1
# fi
# sleep 2
