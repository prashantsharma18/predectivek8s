CPU_QUERY = "avg(rate(container_cpu_usage_seconds_total[1m])) by (pod)"
MEMORY_QUERY = "avg(container_memory_working_set_bytes) by (pod)"
RABBITMQ_QUERY = "rabbitmq_queue_messages"
MODEL_STORAGE = "/models/"
