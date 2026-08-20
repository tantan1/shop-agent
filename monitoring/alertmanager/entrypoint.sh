#!/bin/sh
set -e

# 用 sed 替换 alertmanager.yml.template 中的环境变量占位符
# 将 ${ALERTMANAGER_WEBHOOK_TOKEN} 替换为实际值
sed -i "s|\${ALERTMANAGER_WEBHOOK_TOKEN}|${ALERTMANAGER_WEBHOOK_TOKEN}|g" /etc/alertmanager/alertmanager.yml.template

# 启动 alertmanager
exec /bin/alertmanager --config.file=/etc/alertmanager/alertmanager.yml.template --storage.path=/alertmanager --log.level=info
