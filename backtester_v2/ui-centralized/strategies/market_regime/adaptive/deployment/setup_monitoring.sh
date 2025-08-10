#!/bin/bash
# Monitoring Setup Script for Adaptive Market Regime System

set -e

# Configuration
MONITORING_DIR="/opt/adaptive_regime/monitoring"
PROMETHEUS_VERSION="2.37.0"
ALERTMANAGER_VERSION="0.24.0"
GRAFANA_VERSION="9.1.0"
NODE_EXPORTER_VERSION="1.3.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root"
   exit 1
fi

log_info "Starting monitoring setup for Adaptive Market Regime System"

# Create monitoring user
if ! id -u prometheus >/dev/null 2>&1; then
    log_info "Creating prometheus user"
    useradd --no-create-home --shell /bin/false prometheus
fi

# Create directories
log_info "Creating monitoring directories"
mkdir -p /etc/prometheus /var/lib/prometheus
mkdir -p /etc/alertmanager /var/lib/alertmanager
mkdir -p /etc/grafana /var/lib/grafana
mkdir -p ${MONITORING_DIR}/{prometheus,alertmanager,grafana}

# Install Prometheus
install_prometheus() {
    log_info "Installing Prometheus ${PROMETHEUS_VERSION}"
    
    cd /tmp
    wget -q https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz
    tar xzf prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz
    
    cp prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus /usr/local/bin/
    cp prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool /usr/local/bin/
    
    # Copy config files
    cp ${MONITORING_DIR}/prometheus_config.yml /etc/prometheus/prometheus.yml
    cp -r ${MONITORING_DIR}/alerts /etc/prometheus/
    
    # Set permissions
    chown -R prometheus:prometheus /etc/prometheus /var/lib/prometheus
    
    # Create systemd service
    cat > /etc/systemd/system/prometheus.service << EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file /etc/prometheus/prometheus.yml \
    --storage.tsdb.path /var/lib/prometheus/ \
    --web.console.templates=/etc/prometheus/consoles \
    --web.console.libraries=/etc/prometheus/console_libraries \
    --web.enable-lifecycle

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Cleanup
    rm -rf prometheus-${PROMETHEUS_VERSION}.linux-amd64*
    
    log_info "Prometheus installed successfully"
}

# Install AlertManager
install_alertmanager() {
    log_info "Installing AlertManager ${ALERTMANAGER_VERSION}"
    
    cd /tmp
    wget -q https://github.com/prometheus/alertmanager/releases/download/v${ALERTMANAGER_VERSION}/alertmanager-${ALERTMANAGER_VERSION}.linux-amd64.tar.gz
    tar xzf alertmanager-${ALERTMANAGER_VERSION}.linux-amd64.tar.gz
    
    cp alertmanager-${ALERTMANAGER_VERSION}.linux-amd64/alertmanager /usr/local/bin/
    cp alertmanager-${ALERTMANAGER_VERSION}.linux-amd64/amtool /usr/local/bin/
    
    # Copy config
    cp ${MONITORING_DIR}/alertmanager_config.yml /etc/alertmanager/alertmanager.yml
    
    # Set permissions
    chown -R prometheus:prometheus /etc/alertmanager /var/lib/alertmanager
    
    # Create systemd service
    cat > /etc/systemd/system/alertmanager.service << EOF
[Unit]
Description=AlertManager
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/alertmanager \
    --config.file=/etc/alertmanager/alertmanager.yml \
    --storage.path=/var/lib/alertmanager/

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Cleanup
    rm -rf alertmanager-${ALERTMANAGER_VERSION}.linux-amd64*
    
    log_info "AlertManager installed successfully"
}

# Install Node Exporter
install_node_exporter() {
    log_info "Installing Node Exporter ${NODE_EXPORTER_VERSION}"
    
    cd /tmp
    wget -q https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    tar xzf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    
    cp node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
    
    # Create systemd service
    cat > /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/node_exporter

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Cleanup
    rm -rf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64*
    
    log_info "Node Exporter installed successfully"
}

# Install Grafana
install_grafana() {
    log_info "Installing Grafana ${GRAFANA_VERSION}"
    
    # Add Grafana GPG key
    wget -q -O - https://packages.grafana.com/gpg.key | apt-key add -
    echo "deb https://packages.grafana.com/oss/deb stable main" | tee /etc/apt/sources.list.d/grafana.list
    
    # Install Grafana
    apt-get update -qq
    apt-get install -y grafana
    
    # Configure Grafana
    cat > /etc/grafana/grafana.ini << EOF
[server]
http_port = 3000
domain = localhost

[security]
admin_user = admin
admin_password = regime_admin_2024

[auth.anonymous]
enabled = false

[dashboards]
versions_to_keep = 20

[users]
allow_sign_up = false

[log]
mode = console file
level = info

[metrics]
enabled = true
EOF

    # Import dashboard
    mkdir -p /etc/grafana/provisioning/dashboards
    mkdir -p /var/lib/grafana/dashboards
    
    cp ${MONITORING_DIR}/grafana/regime_dashboard.json /var/lib/grafana/dashboards/
    
    # Create dashboard provisioning config
    cat > /etc/grafana/provisioning/dashboards/regime.yaml << EOF
apiVersion: 1

providers:
  - name: 'Regime System'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
EOF

    # Create datasource provisioning config
    cat > /etc/grafana/provisioning/datasources/prometheus.yaml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
EOF

    # Set permissions
    chown -R grafana:grafana /etc/grafana /var/lib/grafana
    
    log_info "Grafana installed successfully"
}

# Configure firewall
configure_firewall() {
    log_info "Configuring firewall rules"
    
    if command -v ufw &> /dev/null; then
        ufw allow 9090/tcp comment 'Prometheus'
        ufw allow 9093/tcp comment 'AlertManager'
        ufw allow 3000/tcp comment 'Grafana'
        ufw allow 9100/tcp comment 'Node Exporter'
        log_info "Firewall rules added"
    else
        log_warn "UFW not found, please configure firewall manually"
    fi
}

# Start services
start_services() {
    log_info "Starting monitoring services"
    
    systemctl daemon-reload
    
    # Enable and start services
    services=("prometheus" "alertmanager" "node_exporter" "grafana-server")
    
    for service in "${services[@]}"; do
        systemctl enable $service
        systemctl start $service
        
        if systemctl is-active --quiet $service; then
            log_info "$service started successfully"
        else
            log_error "Failed to start $service"
            systemctl status $service --no-pager
        fi
    done
}

# Verify installation
verify_installation() {
    log_info "Verifying monitoring installation"
    
    echo ""
    echo "Service Status:"
    echo "--------------"
    systemctl status prometheus --no-pager | grep Active
    systemctl status alertmanager --no-pager | grep Active
    systemctl status node_exporter --no-pager | grep Active
    systemctl status grafana-server --no-pager | grep Active
    
    echo ""
    echo "Access URLs:"
    echo "-----------"
    echo "Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
    echo "AlertManager: http://$(hostname -I | awk '{print $1}'):9093"
    echo "Grafana: http://$(hostname -I | awk '{print $1}'):3000"
    echo ""
    echo "Grafana Credentials:"
    echo "Username: admin"
    echo "Password: regime_admin_2024"
    echo ""
    
    # Test Prometheus targets
    sleep 5
    if curl -s http://localhost:9090/api/v1/targets | grep -q "up"; then
        log_info "Prometheus targets are up"
    else
        log_warn "Some Prometheus targets may be down"
    fi
}

# Main execution
main() {
    log_info "Installing monitoring components..."
    
    # Update package list
    apt-get update -qq
    
    # Install prerequisites
    apt-get install -y wget curl
    
    # Install components
    install_prometheus
    install_alertmanager
    install_node_exporter
    install_grafana
    
    # Configure firewall
    configure_firewall
    
    # Start services
    start_services
    
    # Verify installation
    verify_installation
    
    log_info "Monitoring setup completed successfully!"
    log_info "Please update AlertManager configuration with your notification endpoints"
}

# Run main
main