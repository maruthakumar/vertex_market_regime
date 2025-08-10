# Operational Procedures - Adaptive Market Regime System

**Version**: 1.0  
**Last Updated**: 2025-06-26  
**Audience**: Operations Team  

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Monitoring & Alerting](#monitoring--alerting)
3. [Incident Response](#incident-response)
4. [Maintenance Procedures](#maintenance-procedures)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Emergency Procedures](#emergency-procedures)
7. [Performance Tuning](#performance-tuning)
8. [Backup & Recovery](#backup--recovery)

## Daily Operations

### Morning Checklist (Start of Day)

1. **System Health Check** (5 minutes)
   ```bash
   # Check service status
   systemctl status adaptive_regime
   
   # Check API health
   curl http://localhost:8080/api/v1/health
   
   # Review overnight alerts
   ```

2. **Performance Review** (10 minutes)
   - Open Grafana dashboard: http://monitoring-server:3000
   - Check key metrics:
     - Regime detection accuracy (should be >85%)
     - Average latency (should be <100ms)
     - Error rate (should be <1%)
     - Resource usage (CPU <80%, Memory <80%)

3. **Log Review** (10 minutes)
   ```bash
   # Check for errors in last 12 hours
   journalctl -u adaptive_regime --since "12 hours ago" | grep -i error
   
   # Check application logs
   tail -n 1000 /var/log/adaptive_regime/error.log | grep -E "ERROR|CRITICAL"
   ```

4. **Database Health** (5 minutes)
   ```bash
   # Check HeavyDB connections
   curl http://localhost:8080/api/v1/system/database
   
   # Verify data freshness
   echo "SELECT MAX(timestamp) FROM regime_history;" | heavysql -p HyperInteractive
   ```

### Hourly Checks

Run automated health check every hour:
```bash
/opt/adaptive_regime/scripts/health_check.sh
```

This script checks:
- Service availability
- API response time
- Component status
- Queue depths
- Error rates

### End of Day Procedures

1. **Daily Report Generation** (15 minutes)
   ```bash
   # Generate daily performance report
   python /opt/adaptive_regime/scripts/daily_report.py
   
   # Email to stakeholders
   ```

2. **Backup Verification**
   ```bash
   # Verify today's backup completed
   ls -la /backup/adaptive_regime/$(date +%Y%m%d)
   ```

3. **Capacity Planning Check**
   - Review resource usage trends
   - Check disk space availability
   - Plan for upcoming maintenance

## Monitoring & Alerting

### Key Metrics to Monitor

| Metric | Warning Threshold | Critical Threshold | Check Frequency |
|--------|------------------|-------------------|-----------------|
| Service Uptime | <99.9% | <99% | Continuous |
| API Latency (p99) | >150ms | >200ms | Every minute |
| Error Rate | >2% | >5% | Every 5 minutes |
| CPU Usage | >70% | >85% | Every minute |
| Memory Usage | >70% | >90% | Every minute |
| Disk Space | <20% free | <10% free | Every 5 minutes |
| Queue Depth | >500 | >1000 | Every minute |
| Regime Accuracy | <85% | <80% | Every 15 minutes |

### Alert Response Procedures

#### Critical Alerts

**RegimeSystemDown**
1. Verify alert is accurate:
   ```bash
   systemctl status adaptive_regime
   curl http://localhost:8080/api/v1/health
   ```

2. If service is down:
   ```bash
   # Check logs for crash reason
   journalctl -u adaptive_regime -n 100
   
   # Attempt restart
   systemctl restart adaptive_regime
   
   # If restart fails, check:
   - Disk space: df -h
   - Memory: free -h
   - Database connectivity
   ```

3. Escalate if service won't start within 15 minutes

**DatabaseConnectionLost**
1. Verify HeavyDB is running:
   ```bash
   systemctl status heavydb
   ```

2. Test connection manually:
   ```bash
   heavysql -h localhost -p 6274 -u admin -p HyperInteractive
   ```

3. Check network connectivity and firewall rules

4. Review connection pool settings

#### Warning Alerts

**HighLatency**
1. Check current load:
   ```bash
   # API request rate
   curl http://localhost:9090/api/v1/query?query=rate(regime_requests_total[5m])
   
   # System load
   uptime
   ```

2. Review slow queries in logs

3. Consider scaling if load is consistently high

**HighMemoryUsage**
1. Check for memory leaks:
   ```bash
   # Process memory over time
   ps aux | grep regime
   
   # Detailed memory map
   pmap -x $(pgrep -f adaptive_regime)
   ```

2. Review recent changes or deployments

3. Plan for restart during maintenance window if needed

### Monitoring Tools Access

| Tool | URL | Username | Purpose |
|------|-----|----------|---------|
| Grafana | http://monitoring:3000 | admin | Visual dashboards |
| Prometheus | http://monitoring:9090 | - | Metrics queries |
| AlertManager | http://monitoring:9093 | - | Alert management |
| Application Logs | /var/log/adaptive_regime/ | - | Detailed logs |

## Incident Response

### Incident Classification

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| P1 - Critical | Complete service outage | 15 minutes | System down, data loss |
| P2 - High | Major functionality impaired | 1 hour | High error rate, component failure |
| P3 - Medium | Minor functionality impaired | 4 hours | Degraded performance |
| P4 - Low | No immediate impact | 24 hours | Non-critical alerts |

### Incident Response Process

1. **Detection & Alert**
   - Automated alert received
   - Manual detection reported
   - User complaint

2. **Initial Assessment** (5 minutes)
   - Verify the issue
   - Determine severity
   - Start incident timer

3. **Notification**
   - P1: Page on-call engineer immediately
   - P2: Notify team lead within 30 minutes
   - P3-P4: Email notification

4. **Investigation**
   ```bash
   # Collect diagnostic information
   /opt/adaptive_regime/scripts/collect_diagnostics.sh
   
   # Review recent changes
   git log --oneline -n 20
   
   # Check deployment history
   cat /var/log/adaptive_regime/deployments.log
   ```

5. **Resolution**
   - Apply fix or workaround
   - Verify resolution
   - Monitor for recurrence

6. **Post-Incident**
   - Document timeline and actions
   - Conduct root cause analysis (RCA)
   - Create follow-up tickets

### Common Issues and Resolutions

#### Issue: High Latency Spikes

**Symptoms**: P99 latency >200ms, user complaints

**Resolution**:
```bash
# 1. Check current regime
curl http://localhost:8080/api/v1/regime/current

# 2. Review component performance
curl http://localhost:8080/api/v1/components

# 3. Restart slow components
systemctl restart adaptive_regime

# 4. If persists, scale workers
vim /etc/adaptive_regime/config.yaml
# Increase: max_workers: 8
systemctl reload adaptive_regime
```

#### Issue: Memory Leak

**Symptoms**: Gradually increasing memory usage

**Resolution**:
```bash
# 1. Confirm memory growth
watch -n 10 'ps aux | grep regime'

# 2. Schedule restart
echo "systemctl restart adaptive_regime" | at 02:00

# 3. Collect heap dump for analysis
kill -USR1 $(pgrep -f adaptive_regime)

# 4. Report to development team
```

## Maintenance Procedures

### Weekly Maintenance

**Every Monday - 30 minutes**

1. **Performance Analysis**
   ```bash
   # Generate weekly performance report
   python /opt/adaptive_regime/scripts/weekly_analysis.py
   ```

2. **Log Rotation Verification**
   ```bash
   # Check log rotation is working
   ls -la /var/log/adaptive_regime/*.gz
   
   # Clean old logs if needed
   find /var/log/adaptive_regime -name "*.gz" -mtime +30 -delete
   ```

3. **Database Maintenance**
   ```sql
   -- Check table sizes
   SELECT table_name, 
          pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) 
   FROM information_schema.tables 
   WHERE table_schema = 'public';
   
   -- Update statistics
   ANALYZE regime_history;
   ```

### Monthly Maintenance

**First Tuesday of Month - 2 hours (maintenance window)**

1. **System Updates**
   ```bash
   # Update system packages
   apt update && apt upgrade -y
   
   # Update Python dependencies
   source /opt/adaptive_regime/venv/bin/activate
   pip list --outdated
   # Only update after testing in staging
   ```

2. **Performance Baseline**
   ```bash
   # Run performance benchmark
   python /opt/adaptive_regime/scripts/performance_benchmark.py
   
   # Compare with previous baseline
   ```

3. **Configuration Review**
   - Review and optimize thresholds
   - Update regime parameters if needed
   - Clean up unused configurations

4. **Security Patches**
   - Apply security updates
   - Rotate API keys and passwords
   - Review access logs

### Quarterly Maintenance

**Every 3 months - 4 hours**

1. **Full System Validation**
   ```bash
   python /opt/adaptive_regime/deployment/validate_deployment.py
   ```

2. **Disaster Recovery Test**
   - Test backup restoration
   - Verify failover procedures
   - Update documentation

3. **Capacity Planning**
   - Review growth trends
   - Plan infrastructure upgrades
   - Budget for next quarter

## Troubleshooting Guide

### Diagnostic Commands

```bash
# System Overview
/opt/adaptive_regime/scripts/system_status.sh

# Component Health
curl http://localhost:8080/api/v1/components | jq

# Recent Errors
journalctl -u adaptive_regime --since "1 hour ago" | grep ERROR

# Database Queries
echo "SELECT COUNT(*) FROM regime_history WHERE timestamp > NOW() - INTERVAL '1 hour';" | heavysql

# Network Connections
netstat -tulpn | grep 8080

# Process Details
ps aux | grep regime
lsof -p $(pgrep -f adaptive_regime)

# Disk I/O
iotop -p $(pgrep -f adaptive_regime)
```

### Common Problems

#### Problem: Service Won't Start

```bash
# 1. Check syntax
/opt/adaptive_regime/venv/bin/python -m py_compile /opt/adaptive_regime/app/*.py

# 2. Verify configuration
/opt/adaptive_regime/venv/bin/python -c "import yaml; yaml.safe_load(open('/etc/adaptive_regime/config.yaml'))"

# 3. Check permissions
ls -la /opt/adaptive_regime/
ls -la /var/log/adaptive_regime/

# 4. Test database connection
python /opt/adaptive_regime/scripts/test_db_connection.py

# 5. Check port availability
netstat -tulpn | grep 8080
```

#### Problem: Regime Detection Errors

```bash
# 1. Verify data quality
curl http://localhost:8080/api/v1/data/quality

# 2. Check feature calculations
tail -f /var/log/adaptive_regime/features.log

# 3. Review recent model updates
ls -la /opt/adaptive_regime/models/

# 4. Test with known data
python /opt/adaptive_regime/scripts/test_regime_detection.py
```

## Emergency Procedures

### Complete System Failure

1. **Immediate Actions** (5 minutes)
   ```bash
   # Verify failure
   curl http://localhost:8080/api/v1/health || echo "System is down"
   
   # Check if server is responsive
   ping localhost
   ssh regime@localhost
   ```

2. **Failover to Backup** (if available)
   ```bash
   # Update load balancer to backup server
   # Or update DNS to point to backup
   ```

3. **Emergency Restart** (10 minutes)
   ```bash
   # Force restart
   systemctl stop adaptive_regime
   pkill -f adaptive_regime  # Force kill if needed
   systemctl start adaptive_regime
   
   # If fails, try safe mode
   SAFE_MODE=1 systemctl start adaptive_regime
   ```

4. **Data Recovery** (if needed)
   ```bash
   # Restore from latest backup
   /opt/adaptive_regime/scripts/restore_backup.sh latest
   ```

### Security Breach

1. **Isolate System**
   ```bash
   # Block external access
   iptables -I INPUT -p tcp --dport 8080 -j DROP
   
   # Keep internal monitoring
   iptables -I INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
   ```

2. **Collect Evidence**
   ```bash
   # Snapshot logs
   tar -czf /tmp/security_incident_$(date +%Y%m%d_%H%M%S).tar.gz /var/log/
   
   # Check for unauthorized access
   last -50
   grep "Failed password" /var/log/auth.log
   ```

3. **Rotate Credentials**
   - Change all passwords
   - Regenerate API keys
   - Update configuration files

4. **Restore Service**
   - Apply security patches
   - Re-enable access gradually
   - Monitor closely

## Performance Tuning

### Quick Optimizations

1. **Increase Worker Processes**
   ```yaml
   # /etc/adaptive_regime/config.yaml
   performance:
     max_workers: 8  # Increase from 4
     batch_size: 2000  # Increase from 1000
   ```

2. **Optimize Database Queries**
   ```bash
   # Add missing indexes
   echo "CREATE INDEX idx_regime_timestamp_regime ON regime_history(timestamp, regime_id);" | heavysql
   ```

3. **Enable Caching**
   ```yaml
   # /etc/adaptive_regime/config.yaml
   caching:
     enabled: true
     ttl: 300  # 5 minutes
     max_size: 1000
   ```

### Advanced Tuning

1. **JVM Optimization** (if using Java components)
   ```bash
   export JVM_OPTS="-Xms4g -Xmx8g -XX:+UseG1GC"
   ```

2. **Network Optimization**
   ```bash
   # Increase connection limits
   sysctl -w net.core.somaxconn=1024
   sysctl -w net.ipv4.tcp_max_syn_backlog=1024
   ```

3. **Profile Application**
   ```bash
   # Enable profiling
   PROFILE=1 systemctl restart adaptive_regime
   
   # Analyze results after 1 hour
   python /opt/adaptive_regime/scripts/analyze_profile.py
   ```

## Backup & Recovery

### Backup Procedures

**Daily Automated Backup** (Runs at 2 AM)
```bash
# Backup script location
/opt/adaptive_regime/scripts/daily_backup.sh

# Manual backup
/opt/adaptive_regime/scripts/backup.sh manual_$(date +%Y%m%d_%H%M%S)
```

**What's Backed Up:**
- Configuration files
- Model files
- Application state
- Recent logs
- Database dump (regime_history)

### Recovery Procedures

1. **Configuration Recovery**
   ```bash
   # List available backups
   ls -la /backup/adaptive_regime/
   
   # Restore configuration
   cp -r /backup/adaptive_regime/20250626/config/* /etc/adaptive_regime/
   ```

2. **Full System Recovery**
   ```bash
   # Stop service
   systemctl stop adaptive_regime
   
   # Restore all components
   /opt/adaptive_regime/scripts/restore_backup.sh 20250626
   
   # Restart service
   systemctl start adaptive_regime
   
   # Verify
   /opt/adaptive_regime/deployment/validate_deployment.py
   ```

3. **Point-in-Time Recovery**
   ```bash
   # Restore to specific time
   /opt/adaptive_regime/scripts/restore_to_time.sh "2025-06-26 14:30:00"
   ```

## Appendix

### Useful Commands Reference

```bash
# Service Management
systemctl {start|stop|restart|status} adaptive_regime

# Log Viewing
journalctl -u adaptive_regime -f  # Follow logs
tail -f /var/log/adaptive_regime/system.log

# API Testing
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/api/v1/regime/current
curl http://localhost:8080/api/v1/metrics

# Database Queries
heavysql -h localhost -p 6274 -u admin -p HyperInteractive

# Performance Testing
ab -n 1000 -c 10 http://localhost:8080/api/v1/regime/current

# System Resources
htop
iotop
nethogs
```

### Contact Information

| Role | Name | Email | Phone | Escalation |
|------|------|-------|-------|------------|
| On-Call Engineer | Rotation | oncall@company.com | +1-555-0100 | Primary |
| Team Lead | John Smith | john.smith@company.com | +1-555-0101 | Secondary |
| Platform Architect | Jane Doe | jane.doe@company.com | +1-555-0102 | Tertiary |
| VP Engineering | Bob Wilson | bob.wilson@company.com | +1-555-0103 | Executive |

### External Dependencies

| Service | Contact | SLA | Notes |
|---------|---------|-----|-------|
| HeavyDB Support | support@heavy.ai | 4 hours | Premium support |
| Cloud Provider | AWS Support | 1 hour | Enterprise support |
| Monitoring | Datadog Support | 2 hours | Pro tier |

---

**Document Version**: 1.0  
**Last Updated**: 2025-06-26  
**Next Review**: 2025-07-26  
**Owner**: Platform Operations Team