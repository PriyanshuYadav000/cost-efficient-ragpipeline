# IT Operations Policy
## Backup and Recovery

The fictional company Northstar Learning Services performs a full backup of its critical databases every day at 11:00 PM. Incremental backups run every four hours during the day. Backup files are retained for 30 calendar days. The IT operations team checks backup completion logs every morning before 9:00 AM.

A backup is considered successful only after the system confirms that the files can be read. The IT team performs a recovery test once every month. Critical databases must be restored within four hours of an approved recovery request. If a backup fails, the on-call engineer must investigate it within 30 minutes and record the incident.

## Access Permissions

New employees receive access only to the systems required for their roles. A manager must approve each access request before the IT team grants permission. Shared user accounts are not allowed. Temporary access must have an expiry date and should be removed when the approved period ends.

When an employee changes departments, the manager must review the employee's existing permissions within two working days. Access belonging to a departing employee must be disabled by the end of their last working day. Privileged administrator access is reviewed every month by the IT security lead.

## Incident Severity

Severity 1 incidents include a complete outage of a critical customer-facing service or a confirmed breach involving sensitive customer data. The on-call engineer must acknowledge a Severity 1 incident within 10 minutes and immediately notify the incident manager.

Severity 2 incidents affect an important service but have a temporary workaround. The assigned engineer must acknowledge them within 30 minutes. Severity 3 incidents have limited impact and must be acknowledged within four business hours.

Every incident must have an incident record containing the time reported, severity, affected service, responsible owner, investigation notes, and resolution. Severity 1 and Severity 2 incidents require a review within three working days after resolution.

## System Monitoring

The monitoring system checks critical service availability every minute. CPU usage above 85 percent for ten consecutive minutes triggers a warning. Available disk space below 15 percent triggers a high-priority alert. Alerts must be assigned to an engineer and tracked until they are resolved.

The operations team reviews monitoring dashboards during each working day. Alert rules must be reviewed quarterly. Changes to production monitoring settings must be recorded with the reason for the change, the person who approved it, and the date it was applied.

## Change Management

Changes to production systems require a recorded change request. The request must describe the purpose, expected impact, implementation steps, and rollback plan. A service owner must approve the change before it is applied.

Routine changes should be scheduled during the approved maintenance window from 10:00 PM to midnight on Wednesdays. Emergency changes may be applied outside that window when needed to restore a critical service. Emergency changes must be documented by the end of the next working day.

After a production change, the engineer must confirm that the service is operating normally and update the change record. Failed changes must be rolled back when the service cannot be restored safely.
