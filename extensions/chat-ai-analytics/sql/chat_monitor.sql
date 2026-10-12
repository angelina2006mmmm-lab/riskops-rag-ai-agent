-- Illustrative monitoring based only on synthetic events.
WITH daily AS (
 SELECT arm,day,COUNT(*) AS sessions,SUM(correct) AS correct,SUM(escalated) AS escalations,
   ROUND(AVG(latency_ms),2) AS avg_latency,
   SUM(CASE WHEN error_code REGEXP 'MISROUTE' THEN 1 ELSE 0 END) AS mismatches
 FROM outcomes GROUP BY arm,day
), running AS (
 SELECT arm,day,sessions,correct,escalations,avg_latency,mismatches,
   SUM(sessions) OVER (PARTITION BY arm ORDER BY day ROWS UNBOUNDED PRECEDING) AS running_sessions,
   SUM(correct) OVER (PARTITION BY arm ORDER BY day ROWS UNBOUNDED PRECEDING) AS running_correct
 FROM daily
)
SELECT arm,day,sessions,correct,escalations,avg_latency,mismatches,running_sessions,running_correct,
 ROUND(1.0*running_correct/NULLIF(running_sessions,0),4) AS running_accuracy,
 (SELECT COUNT(*) FROM outcomes o WHERE o.arm=running.arm AND o.error_code!='NONE') AS all_error_events
FROM running ORDER BY arm,day;
