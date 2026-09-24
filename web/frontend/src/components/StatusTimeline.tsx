import { useEffect, useRef } from "react";
import { JobEvent } from "../types";
import { formatStageLabel } from "../copy";

interface StatusTimelineProps {
  events: JobEvent[];
  status: string;
  statusKey: string;
}

export function StatusTimeline({ events, status, statusKey }: StatusTimelineProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  // 从最新事件提取 FPS 和进度
  const latest = events.length > 0 ? events[events.length - 1] : null;
  const progress = latest?.progress;
  // 优先从实时进度事件取 FPS,其次从完成摘要取
  const fps = latest?.fps ?? latest?.summary?.stats?.fps;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>实时进度</h2>
        <span className={`status-pill ${statusKey}`}>{status}</span>
      </div>

      {(fps !== undefined || progress !== undefined) && (
        <div className="live-stats-bar">
          {progress !== undefined && (
            <div className="live-stat-item">
              <span className="live-stat-label">进度</span>
              <strong className="live-stat-value">{progress}%</strong>
            </div>
          )}
          {fps !== undefined && (
            <div className="live-stat-item">
              <span className="live-stat-label">FPS</span>
              <strong className="live-stat-value">{fps.toFixed(1)}</strong>
            </div>
          )}
        </div>
      )}

      <div className="timeline-scroll" ref={scrollRef}>
        <div className="timeline">
          {events.length === 0 ? (
            <div className="empty-state">等待任务状态更新...</div>
          ) : (
            events.map((event, index) => (
              <article key={`${event.timestamp ?? "event"}-${index}`} className="timeline-item">
                <span className="timeline-dot" />
                <div>
                  <strong>{formatStageLabel(event.stage ?? event.status ?? event.type)}</strong>
                  <p>{event.message ?? "暂无说明"}</p>
                  {typeof event.progress === "number" && (
                    <div className="progress-row">
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${event.progress}%` }} />
                      </div>
                      <span>{event.progress}%</span>
                    </div>
                  )}
                </div>
              </article>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
