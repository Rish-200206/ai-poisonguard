import React, { useMemo } from 'react';

function LabelHeatmap({ heatmapData }) {
  if (!heatmapData || !heatmapData.data) return null;

  const { features, labels, data } = heatmapData;

  // Use first 6 features max for readability
  const displayFeatures = features.slice(0, 6);

  const getIntensityColor = (count, maxCount) => {
    if (maxCount === 0) return 'rgba(59, 130, 246, 0.05)';
    const ratio = count / maxCount;
    if (ratio > 0.7) return 'rgba(239, 68, 68, 0.7)';
    if (ratio > 0.5) return 'rgba(249, 115, 22, 0.5)';
    if (ratio > 0.3) return 'rgba(234, 179, 8, 0.4)';
    if (ratio > 0.1) return 'rgba(59, 130, 246, 0.3)';
    return 'rgba(59, 130, 246, 0.08)';
  };

  // Compute global max for color scaling
  const globalMax = useMemo(() => {
    let max = 0;
    for (const fname of displayFeatures) {
      const bins = data[fname] || [];
      for (const bin of bins) {
        max = Math.max(max, bin.total);
      }
    }
    return max;
  }, [data, displayFeatures]);

  return (
    <div className="glass-card p-6 animate-fadeIn stagger-3">
      <h3 className="text-sm font-semibold uppercase tracking-wider mb-4"
        style={{ color: 'var(--text-secondary)' }}>
        Label Distribution Heatmap
      </h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        Red cells indicate anomalous label concentrations that may signal poisoning
      </p>

      <div style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ minWidth: '500px' }}>
          <thead>
            <tr>
              <th style={{ width: '140px' }}>Feature</th>
              {Array.from({ length: 8 }, (_, i) => (
                <th key={i} style={{ textAlign: 'center', width: '60px' }}>Bin {i + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayFeatures.map((fname) => {
              const bins = data[fname] || [];
              return (
                <tr key={fname}>
                  <td style={{
                    fontWeight: 500,
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    maxWidth: '140px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {fname}
                  </td>
                  {bins.map((bin, i) => {
                    const total = bin.total;
                    const labelCounts = bin.labels || {};
                    const dominantLabel = Object.entries(labelCounts)
                      .sort((a, b) => b[1] - a[1])[0];
                    const purity = dominantLabel
                      ? (dominantLabel[1] / Math.max(total, 1))
                      : 1;
                    // Higher count + lower purity = more anomalous
                    const anomalyIntensity = total * (1 - purity);

                    return (
                      <td key={i} style={{
                        textAlign: 'center',
                        padding: '4px',
                      }}>
                        <div
                          style={{
                            background: getIntensityColor(anomalyIntensity, globalMax * 0.3),
                            borderRadius: '4px',
                            padding: '6px 2px',
                            fontSize: '11px',
                            fontWeight: 500,
                            color: anomalyIntensity > globalMax * 0.15
                              ? 'var(--text-primary)'
                              : 'var(--text-muted)',
                            minHeight: '32px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            transition: 'all 0.2s ease',
                          }}
                          title={`${fname} ${bin.range}\nTotal: ${total}\nLabels: ${JSON.stringify(labelCounts)}`}
                        >
                          {total > 0 ? total : '–'}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-4 text-xs" style={{ color: 'var(--text-muted)' }}>
        <span>Intensity:</span>
        <div className="flex items-center gap-1">
          <div style={{ width: 16, height: 12, borderRadius: 2, background: 'rgba(59, 130, 246, 0.08)' }} />
          <span>Normal</span>
        </div>
        <div className="flex items-center gap-1">
          <div style={{ width: 16, height: 12, borderRadius: 2, background: 'rgba(234, 179, 8, 0.4)' }} />
          <span>Moderate</span>
        </div>
        <div className="flex items-center gap-1">
          <div style={{ width: 16, height: 12, borderRadius: 2, background: 'rgba(239, 68, 68, 0.7)' }} />
          <span>Anomalous</span>
        </div>
      </div>
    </div>
  );
}

export default LabelHeatmap;
