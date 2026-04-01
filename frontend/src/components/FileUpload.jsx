import React, { useState, useCallback } from 'react';
import { Upload, FileText, X, CheckCircle, Database } from 'lucide-react';

function FileUpload({ onFilesSelected, disabled }) {
  const [csvFile, setCsvFile] = useState(null);
  const [modelFile, setModelFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const processFiles = (files) => {
    files.forEach(f => {
      if (f.name.endsWith('.csv')) {
        setCsvFile(f);
      } else if (f.name.endsWith('.pkl') || f.name.endsWith('.h5') || f.name.endsWith('.onnx')) {
        setModelFile(f);
      }
    });
  };

  const handleCsvChange = (e) => {
    if (e.target.files[0]) {
      setCsvFile(e.target.files[0]);
    }
  };

  const handleModelChange = (e) => {
    if (e.target.files[0]) {
      setModelFile(e.target.files[0]);
    }
  };

  const removeCsv = () => setCsvFile(null);
  const removeModel = () => setModelFile(null);

  React.useEffect(() => {
    onFilesSelected({ csv: csvFile, model: modelFile });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [csvFile, modelFile]);

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Drop Zone */}
      <div
        id="drop-zone"
        className={`drop-zone ${dragOver ? 'drag-over' : ''} ${csvFile ? 'has-file' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !disabled && document.getElementById('csv-upload').click()}
        style={{
          opacity: disabled ? 0.5 : 1,
          pointerEvents: disabled ? 'none' : 'auto',
          padding: '32px',
        }}
      >
        <div style={{
          display: 'inline-flex',
          padding: '14px',
          borderRadius: '50%',
          background: csvFile ? 'rgba(52, 211, 153, 0.08)' : 'rgba(99, 102, 241, 0.06)',
          border: `1px solid ${csvFile ? 'rgba(52, 211, 153, 0.15)' : 'rgba(99, 102, 241, 0.1)'}`,
          marginBottom: '12px',
        }}>
          {csvFile ? (
            <Database size={24} style={{ color: 'var(--accent-green)' }} />
          ) : (
            <Upload size={24} style={{ color: 'var(--accent-indigo)', opacity: 0.6 }} />
          )}
        </div>
        <p style={{
          fontSize: '14px',
          color: csvFile ? 'var(--accent-green)' : 'var(--text-secondary)',
          fontWeight: 600,
        }}>
          {csvFile ? 'Files loaded — ready to scan' : 'Drop CSV dataset here, or click to browse'}
        </p>
        <p style={{ fontSize: '11px', marginTop: '4px', color: 'var(--text-muted)' }}>
          Accepts .csv (dataset) and .pkl / .h5 / .onnx (ML model)
        </p>
      </div>

      {/* Hidden inputs */}
      <input id="csv-upload" type="file" accept=".csv" onChange={handleCsvChange} hidden />
      <input id="model-upload" type="file" accept=".pkl,.h5,.onnx" onChange={handleModelChange} hidden />

      {/* File chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {csvFile && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 14px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(52, 211, 153, 0.06)',
            border: '1px solid rgba(52, 211, 153, 0.2)',
          }}>
            <CheckCircle size={14} style={{ color: 'var(--accent-green)' }} />
            <FileText size={14} style={{ color: 'var(--text-secondary)' }} />
            <span style={{ fontWeight: 600, color: 'var(--accent-green)', fontSize: '13px' }}>
              {csvFile.name}
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
              ({formatSize(csvFile.size)})
            </span>
            {!disabled && (
              <button onClick={(e) => { e.stopPropagation(); removeCsv(); }}
                style={{
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  background: 'none',
                  border: 'none',
                  padding: '2px',
                  display: 'flex',
                  marginLeft: '2px',
                }}>
                <X size={14} />
              </button>
            )}
          </div>
        )}
        {modelFile && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 14px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(167, 139, 250, 0.06)',
            border: '1px solid rgba(167, 139, 250, 0.2)',
          }}>
            <CheckCircle size={14} style={{ color: 'var(--accent-purple)' }} />
            <FileText size={14} style={{ color: 'var(--text-secondary)' }} />
            <span style={{ fontWeight: 600, color: 'var(--accent-purple)', fontSize: '13px' }}>
              {modelFile.name}
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
              ({formatSize(modelFile.size)})
            </span>
            {!disabled && (
              <button onClick={(e) => { e.stopPropagation(); removeModel(); }}
                style={{
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  background: 'none',
                  border: 'none',
                  padding: '2px',
                  display: 'flex',
                  marginLeft: '2px',
                }}>
                <X size={14} />
              </button>
            )}
          </div>
        )}
        {!modelFile && csvFile && (
          <button
            id="add-model-btn"
            onClick={() => document.getElementById('model-upload').click()}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 14px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(167, 139, 250, 0.04)',
              border: '1px dashed rgba(167, 139, 250, 0.25)',
              color: 'var(--accent-purple)',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 500,
            }}
          >
            <Upload size={14} />
            <span>+ Add model (.pkl / .h5 / .onnx)</span>
          </button>
        )}
      </div>
    </div>
  );
}

export default FileUpload;
