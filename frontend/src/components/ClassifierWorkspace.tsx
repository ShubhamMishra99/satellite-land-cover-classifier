import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  FileImage,
  Trash2,
  RefreshCw,
  Cpu,
  Layers,
  CheckCircle2,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from 'lucide-react';

interface Top3Item {
  rank: number;
  class: string;
  probability: number;
  percentage: string;
}

interface PredictionData {
  status?: string;
  is_confident?: boolean;
  is_suitable?: boolean;
  threshold_used?: number;
  uncertainty_reason?: string | null;
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  confidence_percentage: string;
  top3: Top3Item[];
  all_probabilities: Record<string, number>;
  model_name: string;
  model_display_name: string;
  original_size: [number, number];
  inference_time_ms: number;
  class_color: string;
  disclaimer: string;
  suitability_stage_failed?: string | null;
  suitability_distance?: number | null;
  suitability_threshold?: number | null;
}

interface SampleItem {
  id: string;
  class_name: string;
  filename: string;
  display_name: string;
  color: string;
  url: string;
}

export const ClassifierWorkspace: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<'resnet18' | 'custom_cnn'>('resnet18');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [showAllClasses, setShowAllClasses] = useState<boolean>(false);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const [sampleTiles, setSampleTiles] = useState<SampleItem[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch available sample tiles from backend
  useEffect(() => {
    fetch('/samples')
      .then((res) => (res.ok ? res.json() : []))
      .then((data: SampleItem[]) => {
        if (Array.isArray(data) && data.length > 0) {
          setSampleTiles(data);
        }
      })
      .catch(() => {
        // Non-critical if offline
      });
  }, []);

  // Handle local file selection
  const handleFileChange = (file: File) => {
    setErrorMsg(null);
    setPrediction(null);

    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
      setErrorMsg('Please select a valid satellite image in JPG, JPEG, or PNG format.');
      return;
    }

    setImageFile(file);

    const objectUrl = URL.createObjectURL(file);
    setImagePreviewUrl(objectUrl);

    // Read image dimensions
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = objectUrl;
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  // Quick select a demo sample tile
  const handleSelectSample = async (sample: SampleItem) => {
    setErrorMsg(null);
    setPrediction(null);
    try {
      const response = await fetch(sample.url);
      if (!response.ok) throw new Error('Failed to fetch sample tile image');
      const blob = await response.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      handleFileChange(file);
    } catch (err: any) {
      setErrorMsg(`Could not load sample tile: ${err.message}`);
    }
  };

  // Reset workspace
  const handleReset = () => {
    setImageFile(null);
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setImagePreviewUrl(null);
    setImageDimensions(null);
    setPrediction(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Submit prediction request to real backend
  const handleAnalyze = async () => {
    if (!imageFile) return;

    setIsLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('model', selectedModel);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errDetail = 'Inference request failed';
        try {
          const errJson = await response.json();
          errDetail = errJson.detail || errDetail;
        } catch {
          // ignore
        }
        throw new Error(errDetail);
      }

      const result: PredictionData = await response.json();
      setPrediction(result);
    } catch (err: any) {
      setErrorMsg(
        err.message || 'Unable to connect to model backend. Make sure the FastAPI server is running on port 8000.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section
      id="classify"
      style={{
        padding: '5rem 0',
        position: 'relative',
        borderTop: '1px solid rgba(255, 255, 255, 0.06)',
      }}
    >
      <div className="container">
        {/* Section Header */}
        <div style={{ textAlign: 'center', maxWidth: '680px', margin: '0 auto 3rem auto' }}>
          <span className="eyebrow-badge" style={{ marginBottom: '0.85rem' }}>
            <Layers size={13} />
            INFERENCE ENGINE
          </span>
          <h2
            style={{
              fontSize: 'clamp(2rem, 3.5vw, 2.75rem)',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              marginBottom: '0.75rem',
            }}
          >
            Satellite Image Intelligence
          </h2>
          <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
            Upload a satellite image to identify its land-use category.
          </p>
        </div>

        {/* Model Selector Bar */}
        <div
          style={{
            maxWidth: '720px',
            margin: '0 auto 2.5rem auto',
            background: 'rgba(8, 16, 36, 0.8)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '0.5rem',
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '0.5rem',
          }}
          role="tablist"
          aria-label="Model Architecture Selection"
        >
          <button
            onClick={() => {
              setSelectedModel('resnet18');
              setPrediction(null);
            }}
            role="tab"
            aria-selected={selectedModel === 'resnet18'}
            style={{
              padding: '0.85rem 1rem',
              borderRadius: '12px',
              border: selectedModel === 'resnet18' ? '1px solid rgba(0, 240, 255, 0.4)' : '1px solid transparent',
              background: selectedModel === 'resnet18' ? 'rgba(0, 240, 255, 0.12)' : 'transparent',
              color: selectedModel === 'resnet18' ? '#FFFFFF' : 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.2rem',
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, fontSize: '0.95rem' }}>
              <Cpu size={16} color={selectedModel === 'resnet18' ? '#00F0FF' : 'currentColor'} />
              <span>ResNet-18 Transfer Learning</span>
            </div>
            <span style={{ fontSize: '0.75rem', color: selectedModel === 'resnet18' ? '#38BDF8' : 'var(--text-muted)' }}>
              Top Performer • 94.99% Test Acc • 11.2M Params
            </span>
          </button>

          <button
            onClick={() => {
              setSelectedModel('custom_cnn');
              setPrediction(null);
            }}
            role="tab"
            aria-selected={selectedModel === 'custom_cnn'}
            style={{
              padding: '0.85rem 1rem',
              borderRadius: '12px',
              border: selectedModel === 'custom_cnn' ? '1px solid rgba(129, 140, 248, 0.4)' : '1px solid transparent',
              background: selectedModel === 'custom_cnn' ? 'rgba(129, 140, 248, 0.12)' : 'transparent',
              color: selectedModel === 'custom_cnn' ? '#FFFFFF' : 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.2rem',
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, fontSize: '0.95rem' }}>
              <Layers size={16} color={selectedModel === 'custom_cnn' ? '#818CF8' : 'currentColor'} />
              <span>Custom CNN Baseline</span>
            </div>
            <span style={{ fontSize: '0.75rem', color: selectedModel === 'custom_cnn' ? '#A5B4FC' : 'var(--text-muted)' }}>
              Trained from Scratch • 75.31% Test Acc • 94.8K Params
            </span>
          </button>
        </div>

        {/* Operating Uncertainty Threshold Indicator */}
        <div
          style={{
            maxWidth: '720px',
            margin: '-1.5rem auto 2.25rem auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.6rem',
            fontSize: '0.78rem',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
            padding: '0.4rem 0.8rem',
            borderRadius: '20px',
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            width: 'fit-content',
          }}
        >
          <span style={{ display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', backgroundColor: selectedModel === 'resnet18' ? '#00F0FF' : '#818CF8' }} />
          <span>
            UNCERTAINTY THRESHOLD: {selectedModel === 'resnet18' ? '60.0% (ResNet-18)' : '40.0% (Custom CNN)'}
          </span>
          <span style={{ color: 'rgba(255, 255, 255, 0.2)' }}>•</span>
          <span style={{ color: 'var(--text-secondary)' }}>Evaluated on EuroSAT Validation Split</span>
        </div>

        {/* Demo Sample Quick-Select Strip */}
        {sampleTiles.length > 0 && (
          <div style={{ marginBottom: '2rem' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '0.75rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <Sparkles size={14} color="#00F0FF" />
                <span>Quick Test with Curated EuroSAT Tiles:</span>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Click any tile to test</span>
            </div>

            <div
              style={{
                display: 'flex',
                gap: '0.65rem',
                overflowX: 'auto',
                paddingBottom: '0.5rem',
              }}
            >
              {sampleTiles.map((sample) => (
                <button
                  key={sample.id}
                  onClick={() => handleSelectSample(sample)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.4rem 0.75rem',
                    borderRadius: '8px',
                    background: 'rgba(11, 20, 44, 0.7)',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    color: '#FFFFFF',
                    fontSize: '0.8rem',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = sample.color;
                    e.currentTarget.style.background = 'rgba(18, 32, 68, 0.9)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                    e.currentTarget.style.background = 'rgba(11, 20, 44, 0.7)';
                  }}
                >
                  <span
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      backgroundColor: sample.color,
                    }}
                  />
                  <span>{sample.class_name}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Dual Panel Workspace */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.25fr)',
            gap: '2rem',
            alignItems: 'stretch',
          }}
          className="workspace-grid"
        >
          {/* =============================================================
              LEFT PANEL: Image Input & Preview
              ============================================================= */}
          <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Satellite Image Input</h3>
              {imageFile && (
                <button
                  onClick={handleReset}
                  className="btn btn-danger"
                  style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
                  title="Remove image"
                >
                  <Trash2 size={14} />
                  <span>Remove</span>
                </button>
              )}
            </div>

            {/* Drag & Drop Zone */}
            {!imagePreviewUrl ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  flex: 1,
                  minHeight: '280px',
                  border: `2px dashed ${isDragOver ? 'var(--cyan)' : 'rgba(56, 189, 248, 0.25)'}`,
                  borderRadius: '14px',
                  background: isDragOver ? 'rgba(0, 240, 255, 0.05)' : 'rgba(6, 12, 28, 0.5)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '2rem',
                  cursor: 'pointer',
                  transition: 'all 0.25s ease',
                  textAlign: 'center',
                }}
              >
                <div
                  style={{
                    width: '64px',
                    height: '64px',
                    borderRadius: '50%',
                    background: 'rgba(0, 240, 255, 0.1)',
                    border: '1px solid rgba(0, 240, 255, 0.3)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '1rem',
                  }}
                >
                  <UploadCloud size={30} color="#00F0FF" />
                </div>
                <p style={{ fontWeight: 600, color: '#FFFFFF', marginBottom: '0.4rem' }}>
                  Drop your satellite image here
                </p>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
                  Supports Sentinel-2 tiles in JPG, JPEG, or PNG
                </p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: '0.85rem', padding: '0.55rem 1.25rem' }}
                >
                  Browse Files
                </button>
              </div>
            ) : (
              /* Image Preview Card */
              <div
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(4, 9, 22, 0.6)',
                  borderRadius: '14px',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  padding: '1.5rem',
                }}
              >
                <div
                  style={{
                    position: 'relative',
                    width: '180px',
                    height: '180px',
                    borderRadius: '12px',
                    overflow: 'hidden',
                    border: '2px solid rgba(0, 240, 255, 0.4)',
                    boxShadow: '0 0 25px rgba(0, 240, 255, 0.2)',
                    marginBottom: '1rem',
                  }}
                >
                  <img
                    src={imagePreviewUrl}
                    alt="Satellite Preview"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      imageRendering: 'pixelated', // Helpful for standard 64x64 EuroSAT tiles
                    }}
                  />
                </div>

                {/* File Information */}
                <div style={{ textAlign: 'center', width: '100%' }}>
                  <div
                    style={{
                      fontWeight: 600,
                      color: '#FFFFFF',
                      fontSize: '0.95rem',
                      maxWidth: '280px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      margin: '0 auto',
                    }}
                  >
                    {imageFile?.name}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8rem',
                      color: 'var(--text-secondary)',
                      marginTop: '0.35rem',
                    }}
                  >
                    {imageDimensions
                      ? `${imageDimensions.width} × ${imageDimensions.height} px`
                      : 'Satellite Tile'}{' '}
                    • {imageFile ? `${(imageFile.size / 1024).toFixed(1)} KB` : ''}
                  </div>
                </div>

                {/* Replace File Button */}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    marginTop: '1.25rem',
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--cyan)',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                  }}
                >
                  <RefreshCw size={13} />
                  <span>Choose a different image</span>
                </button>
              </div>
            )}

            {/* Hidden Input Element */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              style={{ display: 'none' }}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileChange(e.target.files[0]);
                }
              }}
            />

            {/* Action Button */}
            <div style={{ marginTop: '1.5rem' }}>
              <button
                onClick={handleAnalyze}
                disabled={!imageFile || isLoading}
                className="btn btn-primary"
                style={{ width: '100%', padding: '0.9rem 1.5rem' }}
                id="btn-analyze-image"
              >
                {isLoading ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" />
                    <span>Analyzing satellite imagery...</span>
                  </>
                ) : prediction ? (
                  <>
                    <RefreshCw size={18} />
                    <span>Analyze Again</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    <span>Analyze Image</span>
                  </>
                )}
              </button>
            </div>

            {/* Domain Note */}
            <div
              style={{
                marginTop: '1rem',
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.4rem',
                lineHeight: 1.4,
              }}
            >
              <Info size={14} style={{ flexShrink: 0, marginTop: '2px' }} />
              <span>
                Standard EuroSAT imagery consists of 64×64 pixel optical tiles. Uploaded images will be resized and normalized deterministically.
              </span>
            </div>
          </div>

          {/* =============================================================
              RIGHT PANEL: Prediction Results Dashboard
              ============================================================= */}
          <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Classification Results</h3>
              {prediction && (
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.75rem',
                    color: 'var(--cyan)',
                    padding: '0.25rem 0.6rem',
                    borderRadius: '6px',
                    background: 'rgba(0, 240, 255, 0.1)',
                  }}
                >
                  {prediction.inference_time_ms} ms
                </span>
              )}
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div
                style={{
                  padding: '1rem',
                  borderRadius: '10px',
                  background: 'rgba(244, 63, 94, 0.12)',
                  border: '1px solid rgba(244, 63, 94, 0.35)',
                  color: '#FDA4AF',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.65rem',
                  fontSize: '0.9rem',
                  marginBottom: '1rem',
                }}
              >
                <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <div style={{ fontWeight: 600 }}>Inference Notice</div>
                  <div style={{ fontSize: '0.85rem', marginTop: '0.2rem' }}>{errorMsg}</div>
                </div>
              </div>
            )}

            {/* State A: Loading */}
            {isLoading && (
              <div
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '280px',
                  textAlign: 'center',
                }}
              >
                <div
                  style={{
                    width: '60px',
                    height: '60px',
                    borderRadius: '50%',
                    border: '3px solid rgba(0, 240, 255, 0.2)',
                    borderTopColor: 'var(--cyan)',
                    animation: 'spin 1s linear infinite',
                    marginBottom: '1.25rem',
                  }}
                />
                <h4 style={{ fontSize: '1.1rem', marginBottom: '0.35rem' }}>Analyzing satellite imagery...</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Executing forward pass on {selectedModel === 'resnet18' ? 'ResNet-18 Transfer Learning' : 'Custom CNN Baseline'}
                </p>
              </div>
            )}

            {/* State B: Empty State */}
            {!isLoading && !prediction && !errorMsg && (
              <div
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '280px',
                  textAlign: 'center',
                  border: '1px dashed rgba(255, 255, 255, 0.08)',
                  borderRadius: '14px',
                  padding: '2rem',
                }}
              >
                <div
                  style={{
                    width: '56px',
                    height: '56px',
                    borderRadius: '50%',
                    background: 'rgba(255, 255, 255, 0.03)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '1rem',
                  }}
                >
                  <FileImage size={28} color="rgba(255, 255, 255, 0.3)" />
                </div>
                <h4 style={{ fontSize: '1.05rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                  Your classification results will appear here.
                </h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '320px' }}>
                  Select a demo sample tile or upload an image, then click Analyze Image.
                </p>
              </div>
            )}

            {/* State C: Prediction Result Dashboard */}
            {!isLoading && prediction && (
              prediction.is_confident === false ? (
                /* =========================================================
                   Uncertain Prediction View (OOD / Low Confidence Handling)
                   ========================================================= */
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  {/* Uncertain Notice Card */}
                  <div
                    style={{
                      background: 'rgba(26, 18, 7, 0.9)',
                      border: '1px solid rgba(245, 158, 11, 0.45)',
                      borderRadius: '14px',
                      padding: '1.75rem',
                      position: 'relative',
                      overflow: 'hidden',
                      boxShadow: '0 0 30px rgba(245, 158, 11, 0.15)',
                    }}
                  >
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '6px',
                        height: '100%',
                        backgroundColor: '#F59E0B',
                      }}
                    />
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                      <div
                        style={{
                          width: '46px',
                          height: '46px',
                          borderRadius: '12px',
                          background: 'rgba(245, 158, 11, 0.15)',
                          border: '1px solid rgba(245, 158, 11, 0.35)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                        }}
                      >
                        <AlertTriangle size={24} color="#F59E0B" />
                      </div>
                      <div style={{ flex: 1 }}>
                        <div
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.72rem',
                            color: '#FBBF24',
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                            fontWeight: 600,
                          }}
                        >
                          {prediction.status === 'unsupported' || prediction.is_suitable === false
                            ? 'IMAGE SUITABILITY CHECK TRIGGERED'
                            : 'UNCERTAINTY FILTER TRIGGERED'}
                        </div>
                        <h4
                          style={{
                            fontSize: '1.6rem',
                            fontWeight: 800,
                            color: '#FFFFFF',
                            marginTop: '0.2rem',
                            letterSpacing: '-0.01em',
                          }}
                        >
                          {prediction.status === 'unsupported' || prediction.is_suitable === false
                            ? 'Unsupported Image'
                            : 'Uncertain Prediction'}
                        </h4>
                        <p
                          style={{
                            fontSize: '0.98rem',
                            color: 'var(--text-secondary)',
                            lineHeight: 1.55,
                            marginTop: '0.75rem',
                          }}
                        >
                          {prediction.uncertainty_reason ||
                            (prediction.status === 'unsupported' || prediction.is_suitable === false
                              ? 'This image does not appear suitable for satellite land-use classification. Please upload a satellite image containing one of the supported EuroSAT land-use classes.'
                              : 'This image could not be classified confidently. Please upload a clear satellite image containing one of the supported EuroSAT land-use classes.')}
                        </p>
                      </div>
                    </div>

                    {/* Telemetry Breakdown */}
                    <div
                      style={{
                        marginTop: '1.5rem',
                        padding: '1rem',
                        background: 'rgba(4, 9, 22, 0.65)',
                        borderRadius: '10px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        display: 'grid',
                        gridTemplateColumns: 'repeat(3, 1fr)',
                        gap: '0.75rem',
                        fontSize: '0.82rem',
                      }}
                    >
                      {prediction.status === 'unsupported' || prediction.is_suitable === false ? (
                        <>
                          <div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>DOMAIN DISTANCE</div>
                            <div style={{ fontFamily: 'var(--font-mono)', color: '#EF4444', fontWeight: 700, fontSize: '1rem', marginTop: '2px' }}>
                              {prediction.suitability_distance !== undefined && prediction.suitability_distance !== null
                                ? prediction.suitability_distance.toFixed(4)
                                : 'OOD'}
                            </div>
                          </div>
                          <div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>MAX ALLOWED THRESHOLD</div>
                            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)', fontWeight: 700, fontSize: '1rem', marginTop: '2px' }}>
                              {prediction.suitability_threshold !== undefined && prediction.suitability_threshold !== null
                                ? prediction.suitability_threshold.toFixed(2)
                                : '0.38'}
                            </div>
                          </div>
                          <div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>SUITABILITY STAGE</div>
                            <div style={{ color: '#F59E0B', fontWeight: 600, marginTop: '2px', textTransform: 'capitalize' }}>
                              {prediction.suitability_stage_failed ? prediction.suitability_stage_failed.replace('_', ' ') : 'Out of Domain'}
                            </div>
                          </div>
                        </>
                      ) : (
                        <>
                          <div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>OBSERVED CONFIDENCE</div>
                            <div style={{ fontFamily: 'var(--font-mono)', color: '#F59E0B', fontWeight: 700, fontSize: '1rem', marginTop: '2px' }}>
                              {prediction.confidence_percentage}
                            </div>
                          </div>
                          <div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>OPERATING THRESHOLD</div>
                            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)', fontWeight: 700, fontSize: '1rem', marginTop: '2px' }}>
                              {prediction.threshold_used !== undefined ? `${(prediction.threshold_used * 100).toFixed(0)}%` : '60%'}
                            </div>
                          </div>
                          <div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>MODEL EVALUATED</div>
                            <div style={{ color: '#E2E8F0', fontWeight: 600, marginTop: '2px' }}>
                              {prediction.model_display_name}
                            </div>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Action Buttons */}
                    <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="btn btn-primary"
                        style={{ padding: '0.75rem 1.4rem', fontSize: '0.9rem' }}
                        id="btn-upload-another-uncertain"
                      >
                        <UploadCloud size={16} />
                        <span>Upload Another Image</span>
                      </button>

                      <button
                        onClick={() => setShowAllClasses(!showAllClasses)}
                        style={{
                          background: 'transparent',
                          border: '1px solid rgba(255, 255, 255, 0.15)',
                          color: 'var(--text-secondary)',
                          borderRadius: '8px',
                          padding: '0.75rem 1.1rem',
                          fontSize: '0.85rem',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.4rem',
                        }}
                      >
                        <span>{showAllClasses ? 'Hide Diagnostic Output' : 'Inspect Diagnostic Output'}</span>
                        {showAllClasses ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      </button>
                    </div>
                  </div>

                  {/* Diagnostic Softmax Output Drawer */}
                  {showAllClasses && (
                    <div
                      style={{
                        background: 'rgba(4, 9, 22, 0.6)',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        borderRadius: '12px',
                        padding: '1.25rem',
                      }}
                    >
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.85rem' }}>
                        Tentative Softmax distribution (unverified candidate classes):
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.65rem' }}>
                        {Object.entries(prediction.all_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([clsName, prob]) => (
                            <div
                              key={clsName}
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                fontSize: '0.8rem',
                                color: clsName === prediction.predicted_class ? '#F59E0B' : 'var(--text-secondary)',
                              }}
                            >
                              <span>{clsName}</span>
                              <span style={{ fontFamily: 'var(--font-mono)' }}>{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Scientific Disclaimer */}
                  <div
                    style={{
                      padding: '0.75rem 1rem',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid rgba(255, 255, 255, 0.06)',
                      fontSize: '0.75rem',
                      color: 'var(--text-muted)',
                      lineHeight: 1.45,
                    }}
                  >
                    {prediction.disclaimer}
                  </div>
                </div>
              ) : (
                /* =========================================================
                   Confident Prediction Result Dashboard (Accepted)
                   ========================================================= */
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  {/* Main Prediction Banner */}
                  <div
                    style={{
                      background: 'rgba(8, 18, 42, 0.85)',
                      border: `1px solid ${prediction.class_color}55`,
                      borderRadius: '14px',
                      padding: '1.5rem',
                      position: 'relative',
                      overflow: 'hidden',
                      boxShadow: `0 0 25px ${prediction.class_color}22`,
                    }}
                  >
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '6px',
                        height: '100%',
                        backgroundColor: prediction.class_color,
                      }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                          <span
                            style={{
                              fontFamily: 'var(--font-mono)',
                              fontSize: '0.7rem',
                              color: '#34D399',
                              background: 'rgba(52, 211, 153, 0.12)',
                              border: '1px solid rgba(52, 211, 153, 0.3)',
                              padding: '0.15rem 0.45rem',
                              borderRadius: '4px',
                            }}
                          >
                            ACCEPTED PREDICTION
                          </span>
                        </div>
                        <div
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.75rem',
                            color: 'var(--text-muted)',
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                          }}
                        >
                          PREDICTED LAND-USE CLASS
                        </div>
                        <div
                          style={{
                            fontSize: '1.85rem',
                            fontWeight: 800,
                            color: '#FFFFFF',
                            marginTop: '0.2rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.6rem',
                          }}
                        >
                          <span>{prediction.predicted_class}</span>
                          <span
                            style={{
                              width: '12px',
                              height: '12px',
                              borderRadius: '50%',
                              backgroundColor: prediction.class_color,
                              display: 'inline-block',
                            }}
                          />
                        </div>
                      </div>

                      <div style={{ textAlign: 'right' }}>
                        <div
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.75rem',
                            color: 'var(--text-muted)',
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                          }}
                        >
                          CONFIDENCE
                        </div>
                        <div
                          style={{
                            fontFamily: 'var(--font-display)',
                            fontSize: '1.85rem',
                            fontWeight: 800,
                            color: prediction.class_color,
                            marginTop: '0.2rem',
                          }}
                        >
                          {prediction.confidence_percentage}
                        </div>
                      </div>
                    </div>

                    <div
                      style={{
                        marginTop: '0.85rem',
                        fontSize: '0.8rem',
                        color: 'var(--text-secondary)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                      }}
                    >
                      <CheckCircle2 size={14} color="#34D399" />
                      <span>Model: {prediction.model_display_name}</span>
                    </div>
                  </div>

                  {/* Top-3 Candidate Probabilities */}
                  <div>
                    <div
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.75rem',
                        color: 'var(--text-muted)',
                        letterSpacing: '0.08em',
                        textTransform: 'uppercase',
                        marginBottom: '0.85rem',
                      }}
                    >
                      TOP PREDICTIONS
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                      {prediction.top3.map((item) => {
                        const isTop = item.rank === 1;
                        return (
                          <div key={item.class} style={{ position: 'relative' }}>
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                fontSize: '0.85rem',
                                fontWeight: isTop ? 600 : 400,
                                color: isTop ? '#FFFFFF' : 'var(--text-secondary)',
                                marginBottom: '0.35rem',
                              }}
                            >
                              <span>
                                {item.rank}. {item.class}
                              </span>
                              <span style={{ fontFamily: 'var(--font-mono)' }}>{item.percentage}</span>
                            </div>

                            {/* Horizontal Bar */}
                            <div
                              style={{
                                width: '100%',
                                height: '8px',
                                borderRadius: '4px',
                                background: 'rgba(255, 255, 255, 0.07)',
                                overflow: 'hidden',
                              }}
                              role="progressbar"
                              aria-valuenow={Math.round(item.probability * 100)}
                              aria-valuemin={0}
                              aria-valuemax={100}
                            >
                              <div
                                style={{
                                  width: `${Math.min(item.probability * 100, 100)}%`,
                                  height: '100%',
                                  borderRadius: '4px',
                                  background: isTop
                                    ? 'linear-gradient(90deg, #00F0FF, #38BDF8)'
                                    : 'rgba(56, 189, 248, 0.45)',
                                  transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* All 10 Classes Expandable Drawer */}
                  <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '1rem' }}>
                    <button
                      onClick={() => setShowAllClasses(!showAllClasses)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--cyan)',
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        width: '100%',
                      }}
                    >
                      <span>{showAllClasses ? 'Hide full probability distribution' : 'View all 10 class probabilities'}</span>
                      {showAllClasses ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>

                    {showAllClasses && (
                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(2, 1fr)',
                          gap: '0.65rem',
                          marginTop: '1rem',
                          background: 'rgba(4, 9, 22, 0.4)',
                          padding: '1rem',
                          borderRadius: '10px',
                        }}
                      >
                        {Object.entries(prediction.all_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([clsName, prob]) => (
                            <div
                              key={clsName}
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                fontSize: '0.8rem',
                                color: clsName === prediction.predicted_class ? 'var(--cyan)' : 'var(--text-secondary)',
                              }}
                            >
                              <span>{clsName}</span>
                              <span style={{ fontFamily: 'var(--font-mono)' }}>{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    )}
                  </div>

                  {/* Scientific Disclaimer */}
                  <div
                    style={{
                      padding: '0.75rem 1rem',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid rgba(255, 255, 255, 0.06)',
                      fontSize: '0.75rem',
                      color: 'var(--text-muted)',
                      lineHeight: 1.45,
                    }}
                  >
                    {prediction.disclaimer}
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (max-width: 900px) {
          .workspace-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </section>
  );
};
