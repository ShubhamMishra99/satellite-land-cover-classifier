import React from 'react';
import { Satellite, ArrowUp, ShieldAlert, Code2 } from 'lucide-react';

export const Footer: React.FC = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer
      style={{
        background: 'rgba(2, 5, 14, 0.95)',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        padding: '3.5rem 0 2.5rem 0',
        position: 'relative',
        zIndex: 10,
      }}
    >
      <div className="container">
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '2.5rem',
            marginBottom: '3rem',
          }}
        >
          {/* Brand Col */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.85rem' }}>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '8px',
                  background: 'rgba(0, 240, 255, 0.2)',
                  border: '1px solid rgba(0, 240, 255, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Satellite size={16} color="#00F0FF" />
              </div>
              <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.1rem', color: '#FFFFFF' }}>
                EARTH OBSERVER
              </span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '320px' }}>
              Deep learning satellite land-use / land-cover classification platform built for Earth Observation research and academic portfolio defense.
            </p>
          </div>

          {/* Academic Context */}
          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Academic Defense
            </h4>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <li>• Final Year B.Tech / Academic Viva Project</li>
              <li>• Dataset: EuroSAT RGB (27,000 Sentinel-2 Tiles)</li>
              <li>• Test Benchmark: 94.99% Accuracy (ResNet-18)</li>
              <li>• Primary Metric: Macro F1-Score (0.9477)</li>
            </ul>
          </div>

          {/* Model & Architecture */}
          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Core Technologies
            </h4>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <li>• PyTorch 2.14 & Torchvision (Inference Engine)</li>
              <li>• FastAPI (Low-latency Async REST API)</li>
              <li>• React 18, TypeScript, Three.js 3D Visualizer</li>
              <li>• Scikit-learn (Stratified Partition Manager)</li>
            </ul>
          </div>
        </div>

        {/* Closed-set Classification Disclaimer Notice */}
        <div
          style={{
            padding: '1rem 1.25rem',
            borderRadius: '10px',
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            fontSize: '0.78rem',
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.65rem',
            lineHeight: 1.5,
            marginBottom: '2rem',
          }}
        >
          <ShieldAlert size={16} color="#F59E0B" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>
            <strong>Closed-Set Scientific Model Notice:</strong> These models are trained strictly on Sentinel-2 optical Earth Observation imagery across 10 defined EuroSAT classes. Standard photographs of everyday objects (people, animals, arbitrary indoor scenes) will still be projected onto one of the 10 land-cover categories. Softmax confidence alone does not establish whether an arbitrary external photo belongs to the satellite domain.
          </span>
        </div>

        {/* Bottom Bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            paddingTop: '1.5rem',
            borderTop: '1px solid rgba(255, 255, 255, 0.06)',
            fontSize: '0.8rem',
            color: 'var(--text-muted)',
            flexWrap: 'wrap',
            gap: '1rem',
          }}
        >
          <div>© 2026 Earth Observer • Satellite Land-Use Classifier Portfolio</div>
          <button
            onClick={scrollToTop}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: 'transparent',
              border: 'none',
              color: 'var(--cyan)',
              cursor: 'pointer',
              fontSize: '0.8rem',
            }}
          >
            <span>Back to top</span>
            <ArrowUp size={14} />
          </button>
        </div>
      </div>
    </footer>
  );
};
