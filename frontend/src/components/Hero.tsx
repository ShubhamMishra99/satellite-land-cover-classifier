import React from 'react';
import { ArrowRight, Sparkles, Layers, Cpu, Orbit } from 'lucide-react';
import { Earth3D } from './Earth3D';

export const Hero: React.FC = () => {
  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section
      id="hero"
      style={{
        position: 'relative',
        minHeight: '100vh',
        paddingTop: 'calc(var(--nav-height) + 2rem)',
        paddingBottom: '4rem',
        display: 'flex',
        alignItems: 'center',
        overflow: 'hidden',
      }}
    >
      <div className="container" style={{ position: 'relative', zIndex: 10 }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 1fr)',
            gap: '3rem',
            alignItems: 'center',
          }}
          className="hero-grid"
        >
          {/* Left Hero Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
            {/* Eyebrow */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span className="eyebrow-badge">
                <Orbit size={13} />
                EARTH OBSERVATION • COMPUTER VISION
              </span>
            </div>

            {/* Headline */}
            <h1
              style={{
                fontSize: 'clamp(2.5rem, 5vw, 4rem)',
                fontWeight: 800,
                letterSpacing: '-0.03em',
                lineHeight: 1.12,
              }}
            >
              Understand Our Planet.{' '}
              <span
                style={{
                  background: 'linear-gradient(135deg, #00F0FF 0%, #38BDF8 50%, #818CF8 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                One Pixel at a Time.
              </span>
            </h1>

            {/* Supporting Description */}
            <p
              style={{
                fontSize: '1.15rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.65,
                maxWidth: '540px',
              }}
            >
              Classify satellite imagery into ten land-use categories using
              deep learning and transfer learning.
            </p>

            {/* Action Buttons */}
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '1rem',
                marginTop: '0.5rem',
              }}
            >
              <button
                onClick={() => scrollTo('classify')}
                className="btn btn-primary"
                id="btn-explore-classify"
              >
                <span>Explore Classification</span>
                <ArrowRight size={17} />
              </button>

              <button
                onClick={() => scrollTo('insights')}
                className="btn btn-secondary"
                id="btn-view-insights"
              >
                <span>View Model Insights</span>
              </button>
            </div>

            {/* Quick Stats Pill Strip */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '1rem',
                marginTop: '1.5rem',
                paddingTop: '1.75rem',
                borderTop: '1px solid rgba(255, 255, 255, 0.08)',
              }}
              className="hero-stats"
            >
              <div>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-white)' }}>
                  10
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Land-Cover Classes
                </div>
              </div>

              <div>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 800, color: 'var(--cyan)' }}>
                  94.99%
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  ResNet-18 Test Acc
                </div>
              </div>

              <div>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 800, color: 'var(--violet)' }}>
                  Sentinel-2
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  EuroSAT RGB Sensor
                </div>
              </div>
            </div>
          </div>

          {/* Right Hero Column: 3D Earth */}
          <div
            style={{
              position: 'relative',
              width: '100%',
              height: '540px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            className="earth-wrapper"
          >
            {/* Ambient Background Radial Glow */}
            <div
              style={{
                position: 'absolute',
                width: '420px',
                height: '420px',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(0, 240, 255, 0.18) 0%, rgba(2, 132, 199, 0.08) 50%, transparent 75%)',
                filter: 'blur(45px)',
                pointerEvents: 'none',
              }}
            />

            {/* 3D Earth Viewport */}
            <Earth3D className="w-full h-full" />

            {/* Decorative Tech Badges */}
            <div
              className="tech-tag"
              style={{
                position: 'absolute',
                top: '12%',
                right: '4%',
                background: 'rgba(8, 16, 36, 0.75)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(0, 240, 255, 0.3)',
                padding: '0.4rem 0.85rem',
                borderRadius: '8px',
                fontSize: '0.7rem',
                fontFamily: 'var(--font-mono)',
                color: 'var(--cyan)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.5)',
              }}
            >
              <span className="status-dot" style={{ width: '6px', height: '6px' }} />
              EARTH OBSERVATION
            </div>

            <div
              className="tech-tag"
              style={{
                position: 'absolute',
                bottom: '18%',
                left: '2%',
                background: 'rgba(8, 16, 36, 0.75)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(129, 140, 248, 0.3)',
                padding: '0.4rem 0.85rem',
                borderRadius: '8px',
                fontSize: '0.7rem',
                fontFamily: 'var(--font-mono)',
                color: 'var(--violet)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.5)',
              }}
            >
              <Layers size={13} />
              LAND-USE ANALYSIS
            </div>

            <div
              className="tech-tag"
              style={{
                position: 'absolute',
                bottom: '8%',
                right: '10%',
                background: 'rgba(8, 16, 36, 0.75)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                padding: '0.4rem 0.85rem',
                borderRadius: '8px',
                fontSize: '0.7rem',
                fontFamily: 'var(--font-mono)',
                color: '#38BDF8',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.5)',
              }}
            >
              <Cpu size={13} />
              AI CLASSIFICATION
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @media (max-width: 960px) {
          .hero-grid {
            grid-template-columns: 1fr !important;
            text-align: center;
          }
          .hero-grid > div:first-child {
            align-items: center;
          }
          .earth-wrapper {
            height: 380px !important;
          }
          .hero-stats {
            justify-content: center;
          }
          .tech-tag {
            display: none !important;
          }
        }
      `}</style>
    </section>
  );
};
