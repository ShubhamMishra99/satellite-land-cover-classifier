import React from 'react';
import {
  Wheat,
  Trees,
  Flower2,
  Milestone,
  Factory,
  Compass,
  Sprout,
  Home,
  Waves,
  Ship,
  Grid,
} from 'lucide-react';

interface ClassData {
  name: string;
  color: string;
  icon: React.ReactNode;
  category: string;
  description: string;
  spectralTrait: string;
}

export const ClassesGrid: React.FC = () => {
  const classesList: ClassData[] = [
    {
      name: 'AnnualCrop',
      color: '#E6C229',
      icon: <Wheat size={22} color="#E6C229" />,
      category: 'Agriculture',
      description: 'Rotating seasonal field crops such as wheat, corn, barley, and vegetables.',
      spectralTrait: 'Dynamic seasonal NDVI with soil exposure cycles',
    },
    {
      name: 'Forest',
      color: '#2E7D32',
      icon: <Trees size={22} color="#2E7D32" />,
      category: 'Nature',
      description: 'Dense deciduous, coniferous, and mixed woodland canopies.',
      spectralTrait: 'High NIR reflectance and low red visible absorption',
    },
    {
      name: 'HerbaceousVegetation',
      color: '#81C784',
      icon: <Flower2 size={22} color="#81C784" />,
      category: 'Vegetation',
      description: 'Natural grasslands, unmanaged meadows, moors, and wild herbs.',
      spectralTrait: 'Uniform low-biomass green reflectance signatures',
    },
    {
      name: 'Highway',
      color: '#78909C',
      icon: <Milestone size={22} color="#78909C" />,
      category: 'Infrastructure',
      description: 'Paved motorways, bypasses, major transit corridors, and road intersections.',
      spectralTrait: 'Narrow linear geometric features with low asphalt albedo',
    },
    {
      name: 'Industrial',
      color: '#D81B60',
      icon: <Factory size={22} color="#D81B60" />,
      category: 'Urban',
      description: 'Manufacturing plants, commercial complexes, shipping hubs, and depots.',
      spectralTrait: 'Large flat reflective metallic roofs and paved concrete surfaces',
    },
    {
      name: 'Pasture',
      color: '#AEEA00',
      icon: <Compass size={22} color="#AEEA00" />,
      category: 'Agriculture',
      description: 'Managed grazing ranges, enclosed livestock meadows, and mowed pastures.',
      spectralTrait: 'Bright uniform vegetative reflectance with field boundaries',
    },
    {
      name: 'PermanentCrop',
      color: '#FB8C00',
      icon: <Sprout size={22} color="#FB8C00" />,
      category: 'Agriculture',
      description: 'Perennial orchards, vineyards, citrus groves, and olive plantations.',
      spectralTrait: 'Regular furrow grid texture and permanent woody structures',
    },
    {
      name: 'Residential',
      color: '#8E24AA',
      icon: <Home size={22} color="#8E24AA" />,
      category: 'Urban',
      description: 'Single-family subdivisions, high-density residential housing, and towns.',
      spectralTrait: 'High spatial frequency mix of tiled roofs, lawns, and driveways',
    },
    {
      name: 'River',
      color: '#0288D1',
      icon: <Waves size={22} color="#0288D1" />,
      category: 'Hydrology',
      description: 'Active natural waterways, winding river channels, and inland streams.',
      spectralTrait: 'Continuous curvilinear channels with deep water absorption',
    },
    {
      name: 'SeaLake',
      color: '#01579B',
      icon: <Ship size={22} color="#01579B" />,
      category: 'Hydrology',
      description: 'Open sea coastlines, natural freshwater lakes, reservoirs, and estuaries.',
      spectralTrait: 'Homogeneous dark blue/navy low surface reflectance',
    },
  ];

  return (
    <section
      id="classes"
      style={{
        padding: '5rem 0',
        position: 'relative',
        borderTop: '1px solid rgba(255, 255, 255, 0.06)',
      }}
    >
      <div className="container">
        {/* Section Header */}
        <div style={{ textAlign: 'center', maxWidth: '680px', margin: '0 auto 3.5rem auto' }}>
          <span className="eyebrow-badge" style={{ marginBottom: '0.85rem' }}>
            <Grid size={13} />
            TAXONOMY SPECIFICATIONS
          </span>
          <h2
            style={{
              fontSize: 'clamp(2rem, 3.5vw, 2.75rem)',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              marginBottom: '0.75rem',
            }}
          >
            Supported Land-Use Classes
          </h2>
          <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
            Trained on the standardized EuroSAT RGB dataset covering 10 distinct land-cover categories across European Sentinel-2 satellite passes.
          </p>
        </div>

        {/* 10-Class Card Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))',
            gap: '1.25rem',
          }}
        >
          {classesList.map((cls, idx) => (
            <div
              key={cls.name}
              className="glass-card"
              style={{
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                padding: '1.5rem',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <div
                    style={{
                      width: '42px',
                      height: '42px',
                      borderRadius: '10px',
                      background: `${cls.color}15`,
                      border: `1px solid ${cls.color}35`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    {cls.icon}
                  </div>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.75rem',
                      color: 'var(--text-muted)',
                    }}
                  >
                    #{String(idx + 1).padStart(2, '0')}
                  </span>
                </div>

                <div
                  style={{
                    fontSize: '0.7rem',
                    fontFamily: 'var(--font-mono)',
                    color: cls.color,
                    letterSpacing: '0.06em',
                    textTransform: 'uppercase',
                    marginBottom: '0.2rem',
                  }}
                >
                  {cls.category}
                </div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '0.5rem' }}>
                  {cls.name}
                </h3>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45, marginBottom: '0.85rem' }}>
                  {cls.description}
                </p>
              </div>

              <div
                style={{
                  paddingTop: '0.65rem',
                  borderTop: '1px solid rgba(255, 255, 255, 0.06)',
                  fontSize: '0.72rem',
                  color: 'var(--text-muted)',
                }}
              >
                <strong style={{ color: 'var(--text-secondary)' }}>Spectral Signature:</strong> {cls.spectralTrait}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
