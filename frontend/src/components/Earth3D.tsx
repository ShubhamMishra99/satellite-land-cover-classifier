import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface Earth3DProps {
  className?: string;
}

/**
 * Procedurally generates an ultra-clean, realistic 2048x1024 equirectangular
 * Earth texture using multi-frequency noise, continents, oceans, and polar ice caps.
 * Completely self-contained: 100% offline reliability with zero network dependencies.
 */
function createProceduralEarthTexture(): THREE.CanvasTexture {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    // Fallback simple blue canvas
    const fallbackCanvas = document.createElement('canvas');
    fallbackCanvas.width = 64;
    fallbackCanvas.height = 64;
    const fCtx = fallbackCanvas.getContext('2d')!;
    fCtx.fillStyle = '#0a2351';
    fCtx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(fallbackCanvas);
  }

  // 1. Deep Ocean Base with subtle depth gradient
  const oceanGrad = ctx.createLinearGradient(0, 0, 0, height);
  oceanGrad.addColorStop(0, '#041026');
  oceanGrad.addColorStop(0.3, '#081d45');
  oceanGrad.addColorStop(0.5, '#0b2659');
  oceanGrad.addColorStop(0.7, '#081d45');
  oceanGrad.addColorStop(1, '#041026');
  ctx.fillStyle = oceanGrad;
  ctx.fillRect(0, 0, width, height);

  // 2. Generate Continents via Multi-frequency Pseudo-Perlin Noise
  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;

  // Simple pseudo-random permutation hash
  const p: number[] = [];
  for (let i = 0; i < 256; i++) p[i] = Math.floor(Math.sin(i * 12.9898 + 78.233) * 43758.5453) & 255;
  const perm = [...p, ...p];

  const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  const lerp = (t: number, a: number, b: number) => a + t * (b - a);
  const grad = (hash: number, x: number, y: number) => {
    const h = hash & 3;
    const u = h < 2 ? x : y;
    const v = h < 2 ? y : x;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
  };

  const noise = (x: number, y: number) => {
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);
    const u = fade(xf);
    const v = fade(yf);
    const a = perm[X] + Y;
    const b = perm[X + 1] + Y;
    return lerp(
      v,
      lerp(u, grad(perm[a], xf, yf), grad(perm[b], xf - 1, yf)),
      lerp(u, grad(perm[a + 1], xf, yf - 1), grad(perm[b + 1], xf - 1, yf - 1))
    );
  };

  for (let y = 0; y < height; y++) {
    const lat = (y / height - 0.5) * Math.PI; // -PI/2 to PI/2
    const cosLat = Math.cos(lat);
    const isPolar = Math.abs(lat) > 1.15; // Polar regions

    for (let x = 0; x < width; x++) {
      const lon = (x / width) * Math.PI * 2;
      const idx = (y * width + x) * 4;

      // 3D sphere coordinate projection for seam-free mapping
      const nx = Math.cos(lon) * cosLat;
      const ny = Math.sin(lat);
      const nz = Math.sin(lon) * cosLat;

      // Fractal Brownian Motion (3 octaves)
      let val =
        noise(nx * 2.2 + 10, ny * 2.2 + 10) * 0.55 +
        noise(nx * 4.4 + 20, ny * 4.4 + 20) * 0.3 +
        noise(nx * 8.8 + 30, ny * 8.8 + 30) * 0.15;

      // Polar ice caps
      if (isPolar) {
        val += (Math.abs(lat) - 1.15) * 1.8;
      }

      if (val > 0.08) {
        // Landmass
        if (isPolar || val > 0.45) {
          // Glacier / snow / mountain peak
          data[idx] = 230;
          data[idx + 1] = 240;
          data[idx + 2] = 250;
        } else if (val > 0.28) {
          // Mountain / highland / arid desert
          data[idx] = 168;
          data[idx + 1] = 135;
          data[idx + 2] = 95;
        } else if (val > 0.16) {
          // Lush forest / dense vegetation
          data[idx] = 34;
          data[idx + 1] = 110;
          data[idx + 2] = 48;
        } else {
          // Coastal grassland / shelf
          data[idx] = 52;
          data[idx + 1] = 138;
          data[idx + 2] = 85;
        }
      } else if (val > 0.02) {
        // Shallow coastal reef water
        data[idx] = 18;
        data[idx + 1] = 68;
        data[idx + 2] = 125;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // 3. Overlay subtle atmospheric swirl streaks on canvas
  ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
  for (let i = 0; i < 20; i++) {
    ctx.beginPath();
    const cy = 200 + (i * 32) % 650;
    const cx = (i * 125) % width;
    ctx.ellipse(cx, cy, 220, 35, 0.2, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

export const Earth3D: React.FC<Earth3DProps> = ({ className = '' }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [hasWebGL, setHasWebGL] = useState(true);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // 1. WebGL Support Verification
    try {
      const testCanvas = document.createElement('canvas');
      const gl = testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl');
      if (!gl) {
        setHasWebGL(false);
        return;
      }
    } catch {
      setHasWebGL(false);
      return;
    }

    // 2. Scene, Camera, and Renderer Setup
    const scene = new THREE.Scene();
    const width = container.clientWidth || 500;
    const height = container.clientHeight || 500;

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
    camera.position.set(0, 0, 5.8);

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    // 3. Earth Globe Mesh
    const earthTexture = createProceduralEarthTexture();
    const earthGeometry = new THREE.SphereGeometry(2, 64, 64);
    const earthMaterial = new THREE.MeshStandardMaterial({
      map: earthTexture,
      roughness: 0.65,
      metalness: 0.1,
    });
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    earthMesh.rotation.z = (23.5 * Math.PI) / 180; // Earth axial tilt
    scene.add(earthMesh);

    // 4. Atmospheric Outer Glow Shell (Layered Additive Blend)
    const atmosphereGeometry = new THREE.SphereGeometry(2.06, 48, 48);
    const atmosphereMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color('#00E5FF'),
      transparent: true,
      opacity: 0.18,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    });
    const atmosphereMesh = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    scene.add(atmosphereMesh);

    // 5. Starfield Background Particles
    const starCount = 1200;
    const starGeometry = new THREE.BufferGeometry();
    const starPositions = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);

    const colorWhite = new THREE.Color('#FFFFFF');
    const colorCyan = new THREE.Color('#38BDF8');
    const colorViolet = new THREE.Color('#818CF8');

    for (let i = 0; i < starCount; i++) {
      const radius = 30 + Math.random() * 40;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      starPositions[i * 3 + 2] = radius * Math.cos(phi);

      const chosenColor = Math.random() > 0.6 ? colorCyan : Math.random() > 0.8 ? colorViolet : colorWhite;
      starColors[i * 3] = chosenColor.r;
      starColors[i * 3 + 1] = chosenColor.g;
      starColors[i * 3 + 2] = chosenColor.b;
    }

    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

    const starMaterial = new THREE.PointsMaterial({
      size: 0.35,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });
    const starField = new THREE.Points(starGeometry, starMaterial);
    scene.add(starField);

    // 6. Orbital Ring System
    const ringCurve = new THREE.EllipseCurve(0, 0, 2.9, 2.9, 0, 2 * Math.PI, false, 0);
    const ringPoints = ringCurve.getPoints(128);
    const ringGeometry = new THREE.BufferGeometry().setFromPoints(
      ringPoints.map((p) => new THREE.Vector3(p.x, 0, p.y))
    );
    const ringMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color('#00F0FF'),
      transparent: true,
      opacity: 0.35,
      linewidth: 1,
    });
    const orbitalRing = new THREE.Line(ringGeometry, ringMaterial);
    orbitalRing.rotation.x = (55 * Math.PI) / 180;
    orbitalRing.rotation.y = (20 * Math.PI) / 180;
    scene.add(orbitalRing);

    // 7. Orbiting Satellite Marker
    const satGroup = new THREE.Group();
    const satBodyGeom = new THREE.BoxGeometry(0.08, 0.08, 0.12);
    const satMaterial = new THREE.MeshBasicMaterial({ color: '#FFFFFF' });
    const satMesh = new THREE.Mesh(satBodyGeom, satMaterial);

    // Solar panels
    const panelGeom = new THREE.PlaneGeometry(0.24, 0.06);
    const panelMat = new THREE.MeshBasicMaterial({ color: '#38BDF8', side: THREE.DoubleSide });
    const panelMesh = new THREE.Mesh(panelGeom, panelMat);
    panelMesh.rotation.y = Math.PI / 2;
    satMesh.add(panelMesh);

    // Beacon light
    const beaconGeom = new THREE.SphereGeometry(0.04, 8, 8);
    const beaconMat = new THREE.MeshBasicMaterial({ color: '#00F0FF' });
    const beacon = new THREE.Mesh(beaconGeom, beaconMat);
    beacon.position.y = 0.08;
    satMesh.add(beacon);

    satGroup.add(satMesh);
    orbitalRing.add(satGroup);

    // 8. Dynamic Lighting (Sun Key Light + Ambient Blue Fill)
    const sunLight = new THREE.DirectionalLight('#FFFFFF', 2.8);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    const spaceAmbient = new THREE.AmbientLight('#071330', 1.1);
    scene.add(spaceAmbient);

    const cyanRimLight = new THREE.DirectionalLight('#00F0FF', 1.4);
    cyanRimLight.position.set(-5, -2, -3);
    scene.add(cyanRimLight);

    // 9. Interactive Drag / Rotation State
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let mouseVelocityX = 0;
    let mouseVelocityY = 0;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      mouseVelocityX = deltaX * 0.005;
      mouseVelocityY = deltaY * 0.005;
      earthMesh.rotation.y += mouseVelocityX;
      earthMesh.rotation.x += mouseVelocityY;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    container.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    // 10. Animation Loop with Reduced-Motion and Visibility Check
    let animationFrameId: number;
    let satAngle = 0;
    let isVisible = true;

    // Check reduced motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Intersection observer to pause rendering when scrolled off-screen
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          isVisible = entry.isIntersecting;
        });
      },
      { threshold: 0.1 }
    );
    observer.observe(container);

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      if (!isVisible) return;

      // Axial rotation
      if (!isDragging) {
        if (!prefersReducedMotion) {
          earthMesh.rotation.y += 0.0015;
          // Apply gentle inertia dampening
          mouseVelocityX *= 0.95;
          mouseVelocityY *= 0.95;
          earthMesh.rotation.y += mouseVelocityX;
          earthMesh.rotation.x += mouseVelocityY;
        }
      }

      // Orbiting satellite progression
      satAngle += 0.012;
      satMesh.position.set(2.9 * Math.cos(satAngle), 0, 2.9 * Math.sin(satAngle));
      satMesh.rotation.y = -satAngle;

      // Subtle starfield twinkling
      starField.rotation.y += 0.0001;

      renderer.render(scene, camera);
    };

    animate();

    // 11. Responsive Window Resize Handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // 12. Cleanup on Unmount
    return () => {
      cancelAnimationFrame(animationFrameId);
      observer.disconnect();
      window.removeEventListener('resize', handleResize);
      container.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }

      // Dispose Three.js GPU memory
      earthGeometry.dispose();
      earthMaterial.dispose();
      earthTexture.dispose();
      atmosphereGeometry.dispose();
      atmosphereMaterial.dispose();
      starGeometry.dispose();
      starMaterial.dispose();
      ringGeometry.dispose();
      ringMaterial.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className={`relative w-full h-full flex items-center justify-center select-none ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{ cursor: isHovered ? 'grab' : 'default' }}
      aria-label="3D Interactive Earth Visualization"
      role="img"
    >
      {!hasWebGL && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-space/80 backdrop-blur-md rounded-2xl border border-cyan-500/20">
          <div className="w-48 h-48 rounded-full border-2 border-cyan-400/40 bg-gradient-to-tr from-cyan-950 via-blue-900 to-indigo-950 flex items-center justify-center shadow-neon relative overflow-hidden mb-4">
            <div className="absolute inset-2 rounded-full border border-cyan-300/20 animate-spin" style={{ animationDuration: '40s' }} />
            <div className="w-32 h-32 rounded-full bg-cyan-500/10 backdrop-blur-sm flex items-center justify-center text-cyan-300 font-mono text-xs">
              EARTH • 3D FALLBACK
            </div>
          </div>
          <p className="text-sm text-slate-300 font-medium">WebGL unavailable on this device</p>
          <p className="text-xs text-slate-500 mt-1">Satellite classification remains fully active</p>
        </div>
      )}
    </div>
  );
};
