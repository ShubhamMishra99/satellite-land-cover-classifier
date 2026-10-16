import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/classes': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/samples': 'http://127.0.0.1:8000',
      '/model-metrics': 'http://127.0.0.1:8000',
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
