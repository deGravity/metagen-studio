import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Backend port is configurable so the dev backend can move off a port that's
// already taken (e.g. a vLLM server on the box's default :8000). run.sh exports
// STUDIO_BACKEND_PORT to both uvicorn and this dev server, so the proxy target
// always tracks the backend.
const backendPort = process.env.STUDIO_BACKEND_PORT || '8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // trusted intranet: allow access via any *.csail.mit.edu hostname
    allowedHosts: ['.csail.mit.edu'],
    proxy: {
      '/api': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
});
