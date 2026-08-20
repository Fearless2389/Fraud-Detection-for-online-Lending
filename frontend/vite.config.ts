import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * Dev server proxies `/api` and `/health` to the backend so the browser makes
 * same-origin requests locally - no CORS preflight, and no backend URL baked
 * into the bundle.
 *
 * For a deployed build, set VITE_API_BASE_URL and the client calls the remote
 * host directly (the backend's CORS_ORIGINS must then include the frontend
 * origin). Deployment is therefore configuration, not a code change.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backend = env.BACKEND_URL || 'http://127.0.0.1:8000';

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        '/api': { target: backend, changeOrigin: true },
        '/health': { target: backend, changeOrigin: true },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});
