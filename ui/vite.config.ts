import { fileURLToPath, URL } from 'node:url';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';
import { defineConfig } from 'vite';

const apiProxy = {
  target: process.env.QTS_API_PROXY ?? 'http://localhost:8000',
  changeOrigin: true,
};

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    mode === 'analyze' &&
      visualizer({ filename: 'dist/bundle-report.html', gzipSize: true, template: 'treemap' }),
  ],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  // `/api` goes to the qts API (uvicorn qts_api.main:app, port 8000 by default) in dev and
  // preview, so the UI calls the same origin and needs no CORS. Override with QTS_API_PROXY.
  server: { port: 5173, proxy: { '/api': apiProxy } },
  preview: { proxy: { '/api': apiProxy } },
  build: {
    target: 'es2022',
    sourcemap: true,
    reportCompressedSize: true,
    chunkSizeWarningLimit: 250,
    rolldownOptions: {
      output: {
        // Long-lived vendor chunks cache across deploys; uPlot and d3 stay in the lazy
        // chart chunks that import them.
        codeSplitting: {
          groups: [
            { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
            {
              name: 'router',
              test: /node_modules[\\/](react-router|cookie|set-cookie-parser)[\\/]/,
            },
            { name: 'query', test: /node_modules[\\/]@tanstack[\\/]/ },
          ],
        },
      },
    },
  },
}));
