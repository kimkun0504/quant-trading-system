import { fileURLToPath, URL } from 'node:url';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';
import { defineConfig } from 'vite';

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    mode === 'analyze' &&
      visualizer({ filename: 'dist/bundle-report.html', gzipSize: true, template: 'treemap' }),
  ],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: { port: 5173 },
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
