import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
const target = 'http://127.0.0.1:8766';
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1', port: 5173, strictPort: true,
    proxy: {
      '/api': {
        target, changeOrigin: true,
        configure(proxy) {
          proxy.on('proxyReq', (outgoing, incoming) => {
            if (/^http:\/\/(127\.0\.0\.1|localhost):5173$/.test(incoming.headers.origin || '')) {
              outgoing.setHeader('Origin', target);
            }
          });
        },
      },
      '/artifacts': { target, changeOrigin: true },
    },
  },
});
