/** @type {import('next').NextConfig} */
const path = require('path');
const NodePolyfillPlugin = require('node-polyfill-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');

const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

const nextConfig = {
  reactStrictMode: true,
  // These files are unhashed (same URL across deploys even when content
  // changes, e.g. an onnxruntime-web version bump), so browsers must
  // revalidate rather than cache indefinitely. This can't be done for
  // anything under /_next/static/ - Next.js hardcodes an unconditional
  // immutable 1-year Cache-Control for that whole path in its static file
  // server (next-server.js), which silently overrides any headers() config
  // for it. That's why these are copied to /runtime/ (served from
  // public/runtime/) instead, where our own Cache-Control below applies.
  async headers() {
    return [
      {
        source: '/runtime/:path*',
        headers: [{ key: 'Cache-Control', value: 'no-cache' }],
      },
    ];
  },
  webpack: (config, {}) => {
    config.resolve.extensions.push('.ts', '.tsx');
    config.resolve.fallback = { fs: false };

    config.plugins.push(
      new NodePolyfillPlugin(),
      new CopyPlugin({
        patterns: [
          {
            from: './node_modules/onnxruntime-web/dist/*.wasm',
            to: path.resolve(__dirname, 'public', 'runtime', '[name][ext]'),
          },
          {
            from: './models',
            to: path.resolve(__dirname, 'public', 'runtime'),
          },
        ],
      })
    );

    return config;
  },
};

const withPWA = require('next-pwa')({
  dest: 'public',
});

module.exports = withBundleAnalyzer(withPWA(nextConfig));

// module.exports = nextConfig
