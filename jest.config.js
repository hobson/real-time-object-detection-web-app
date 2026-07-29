const nextJest = require('next/jest');

// next/jest handles the SWC transform + next.config.js env/webpack-alias
// wiring for us, so tests compile the same TS/TSX the app itself builds.
const createJestConfig = nextJest({ dir: './' });

const customJestConfig = {
  testEnvironment: 'jest-environment-jsdom',
  testPathIgnorePatterns: ['<rootDir>/node_modules/', '<rootDir>/inference-server/', '<rootDir>/training/'],
};

module.exports = createJestConfig(customJestConfig);
