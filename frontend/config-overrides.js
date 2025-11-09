const webpack = require('webpack');
const path = require('path');

module.exports = function override(config, env) {
    // Add fallbacks for Node.js core modules
    config.resolve.fallback = {
        ...config.resolve.fallback,
        "buffer": require.resolve("buffer"),
        "stream": require.resolve("stream-browserify"),
        "assert": require.resolve("assert"),
        "process": require.resolve("process/browser"),
    };

    // Fix for axios and other modules that use process/browser
    // Add alias with explicit .js extension
    config.resolve.alias = {
        ...config.resolve.alias,
        "process/browser": path.resolve(__dirname, "node_modules/process/browser.js"),
    };

    // Add plugins
    config.plugins = [
        ...config.plugins,
        new webpack.ProvidePlugin({
            Buffer: ['buffer', 'Buffer'],
            process: 'process/browser',
        }),
    ];

    return config;
};
