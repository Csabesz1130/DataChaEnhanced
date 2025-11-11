const webpack = require('webpack');
const path = require('path');

module.exports = function override(config, env) {
    // Add fallbacks for Node.js core modules
    config.resolve.fallback = {
        ...config.resolve.fallback,
        "buffer": require.resolve("buffer"),
        "stream": require.resolve("stream-browserify"),
        "assert": require.resolve("assert"),
        "process": require.resolve("process/browser.js"),
    };

    // Fix for axios and other modules that use process/browser
    // Add alias with explicit .js extension - try multiple patterns
    const processBrowserPath = path.resolve(__dirname, "node_modules/process/browser.js");
    config.resolve.alias = {
        ...config.resolve.alias,
        "process/browser": processBrowserPath,
        "process/browser.js": processBrowserPath,
    };

    // Add plugins - use module name, webpack will resolve via alias
    config.plugins = [
        ...config.plugins,
        new webpack.ProvidePlugin({
            Buffer: ['buffer', 'Buffer'],
            process: 'process/browser',
        }),
        // Replace process/browser imports with the actual file - catch all variations
        // Use a function to handle the replacement more reliably
        new webpack.NormalModuleReplacementPlugin(
            /^process\/browser$/,
            (resource) => {
                resource.request = processBrowserPath;
            }
        ),
        new webpack.NormalModuleReplacementPlugin(
            /process\/browser$/,
            processBrowserPath
        ),
        new webpack.NormalModuleReplacementPlugin(
            /process\/browser\.js$/,
            processBrowserPath
        ),
    ];

    // Ensure .js extension is in resolve extensions
    if (!config.resolve.extensions) {
        config.resolve.extensions = ['.js', '.jsx', '.json'];
    } else if (!config.resolve.extensions.includes('.js')) {
        config.resolve.extensions.unshift('.js');
    }

    // Configure module resolution to be more flexible with ESM imports
    config.resolve.conditionNames = ['browser', 'require', 'node', 'default'];
    
    // Try to make webpack less strict about ESM resolution
    // Set fullySpecified to false for all dependency types
    config.resolve.fullySpecified = false;
    
    // Also configure by dependency type
    if (!config.resolve.byDependency) {
        config.resolve.byDependency = {};
    }
    config.resolve.byDependency['esm'] = {
        fullySpecified: false,
    };
    config.resolve.byDependency['import'] = {
        fullySpecified: false,
    };

    return config;
};
