module.exports = {
  publicPath: process.env.NODE_ENV === 'production' ? '/disco/' : '/',
  configureWebpack: {
    module: { 
      noParse: /wrtc/, 
      rules: [
        {
          test: /\.html$/i,
          loader: "html-loader",
        },
      ], 
    },
    resolve: {
      fallback: {
        crypto: require.resolve('crypto-browserify'),
        path: require.resolve('path-browserify'),
        stream: require.resolve('stream-browserify'),
        assert: require.resolve("assert"),
        zlib: false, // require.resolve("browserify-zlib"),
        https: false, // require.resolve("https-browserify"),
        http: false, // require.resolve("stream-http"),
        timers: false, // require.resolve("timers-browserify"),
        os: false, // require.resolve("os-browserify/browser"),
        constants: false, // require.resolve("constants-browserify"),
        fs: false,
        child_process: false,
        tls: false,
        net: false,
      }
    }
  },
  chainWebpack: (config) => {
    config.plugin('html').tap((args) => {
      args[0].title = 'Disco'
      return args
    })
  }
}
