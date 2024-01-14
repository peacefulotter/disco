# Prerequisites

-   nvm

    -   install: https://github.com/nvm-sh/nvm#installing-and-updating

-   Bun
    -   install: https://github.com/oven-sh/bun

# Installation

```sh
# install dependencies in discojs-core
cd discojs/discojs-core/
bun install
# install dependencies in discojs-node
cd ../discojs-node/
bun install
# install dependencies in server
cd ../../server/
bun install
# install dependencies in experiment and download + preprocess dataset
cd ../experiment
bun install
# FIXME: npm i -S ../discojs/discojs-node/
./install-wikitext.sh
bun ./preprocess.ts
bun run dev

# -- For web version only:
# install dependencies in discojs-web
cd ../discojs/discojs-web/
bun install
# install dependencies for the browser server and run it
cd ../../browser/server
bun install
bun run dev
# [in a separate terminal] install dependencies for the browser server and run it
cd ../client
nvm use 18    # Node version 18.x or later is required for NextJS
bun install
bun run dev
# Navigate to http://localhost:3000 on your browser of choice and click on "train"
# If you would like to use WebGPU then firefox won't work, please run the following command to run chrome with WebGPU enabled
# (I advise to run this command in a separate terminal tab as well because you will have logs even in detach mode)
google-chrome --enable-unsafe-webgpu --enable-features=Vulkan,UseSkiaRenderer &
# Or from the browser/client/ directory
./chrome-webgpu.sh # equivalent to command above
```

# Running tests

The following will run tests for the web and node text loaders. You need to follow the prerequisites + installation steps before being able to run the tests.

```sh
# Since the following commands will also test the web version,
# the websocket server needs to be running
cd browser/server/
bun --bun socket.ts
# In a new terminal tab, run the tests
cd ../../discojs
bun --bun test text_loader.spec.ts # will run tests with a filename matching text_loader.spec.ts
```

# TODO

1. Benchmark all
2. Try new dataset
3. Try new model

# Future work

1. Disco support for various backends (for WebGPU especially) using `tf.setBackend`, and benchmark on them
2. Support for dedicated tfjs model, which allows custom training loop, e.g. `GPTModel extends Model`. This is partially implemented but not fully (issues in Trainer / TrainerBuilder?)
3. Refactor Task, add generic types
4. QloRA in disco core or at least for GPT
