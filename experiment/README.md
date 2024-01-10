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
bun ./core/preprocess.ts
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
cd ../web
bun install
bun run dev
```

# TODO

1. Benchmark all
2. Try new dataset
3. Try new model

# Future work

1. Disco support for various backends (for WebGPU especially) using `tf.setBackend`, and benchmark on them
2. Support for dedicated tfjs model, which allows custom training loop, e.g. `GPTModel extends Model`. This is partially implemented but not fully (issues in Trainer / TrainerBuilder?)
3. QloRA on GPT
