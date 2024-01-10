# Installation

```sh
# install dependencies in discojs-core
cd discojs/discojs-core/
bun install
# install dependencies in discojs-node
cd ../discojs-node/
bun install
# install dependencies in server
cd ../server/
bun install
# install dependencies in experiment and download + preprocess dataset
cd ../experiment
bun install
# FIXME: npm i -S ../discojs/discojs-node/
./install-wikitext.sh
bun ./core/preprocess.ts
bun run dev

# -- For web version only:
cd ../../browser/server
bun install
bun run dev
# in a separate terminal
cd ../../browser/server
bun install
bun run dev
```
