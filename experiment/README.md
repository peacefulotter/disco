# Installation

```sh
bun install
# FIXME: npm i -S ../discojs/discojs-node/
./install-wikitext.sh
bun ./core/preprocess.ts
bun run dev
```

## Disco.js API

In `example.ts` (line 100) the following function gives an example of the Disco.js API.

To run the clients with decentralized training or local training, set the training scheme to: `TrainingSchemes.DECENTRALIZED` or `TrainingSchemes.LOCAL`, respectively.

```js
async function runUser(url: URL): Promise<void> {
    // Load the data, the dataset must be of type data.Data, see discojs import above.
    const data = await loadData(TASK)

    // Start training
    const disco = new Disco(TASK, { url })
    await disco.fit(data)

    // Stop training and cleanly disconnect from the remote server
    await disco.close()
}
```

To simulate more or less users, simply add more in line 70:

```js
// Add more users to the list to simulate more clients
await Promise.all([runUser(url), runUser(url)])
```
