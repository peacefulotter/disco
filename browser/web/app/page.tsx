'use client'
import path from 'path'
import * as disco from '@epfml/discojs-web'

const config: disco.dataset.loader.TextConfig & { batchSize: number } = {
    blockSize: 3,
    batchSize: 2,
    vocabSize: 50257,
}

const datasetsFolder = path.join('../../experiment', 'datasets', 'wikitext-103')

// TODO: don't hardcode this => how to pass a TextSource to websocket
const source: disco.dataset.TextSource = {
    train: [path.join(datasetsFolder, 'test.tokens')],
    validation: [path.join(datasetsFolder, 'validation.tokens')],
}

const getTokenizedSample = async (task: disco.Task) => {
    const loaded = await new disco.browser.dataset.loader.WebTextLoader(task).loadAll(
        source,
        config
    )
    const ds = loaded.train.dataset as disco.dataset.TokenizedDataset
    const iter = await ds.batch(config.batchSize).iterator()
    const { value, done } = (await iter.next()) as disco.dataset.TokenizedIterResult
    return { value, done }
}

export default async function Home() {
    const task = disco.defaultTasks.wikitext.getTask()
    const { value } = await getTokenizedSample(task)
    const { xs, ys } = value
    return (
        <main className='flex min-h-screen flex-col items-center justify-between p-24'>
            {JSON.stringify(xs.shape, null, 4)}
            {JSON.stringify(ys.shape, null, 4)}
        </main>
    )
}
