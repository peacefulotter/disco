'use client'
import path from 'path'
import * as disco from '@epfml/discojs-web'
import { useEffect, useRef, useState } from 'react'
import { main } from '@/disco/main'

const task = disco.defaultTasks.wikitext.getTask()

const config: disco.dataset.loader.TextConfig & { batchSize: number } = {
    blockSize: 3,
    vocabSize: 50257,
    batchSize: task.trainingInformation.batchSize,
}

const datasetsFolder = path.join('../../experiment', 'datasets', 'wikitext-103')

const source: disco.dataset.TextSource = {
    train: [path.join(datasetsFolder, 'test.tokens')],
    validation: [path.join(datasetsFolder, 'validation.tokens')],
}

const getDatasplit = async (task: disco.Task) => {
    return await new disco.browser.dataset.loader.WebTextLoader(task).loadAll(source, config)
}

const getTokenizedSample = async (datasplit: disco.dataset.DataSplit) => {
    const ds = datasplit.train.dataset as disco.dataset.TokenizedDataset
    const iter = await ds.batch(config.batchSize).iterator()
    const { value, done } = (await iter.next()) as disco.dataset.TokenizedIterResult
    return { value, done }
}

export default function Home() {
    const [sample, setSample] = useState<disco.dataset.BatchedTokenizedTensorSample>()

    const initialized = useRef(false)

    useEffect(() => {
        const foo = async () => {
            const datasplit = await getDatasplit(task)
            console.log(datasplit)
            const { value } = await getTokenizedSample(datasplit)
            console.log(value)
            setSample(value)
            const url = new URL('', 'http://localhost:8000')
            main(url, task, datasplit).catch(console.error)
        }
        if (!initialized.current) {
            initialized.current = true
            foo()
        }
    }, [])

    return (
        <main className='flex min-h-screen flex-col items-center justify-between p-24'>
            {sample && JSON.stringify(sample.xs.shape, null, 4)}
            {sample && JSON.stringify(sample.ys.shape, null, 4)}
        </main>
    )
}
