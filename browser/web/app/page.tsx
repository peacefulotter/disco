'use client'
import path from 'path'
import * as disco from '@epfml/discojs-web'
import { useState } from 'react'
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
    const [running, setRunning] = useState<boolean>(false)

    const getSample = async () => {
        const datasplit = await getDatasplit(task)
        const { value } = await getTokenizedSample(datasplit)
        setSample(value)
    }

    const startTraining = async () => {
        // FIXME: url in .env (or fetched from backend?)
        const datasplit = await getDatasplit(task)
        const url = new URL('', 'http://localhost:8000')
        setRunning(true)
        await main(url, task, datasplit).catch(console.error)
        setRunning(false)
    }

    return (
        <main className='flex p-24 gap-8'>
            <div className='flex justify-between items-center gap-8 bg-slate-800 rounded py-4 px-8'>
                <button onClick={getSample} className='bg-slate-700 rounded px-4 py-2'>
                    Get sample
                </button>
                <div>
                    <p>xs: {sample && <b>{JSON.stringify(sample.xs.shape, null, 4)}</b>}</p>
                    <p>ys: {sample && <b>{JSON.stringify(sample.ys.shape, null, 4)}</b>}</p>
                </div>
            </div>
            <div className='flex justify-between items-center gap-8 bg-slate-800 rounded p-4'>
                <button onClick={startTraining} className='bg-slate-700 rounded px-4 py-2'>
                    Train
                </button>
                <div>
                    Running?: <b>{running ? 'true' : 'false'}</b>
                </div>
            </div>
        </main>
    )
}
