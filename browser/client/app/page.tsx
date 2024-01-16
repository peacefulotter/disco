'use client'
import path from 'path'
import { useEffect, useState } from 'react'
import * as disco from '@epfml/discojs-web'
import { main } from '@/disco/main'
import { getConfig, GPTConfigWithWandb } from '@epfml/gpt-tfjs'

const task = disco.defaultTasks.wikitext.getTask()
const config = task.trainingInformation.modelConfig

// TODO: source: TextSource should be loaded using some generic script using the task definition
// a script to search for dataset files corresponding to a task is defined under experiment/data.ts (getDatasetSource)
// this script should be used here as well (see TODO comment in that file)
const datasetsFolder = path.join('../../experiment', 'datasets', 'wikitext-103')
const source: disco.dataset.TextSource = {
    train: [path.join(datasetsFolder, 'test.tokens')],
    validation: [path.join(datasetsFolder, 'validation.tokens')],
}

const getDatasplit = async (task: disco.Task) => {
    return await new disco.browser.dataset.loader.WebTextLoader(task).loadAll(
        source,
        config
    )
}

const getTokenizedSample = async (datasplit: disco.dataset.DataSplit) => {
    const ds = datasplit.train.dataset as disco.dataset.TokenizedDataset
    const iter = await ds.iterator() // .batch(config.batchSize)
    const { value, done } =
        (await iter.next()) as disco.dataset.TokenizedIterResult
    return { value, done }
}

export default function Home() {
    const [sample, setSample] =
        useState<disco.dataset.BatchedTokenizedTensorSample>()
    const [running, setRunning] = useState<boolean>(false)
    const [config, setConfig] = useState<GPTConfigWithWandb>({})
    const [availableBackends, setAvailableBackends] = useState<string[]>([])
    const [backendName, setBackendName] = useState<string>(() =>
        disco.tf.getBackend()
    )

    useEffect(() => {
        setConfig(getConfig(task.trainingInformation.modelConfig))
        setAvailableBackends(disco.tf.engine().backendNames())
    }, [])

    const getSample = async () => {
        const datasplit = await getDatasplit(task)
        const { value } = await getTokenizedSample(datasplit)
        setSample(value)
    }

    const startTraining = async () => {
        // FIXME: url in .env (or fetched from backend?)
        const datasplit = await getDatasplit(task as disco.Task)
        const url = new URL('', 'http://localhost:8000')
        setRunning(true)
        await main(url, task, datasplit).catch(console.error)
        setRunning(false)
    }

    const setBackend = (backendName: string) => async () => {
        await disco.tf.setBackend(backendName)
        await disco.tf.ready()

        const tfBackend = disco.tf.getBackend()
        if (tfBackend !== backendName) {
            throw new Error('backend not properly set, got: ' + tfBackend)
        }

        console.log('Backend set to:', tfBackend)
        setBackendName(tfBackend)
    }

    return (
        <main className="flex flex-col p-24 gap-8">
            <div className="flex gap-8">
                <div className="flex justify-between items-center gap-8 bg-slate-800 rounded py-4 px-8">
                    <p>Backend availables: {availableBackends.join(', ')}</p>
                    {availableBackends.map((backendName, i) => (
                        <button
                            key={`btn-${i}`}
                            onClick={setBackend(backendName)}
                            className="bg-slate-700 rounded px-4 py-2 capitalize"
                        >
                            {backendName}
                        </button>
                    ))}
                    <p>Backend set to: {backendName}</p>
                </div>
                <div className="flex justify-between items-center gap-8 bg-slate-800 rounded py-4 px-8">
                    <button
                        onClick={getSample}
                        className="bg-slate-700 rounded px-4 py-2"
                    >
                        Get sample
                    </button>
                    <div>
                        <p>
                            xs:{' '}
                            {sample && (
                                <b>
                                    {JSON.stringify(sample.xs.shape, null, 4)}
                                </b>
                            )}
                        </p>
                        <p>
                            ys:{' '}
                            {sample && (
                                <b>
                                    {JSON.stringify(sample.ys.shape, null, 4)}
                                </b>
                            )}
                        </p>
                    </div>
                </div>
                <div className="flex justify-between items-center gap-8 bg-slate-800 rounded p-4">
                    <button
                        onClick={startTraining}
                        className="bg-slate-700 rounded px-4 py-2"
                    >
                        Train
                    </button>
                    <div className="pr-4">
                        Running?: <b>{running ? 'true' : 'false'}</b>
                    </div>
                </div>
            </div>
            <pre className="bg-slate-800 rounded p-4 max-w-min">
                {JSON.stringify(config, undefined, 4)}
            </pre>
        </main>
    )
}
