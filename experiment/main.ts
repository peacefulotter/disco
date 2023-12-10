import { data, Disco, fetchTasks, Task } from '@epfml/discojs-node'

import { startServer } from './start_server'
import { loadData } from '@/data'

/**
 * Example of discojs API, we load data, build the appropriate loggers, the disco object
 * and finally start training.
 */
async function runUser(
    url: URL,
    task: Task,
    dataset: data.DataSplit
): Promise<void> {
    // Start federated training
    const disco = new Disco(task, { url })
    await disco.fit(dataset)

    // Stop training and disconnect from the remote server
    await disco.close()
}

async function main(): Promise<void> {
    console.log('starting server')

    const [server, serverUrl] = await startServer()

    console.log('before')
    const tasks = await fetchTasks(serverUrl)
    console.log('len(tasks)', tasks.size, [...tasks.keys()])

    // Choose your task to train
    const task = tasks.get('wikitext-103')
    console.log(task)

    if (task === undefined) {
        throw new Error('task not found')
    }

    const dataset = await loadData(task)

    // Add more users to the list to simulate more clients
    await Promise.all([
        runUser(serverUrl, task, dataset),
        runUser(serverUrl, task, dataset),
    ])

    await new Promise((resolve, reject) => {
        server.once('close', resolve)
        server.close(reject)
    })
}

main().catch(console.error)
