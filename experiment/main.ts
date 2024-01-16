import { dataset, Disco, fetchTasks, Task, tf } from '@epfml/discojs-node'

import { startDisco } from '@epfml/disco-server'
import { loadData } from './data'

/**
 * Example of discojs API, we load data, build the appropriate loggers, the disco object
 * and finally start training.
 */
async function runUser(
    url: URL,
    task: Task,
    dataset: dataset.DataSplit
): Promise<void> {
    // Start federated training
    const disco = new Disco(task, { url })
    await disco.fit(dataset)

    // Stop training and disconnect from the remote server
    await disco.close()
}

async function main(): Promise<void> {
    const [server, serverUrl] = await startDisco()

    const tasks = await fetchTasks(serverUrl)

    // Choose your task to train
    const task = tasks.get('wikitext-103')

    if (task === undefined) {
        throw new Error('task not found')
    }

    const dataset = await loadData(task)

    // Add more users to the list to simulate more clients
    await Promise.all([runUser(serverUrl, task, dataset)])

    await new Promise((resolve, reject) => {
        server.once('close', resolve)
        server.close(reject)
    })
}

main().catch(console.error)
