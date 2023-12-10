import { tf } from '@epfml/discojs-node'
import fs from 'fs'
import { readdir } from 'fs/promises'
import path from 'path'
import { getDataset as getBackboneDataset } from './dataset'
import { Config } from './tfjs-types'
import config from './config'

// For ts-node-esm
import { fileURLToPath } from 'url'
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function getFileStream(file: string) {
    const datasetDir = getDatasetDir(config)

    // blockSize to get an initial full x
    // + batchSize to retrieve a batch from it (each element of the batch is a shift by << n for the nth elt in batch)
    // + 1 for the last y of the batch
    // * 2 because we store the tokens into 2 bytes (16 bites uint)
    // TODO: sure about this?
    const highWaterMark = (config.blockSize + 1) * config.batchSize * 2 // (config.blockSize + config.batchSize + 1) * 2
    console.log('highWaterMark', highWaterMark)

    return fs.createReadStream(path.join(datasetDir, file), {
        highWaterMark, // set this to seq length * 2 because we store uint16,
    })
}

const getDatasetDir = (config: Config) =>
    path.join(__dirname, '../../', 'datasets', config.dataset)

export async function getDatasetFile(split: string) {
    const datasetDir = getDatasetDir(config)
    console.log('Preprocessed dataset located at:', datasetDir)

    const files = await readdir(datasetDir)
    const file = files.filter((f) => f.endsWith('pp') && f.includes(split))[0]
    console.log(
        'Found',
        files.length,
        'files in dataset, selected',
        file,
        'under',
        datasetDir
    )
    return file
}

export function getIteratorDatasetFromFile(
    file: string
): AsyncIterator<Buffer> {
    const getStream = () => {
        const stream = getFileStream(file)
        return {
            stream,
            iter: stream.iterator() as AsyncIterableIterator<Buffer>,
        }
    }
    let { stream, iter } = getStream()
    return {
        next: async () => {
            let sample = await iter.next()
            if (sample.done) {
                stream.close()
                const newStream = getStream()
                stream = newStream.stream
                iter = newStream.iter
                sample = await iter.next()
            }
            return sample
        },
    }
}

export async function getDataset(split: string) {
    const file = await getDatasetFile(split)
    let stream = getIteratorDatasetFromFile(file)
    const requestNext = async () => {
        const { value } = await stream.next()
        return value
    }
    return getBackboneDataset(tf, config, requestNext)
}
