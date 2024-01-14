import fs from 'fs'
import path from 'path'
import { describe, test, expect } from 'bun:test'
import { encode, decode } from 'gpt-tokenizer/model/text-davinci-003'

import { tf, node, defaultTasks, dataset } from '../..'

/**
 * ================================================
 * Assumes you have followed the installation steps
 * in disco/experiment (see README.md)
 * ================================================
 */

const datasetsFolder = path.join(
    /* @ts-ignore */
    import.meta.dir,
    '../../../../../experiment',
    'datasets',
    'wikitext-103'
)
const source: dataset.TextSource = {
    train: [path.join(datasetsFolder, 'test.tokens')],
    validation: [path.join(datasetsFolder, 'validation.tokens')],
}

const task = defaultTasks.wikitext.getTask()
const config = {
    ...task.trainingInformation.modelConfig,
    blockSize: 16,
    batchSize: 4,
    vocabSize: 50257,
}

const BENCHMARK_ITERATIONS = 1_000
const BENCHMARK_BLOCK_SIZES = [16, 32, 64, 128]

// config: gpt.GPTConfig
const getIterator = async (config: any) => {
    const loaded = await new node.dataset.loader.NodeTextLoader(task).loadAll(
        source,
        config
    )
    const ds = loaded.train.dataset as dataset.TokenizedDataset
    const iter = await ds.batch(config.batchSize).iterator()
    return {
        next: async () => {
            const { value } = (await iter.next()) as dataset.TokenizedIterResult
            const { xs, ys } = value
            const x = xs.flatten()
            const y = (ys.argMax(2) as tf.Tensor2D).flatten() // get indices of max values along last axis
            return { xs, ys, x, y }
        },
    }
}

/**
 * Reads the RAW dataset (not preprocessed) and tokenizes the equivalent of the first batch.
 */
const getRawTokenizedSample = async (
    sampleSize: number,
    tokensLength: number
) => {
    const wikiRaw = fs.createReadStream(path.join(datasetsFolder, 'test'), {
        encoding: 'utf8',
        start: 0,
        end: sampleSize,
    })
    const iter = wikiRaw.iterator()
    const { value: chunk } = await iter.next()
    const tokens = encode(chunk).slice(0, tokensLength)
    return tokens
}

describe('node text loader', () => {
    test('loads a batched sample', async () => {
        const iter = await getIterator(config)
        const { xs, ys, x, y } = await iter.next()

        expect(xs.shape).toEqual([config.batchSize, config.blockSize])
        expect(ys.shape).toEqual([
            config.batchSize,
            config.blockSize,
            config.vocabSize,
        ])
        tf.dispose([xs, ys, x, y])
    })

    test('x without [0] equals y without [-1]', async () => {
        const TEST_SIZE = 10
        const iter = await getIterator(config)
        for (let i = 0; i < TEST_SIZE; i++) {
            const { xs, ys, x, y } = await iter.next()
            const x_arr = await xs.array()
            const y_arr = await (ys.argMax(2) as tf.Tensor2D).array()
            for (let i = 0; i < config.batchSize; i++) {
                // console.log('x=', decode(x_arr[i]).trim())
                // console.log('y=', decode(y_arr[i]).trim())
                expect(x_arr[i].slice(1)).toEqual(y_arr[i].slice(0, -1))
            }
            tf.dispose([xs, ys, x, y])
        }
    })

    test('dataset is tokenized properly', async () => {
        const iter = await getIterator(config)
        const { xs, ys, x, y } = await iter.next()

        /**
         * Flatten the batch by taking the first token in x and the rest in y, since y is x shifted by 1 + 1 token
         * e.g. [a, b, c, d, e, f] -> x = [a, b, c, d, e] and y = [b, c, d, e, f]
         * thus x[0] + y = [a, b, c, d, e, f]
         **/
        const xs_arr = await xs.array()
        const ys_arr = await (ys.argMax(2) as tf.Tensor2D).array() // get indices of max values along last axis
        const sample: number[] = []
        for (let i = 0; i < config.batchSize; i++) {
            sample.push(xs_arr[i][0], ...ys_arr[i])
        }
        const textLength = decode(sample).length
        const tokens = await getRawTokenizedSample(textLength, sample.length)

        expect(sample.length).toBe(tokens.length)
        expect(sample).toEqual(tokens)

        tf.dispose([xs, ys, x, y])
    })

    test(`benchmark ${BENCHMARK_ITERATIONS} iterations for block sizes: ${BENCHMARK_BLOCK_SIZES}`, async () => {
        for (const blockSize of BENCHMARK_BLOCK_SIZES) {
            const iter = await getIterator({ ...config, blockSize })
            const benchmarkStart = Date.now()
            for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
                const { xs, ys, x, y } = await iter.next()
                tf.dispose([xs, ys, x, y])
            }
            const benchmarkEnd = Date.now()
            const ms = benchmarkEnd - benchmarkStart
            console.log(
                `[batchSize=${
                    config.batchSize
                }, blockSize=${blockSize}] Time taken: ${
                    ms / 1000
                }s, time per iteration: ${(ms / BENCHMARK_ITERATIONS).toFixed(
                    3
                )}ms`
            )
        }
    }, 256_000)
})
