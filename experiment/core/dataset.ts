import {
    Config,
    AsyncTokenizedGenerator,
    EncodedDataset,
    TokenizedSample,
} from './tfjs-types'

const tokenizedIterator = (
    tf: any,
    gen: () => AsyncTokenizedGenerator,
    vocabSize: number
): EncodedDataset => {
    return tf.data.generator(gen as any).map((v: any & TokenizedSample) => ({
        x: tf.tensor1d(v.x, 'int32'),
        y: tf.oneHot(v.y, vocabSize),
    }))
}

const toUInt16 = (low: number, high: number) => {
    low &= 0xff
    high &= 0xff
    return (high << 8) | low
}

export async function getDataset(
    tf: any,
    config: Config,
    requestNext: () => Promise<number[]>
) {
    const { vocabSize } = config
    const sampleSize = config.blockSize + 1

    async function* generator(): AsyncTokenizedGenerator {
        while (true) {
            const chunk = await requestNext()
            if (!chunk) break

            // const tokens = []
            // for (let i = 0; i < chunk.length; i += 2) {
            //     const low = chunk[i]
            //     const high = chunk[i + 1]
            //     const token = toUInt16(low, high)
            //     tokens.push(token)
            // }

            // for (let i = 0; i < config.batchSize; i++) {
            //     const x = tokens.slice(i, i + config.blockSize)
            //     const y = tokens.slice(i + 1, i + config.blockSize + 1)
            //     yield { x, y }
            //     // await sleep(1)
            // }

            for (let i = 0; i < config.batchSize; i++) {
                const x = []
                const y = []
                for (let j = 0; j < sampleSize; j++) {
                    const idx = (i * sampleSize + j) * 2
                    const low = chunk[idx]
                    const high = chunk[idx + 1]
                    const token = toUInt16(low, high)
                    if (j < sampleSize - 1) x.push(token)
                    if (j > 0) y.push(token)
                }
                yield { x, y }
            }
        }
    }

    return tokenizedIterator(tf, generator, vocabSize)
}
